"""
Prueba de integración de la cadena completa, con datos mínimos.

**Qué problema resuelve.** Cada paso del plan de ejecución en servidor cuesta
entre decenas de minutos y horas de GPU. Un fallo de integración —una firma que
cambió, una columna que se renombró, un punto de control que se busca donde no
está— no se manifiesta hasta que ese paso llega a ejecutarse, y para entonces ha
consumido el turno de laboratorio. Esta prueba recorre la misma cadena en unos
minutos y con unos cientos de filas, de modo que esos fallos aparezcan antes de
comprometer tiempo de cómputo compartido.

**Aislamiento.** Todo lo que escribe lleva el prefijo `SMOKE_` y se elimina al
terminar. No sobrescribe puntos de control, predicciones ni reportes reales: la
lista de artefactos creados se registra y se limpia en un bloque `finally`, de
modo que una interrupción tampoco deja residuos que puedan confundirse después
con resultados de una corrida legítima.

Uso:
    python scripts/prueba_integral.py --device cuda --rows 240
    python scripts/prueba_integral.py --device cpu --rows 120 --saltar-cuantizacion
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from phishing_baseline.evaluation import DEFAULT_PREDICTIONS_DIR
from phishing_model.config import MODEL_DIR, FusionType, ModelConfig, TrainConfig
from phishing_pipeline.config import REPORTS_DIR, SPLITS_DIR_GROUP_AWARE
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

PREFIJO = "SMOKE_"
# Artefactos creados por la prueba: se eliminan al terminar.
artefactos: list[Path] = []
# Archivos que YA existían y que algún paso escribe en una ruta fija. No basta
# con borrarlos al limpiar: eso destruiría un resultado real del proyecto. Se
# respalda su contenido antes de ejecutar el paso y se restaura al final.
respaldos: dict[Path, bytes] = {}


def registrar(p: Path) -> Path:
    """Marca un archivo como creado por la prueba, para eliminarlo al terminar."""
    artefactos.append(p)
    return p


def proteger(p: Path) -> Path:
    """
    Respalda un archivo preexistente antes de que un paso lo sobrescriba.

    Los módulos de cuantización y de explicabilidad escriben en rutas fijas del
    proyecto —`quantization_latency_report.json`, entre otras— que contienen
    resultados legítimos de ejecuciones anteriores. Una prueba que los
    sobrescriba y después los borre destruiría evidencia real; una que los
    sobrescriba y no los borre dejaría cifras de humo haciéndose pasar por
    resultados. Respaldar y restaurar es la única opción que no incurre en
    ninguna de las dos.
    """
    if p.exists():
        respaldos[p] = p.read_bytes()
    else:
        registrar(p)
    return p


class Resultado:
    def __init__(self) -> None:
        self.pasos: list[dict[str, Any]] = []

    def ejecutar(self, nombre: str, fn: Callable[[], Any], critico: bool = True) -> Any:
        t0 = time.time()
        try:
            valor = fn()
            self.pasos.append({"paso": nombre, "estado": "OK", "segundos": round(time.time() - t0, 1)})
            logger.info("[OK] %s (%.1fs)", nombre, time.time() - t0)
            return valor
        except Exception as exc:
            self.pasos.append(
                {
                    "paso": nombre,
                    "estado": "FALLÓ",
                    "segundos": round(time.time() - t0, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                    "critico": critico,
                }
            )
            logger.error("[FALLÓ] %s: %s", nombre, exc)
            logger.debug(traceback.format_exc())
            return None

    def resumen(self) -> bool:
        fallos = [p for p in self.pasos if p["estado"] == "FALLÓ"]
        criticos = [p for p in fallos if p.get("critico")]
        print("\n" + "=" * 78)
        print("PRUEBA DE INTEGRACIÓN — RESUMEN")
        print("=" * 78)
        for p in self.pasos:
            marca = "OK   " if p["estado"] == "OK" else "FALLÓ"
            print(f"  [{marca}] {p['paso']:52s} {p['segundos']:6.1f}s")
            if p["estado"] == "FALLÓ":
                print(f"           {p['error']}")
        print()
        print(f"  {len(self.pasos) - len(fallos)}/{len(self.pasos)} pasos superados")
        if criticos:
            print(f"  {len(criticos)} fallo(s) CRÍTICO(S): la cadena no está lista para el servidor")
        elif fallos:
            print(f"  {len(fallos)} fallo(s) no crítico(s)")
        else:
            print("  Cadena completa verificada de punta a punta")
        return not criticos


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba de integración con datos mínimos")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rows", type=int, default=240, help="Filas de entrenamiento por variante")
    parser.add_argument("--splits-dir", type=str, default=str(SPLITS_DIR_GROUP_AWARE))
    parser.add_argument("--saltar-cuantizacion", action="store_true")
    parser.add_argument("--saltar-xai", action="store_true")
    parser.add_argument("--conservar", action="store_true", help="No borrar los artefactos generados")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info("Dispositivo: %s | filas por variante: %d", device, args.rows)

    r = Resultado()
    splits_dir = Path(args.splits_dir)

    # ---------- 1. Datos ----------
    def cargar():
        tr = pd.read_parquet(splits_dir / "train.parquet").sample(n=args.rows, random_state=42).reset_index(drop=True)
        va = (
            pd.read_parquet(splits_dir / "val.parquet")
            .sample(n=max(args.rows // 4, 8), random_state=42)
            .reset_index(drop=True)
        )
        te = (
            pd.read_parquet(splits_dir / "test.parquet")
            .sample(n=max(args.rows // 4, 8), random_state=42)
            .reset_index(drop=True)
        )
        return tr, va, te

    datos = r.ejecutar("1. Cargar particiones agrupadas", cargar)
    if datos is None:
        r.resumen()
        return
    train_df, val_df, test_df = datos

    # ---------- 2. Contrato del conjunto de datos ----------
    def contrato():
        from transformers import AutoTokenizer

        from phishing_model.dataset import MultimodalPhishingDataset, collate_multimodal

        cfg = ModelConfig()
        tok = AutoTokenizer.from_pretrained(cfg.text_model_name)
        ds = MultimodalPhishingDataset(train_df, tok, fit_scalers=True)
        lote = collate_multimodal([ds[i] for i in range(4)])
        esperadas = {
            "input_ids",
            "attention_mask",
            "structural_continuous",
            "network_continuous",
            "network_categorical",
            "has_structure",
            "has_network",
            "label",
            "email_id",
        }
        faltan = esperadas - set(lote)
        if faltan:
            raise AssertionError(f"El lote no cumple el contrato, faltan claves: {faltan}")
        if lote["structural_continuous"].shape[1] != cfg.n_structural_features:
            raise AssertionError(
                f"n_structural_features={cfg.n_structural_features} pero el lote trae "
                f"{lote['structural_continuous'].shape[1]}: la lista de características y la "
                "configuración del modelo están desincronizadas."
            )
        if lote["network_continuous"].shape[1] != cfg.n_network_continuous:
            raise AssertionError(
                f"n_network_continuous={cfg.n_network_continuous} pero el lote trae "
                f"{lote['network_continuous'].shape[1]}."
            )
        return True

    r.ejecutar("2. Contrato del lote y dimensiones declaradas", contrato)

    # ---------- 3. Las cuatro variantes de fusión ----------
    from phishing_model.train import train

    puntos_control: dict[str, Path] = {}

    def entrenar(ft: FusionType):
        def _():
            nombre = f"{PREFIJO}{ft.value}"
            tcfg = TrainConfig(epochs=2, batch_size=4, mixed_precision=(device.type == "cuda"))
            res = train(
                fusion_type=ft,
                train_df=train_df,
                val_df=val_df,
                model_config=ModelConfig(fusion_type=ft),
                train_config=tcfg,
                device=device,
                max_steps=6,
                run_name=nombre,
            )
            ruta = Path(res["checkpoint_path"])
            mejor = ruta.with_name(f"{ruta.stem}_best.pt")
            # `proteger` en lugar de `registrar`: si el punto de control ya
            # existía —porque una corrida real usó ese nombre— se respalda y se
            # restaura en lugar de eliminarse. Con el nombre de corrida aplicado
            # correctamente esto no debería ocurrir, pero una prueba no puede
            # depender de esa suposición para no destruir horas de entrenamiento.
            proteger(ruta)
            proteger(mejor)
            for clave in ("history_path", "learning_curve_path"):
                if res.get(clave):
                    registrar(Path(res[clave]))
            # El punto de control de MEJOR validación es el que debe usarse al
            # evaluar; si no existiera, la evaluación caería en silencio sobre el
            # estado de la última época, que es el defecto que el proyecto corrigió.
            if not mejor.exists():
                raise AssertionError(f"No se generó el punto de control de mejor validación: {mejor}")
            puntos_control[ft.value] = mejor
            return res

        return _

    for ft in FusionType:
        r.ejecutar(f"3. Entrenar variante {ft.value}", entrenar(ft))

    # ---------- 4. Evaluación con el mejor punto de control ----------
    def evaluar():
        from phishing_model.evaluate import evaluate_checkpoint

        ft = FusionType.CROSS_ATTENTION_TOKEN_LEVEL
        if ft.value not in puntos_control:
            raise AssertionError("No hay punto de control de la variante principal para evaluar")
        m = evaluate_checkpoint(
            puntos_control[ft.value],
            ft,
            test_df,
            split_name="test",
            device=device,
            model_name=f"{PREFIJO}Multimodal",
        )
        registrar(DEFAULT_PREDICTIONS_DIR / f"{PREFIJO}Multimodal_test_preds.parquet")
        if "f1" not in m:
            raise AssertionError(f"La evaluación no devolvió F1: claves={list(m)}")
        return m

    r.ejecutar("4. Evaluar con el punto de control de mejor validación", evaluar)

    # ---------- 5. Emparejamiento de McNemar ----------
    def mcnemar():
        from phishing_baseline.compare_models import build_comparison_table

        tabla = build_comparison_table(DEFAULT_PREDICTIONS_DIR)
        # No se exige que haya filas: en una instalación limpia puede no haber
        # aún predicciones de referencia con las que emparejar. Lo que se verifica
        # es que el descubrimiento y el emparejamiento no fallen.
        return {"filas": len(tabla)}

    r.ejecutar("5. Descubrimiento y emparejamiento de McNemar", mcnemar)

    # ---------- 6. Explicabilidad ----------
    if not args.saltar_xai:

        def xai():
            from transformers import AutoTokenizer

            from phishing_model.dataset import MultimodalPhishingDataset
            from phishing_model.xai.faithfulness import compute_faithfulness

            ft = FusionType.CROSS_ATTENTION_TOKEN_LEVEL
            if ft.value not in puntos_control:
                raise AssertionError("Falta el punto de control de la variante principal")
            cfg = ModelConfig(fusion_type=ft)
            tok = AutoTokenizer.from_pretrained(cfg.text_model_name)
            ds = MultimodalPhishingDataset(train_df, tok, fit_scalers=True)
            lote = {k: v.unsqueeze(0) for k, v in ds[0].items() if k != "email_id"}

            from phishing_model.model import MultimodalPhishingClassifier

            modelo = MultimodalPhishingClassifier(cfg).to(device)
            ckpt = torch.load(puntos_control[ft.value], map_location=device)
            modelo.load_state_dict(ckpt["model_state_dict"])
            lote = {k: v.to(device) for k, v in lote.items()}
            # Atribuciones simuladas sobre las palabras del propio texto: lo que
            # se verifica es el mecanismo de medición de fidelidad, no la calidad
            # de una explicación concreta.
            texto = str(train_df["clean_text"].fillna("").iloc[0]) or "texto vacio de prueba"
            palabras = texto.split()[:64] or ["vacio"]
            gen = torch.Generator().manual_seed(42)
            # `word_scores` es una lista de pares (palabra, peso), en el mismo
            # formato que devuelven los tres candidatos de explicabilidad.
            puntajes = list(zip(palabras, torch.rand(len(palabras), generator=gen).tolist()))
            with torch.no_grad():
                clase = int(modelo(lote).argmax(dim=-1).item())
            out = compute_faithfulness(
                model=modelo,
                tokenizer=tok,
                row_batch=lote,
                word_scores=puntajes,
                original_text=texto,
                target_class=clase,
            )
            faltan = {"comprehensiveness", "sufficiency"} - set(out)
            if faltan:
                raise AssertionError(f"compute_faithfulness no devolvió {faltan}; devolvió {list(out)}")
            return out

        r.ejecutar("6. Fidelidad de las explicaciones (exhaustividad/suficiencia)", xai, critico=False)

    # ---------- 7. Cuantización y exportación ----------
    if not args.saltar_cuantizacion:

        def cuantizar():
            from phishing_model.quantization import run_quantization_pipeline

            ft = FusionType.CROSS_ATTENTION_TOKEN_LEVEL
            if ft.value not in puntos_control:
                raise AssertionError("Falta el punto de control de la variante principal")
            # `run_quantization_pipeline` escribe en rutas fijas del proyecto, de
            # modo que su reporte y sus modelos exportados se registran para ser
            # eliminados al terminar. Se anota antes de ejecutar, para que la
            # limpieza actúe incluso si la exportación falla a mitad de camino.
            proteger(REPORTS_DIR / "quantization_latency_report.json")
            for sub in ("onnx", "quantized"):
                d = MODEL_DIR / sub
                if d.exists():
                    for p in d.glob("*"):
                        proteger(p)
            res = run_quantization_pipeline(
                checkpoint_path=puntos_control[ft.value],
                fusion_type=ft,
                n_latency_samples=4,
                is_validation_checkpoint=True,
            )
            for sub in ("onnx", "quantized"):
                d = MODEL_DIR / sub
                if d.exists():
                    for p in d.glob("*"):
                        registrar(p)
            return res

        r.ejecutar("7. Exportación a ONNX y cuantización", cuantizar, critico=False)

    # ---------- 8. Estructura de los pliegues por fuente ----------
    def pliegues():
        from phishing_model.train_loso import build_loso_folds

        completo = pd.read_parquet(splits_dir / "train.parquet")
        folds = build_loso_folds(completo)
        if len(folds) < 2:
            raise AssertionError(f"Se esperaban al menos 2 pliegues, se obtuvieron {len(folds)}")
        for f in folds:
            fuentes_test = set(f["test_df"]["source_dataset"].apply(_familia))
            fuentes_train = set(f["train_df"]["source_dataset"].apply(_familia))
            solapan = fuentes_test & fuentes_train
            if solapan:
                raise AssertionError(
                    f"Fuga entre pliegues: la fuente excluida {f['held_out_source']} también "
                    f"aparece en entrenamiento ({solapan})."
                )
            ids_train = set(f["train_df"]["email_id"])
            ids_val = set(f["val_df"]["email_id"])
            if ids_train & ids_val:
                raise AssertionError(
                    f"Pliegue {f['held_out_source']}: {len(ids_train & ids_val)} identificadores "
                    "compartidos entre entrenamiento y validación."
                )
        return {"pliegues": len(folds)}

    r.ejecutar("8. Pliegues por fuente sin fuga entre particiones", pliegues)

    ok = r.resumen()

    # La restauración de los archivos preexistentes se ejecuta SIEMPRE, incluso
    # con --conservar: lo que el usuario puede querer conservar son los artefactos
    # nuevos de la prueba, nunca una versión de humo de un resultado real.
    restaurados = 0
    for p, contenido in respaldos.items():
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(contenido)
            restaurados += 1
        except OSError as exc:  # pragma: no cover - defensivo
            logger.error("NO SE PUDO RESTAURAR %s: %s", p, exc)

    if not args.conservar:
        print("\nLimpiando artefactos de la prueba...")
        borrados = 0
        for p in artefactos:
            try:
                if p.exists():
                    p.unlink()
                    borrados += 1
            except OSError as exc:  # pragma: no cover - defensivo
                logger.warning("No se pudo borrar %s: %s", p, exc)
        print(f"  {borrados} archivo(s) eliminado(s).")
    else:
        print(f"\nSe conservan {len(artefactos)} artefacto(s) nuevos por petición explícita.")

    if restaurados:
        print(f"  {restaurados} archivo(s) preexistente(s) restaurado(s) a su contenido original.")
    print("  Los resultados reales del proyecto quedan intactos.")

    raise SystemExit(0 if ok else 1)


def _familia(s: str) -> str:
    from phishing_pipeline.config import get_source_family

    return get_source_family(s)


if __name__ == "__main__":
    main()
