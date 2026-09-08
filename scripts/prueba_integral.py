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
import sys
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

# Archivos que YA existían y que algún paso escribe en una ruta fija. No basta
# con borrarlos al limpiar: eso destruiría un resultado real del proyecto. Se
# respalda su contenido antes de ejecutar el paso y se restaura al final.
respaldos: dict[Path, bytes] = {}

# Artefactos creados por la prueba: se eliminan al terminar.
artefactos: list[Path] = []


def registrar(p: Path) -> Path:
    """Marca un archivo como creado por la prueba, para eliminarlo al terminar.

    Un archivo que ya existia y fue respaldado NO se registra: la limpieza corre
    despues de la restauracion, de modo que registrarlo lo borraria justo despues
    de haberlo devuelto a su contenido original. Se comprobo: la prueba dejaba
    vacia la carpeta `data/model/onnx/` y se llevaba por delante los modelos
    exportados, que son el medio de verificacion de R2.3.
    """
    if p in respaldos:
        return p
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


class Omitido(Exception):
    """El paso no puede ejecutarse porque falta un artefacto reproducible.

    No es un fallo: el corpus y sus particiones no se versionan por tamaño, de
    modo que en un clon reciente estos pasos no tienen sobre qué operar hasta
    que se ejecute el pipeline. Contarlos como fallo enmascararía los fallos de
    verdad.
    """


def corpus_del_trabajo():
    """Ruta del corpus sobre el que se reporta el trabajo, o `Omitido`.

    Se prefiere la variante reducida por ser mucho más liviana de leer y
    contener las mismas columnas que estas comprobaciones necesitan.
    """
    from phishing_pipeline.config import PROCESSED_DIR

    for nombre in ("Dataset_Real_slim.parquet", "Dataset_Real.parquet"):
        ruta = PROCESSED_DIR / nombre
        if ruta.exists():
            return ruta
    raise Omitido(
        "no está el corpus en data/Datasets_Procesados/. No se versiona por tamaño; "
        "se reconstruye con `python -m phishing_pipeline.downloaders.correo_real` "
        "y `python -m phishing_pipeline.corpus_real`."
    )


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
        except Omitido as exc:
            self.pasos.append(
                {"paso": nombre, "estado": "OMITIDO", "segundos": round(time.time() - t0, 1),
                 "error": str(exc)}
            )
            logger.warning("[OMITIDO] %s: %s", nombre, exc)
            return None
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
        omitidos = [p for p in self.pasos if p["estado"] == "OMITIDO"]
        for p in self.pasos:
            marca = {"OK": "OK   ", "OMITIDO": "OMIT."}.get(p["estado"], "FALLÓ")
            print(f"  [{marca}] {p['paso']:52s} {p['segundos']:6.1f}s")
            if p["estado"] in ("FALLÓ", "OMITIDO"):
                print(f"           {p['error']}")
        print()
        ejecutados = len(self.pasos) - len(omitidos)
        print(f"  {ejecutados - len(fallos)}/{ejecutados} pasos superados"
              + (f", {len(omitidos)} omitido(s)" if omitidos else ""))
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
    # Se conserva por compatibilidad, pero ya no gobierna la carga: la partición
    # se construye con el mismo código que la cola.
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
        """Muestra de la partición VIGENTE, la misma que construyen los experimentos.

        Antes leía `splits_group_aware/`, que es la partición de la construcción
        de corpus anterior. La prueba que existe para detectar que la cadena se
        desincroniza estaba, ella misma, validándola contra datos de otra época:
        pasaba en verde mientras los experimentos corrían sobre otro corpus. Se
        construye ahora la partición con el mismo código que usa la cola, de modo
        que no puedan divergir.
        """
        raiz = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(raiz / "scripts" / "experimentos"))
        from comun import cargar_corpus, particion_agrupada

        corpus = cargar_corpus()
        particion = particion_agrupada(corpus, agrupar=True)
        logger.info("Corpus vigente: %d filas | %s", len(corpus), particion.resumen())

        def muestra(indices, n):
            return corpus.loc[indices].sample(
                n=min(n, len(indices)), random_state=42).reset_index(drop=True)

        return (muestra(particion.entrenamiento, args.rows),
                muestra(particion.validacion, max(args.rows // 4, 8)),
                muestra(particion.prueba, max(args.rows // 4, 8)))

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
            from phishing_model.config import get_checkpoint_path

            nombre = f"{PREFIJO}{ft.value}"
            # Se decide ANTES de entrenar si estos archivos ya existían. Hacerlo
            # después es inútil: el propio entrenamiento acaba de crearlos, de
            # modo que `proteger` los tomaría por preexistentes, los respaldaría
            # y los restauraría al final, dejando en disco los artefactos que la
            # prueba debía eliminar.
            ruta_prevista = get_checkpoint_path(nombre)
            proteger(ruta_prevista)
            proteger(ruta_prevista.with_name(f"{ruta_prevista.stem}_best.pt"))

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
            if ruta != ruta_prevista:
                # La ruta real no coincide con la prevista: se anota igualmente
                # para no dejar residuo si la convención de nombres cambiara.
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

            from phishing_model.model import MultimodalPhishingClassifier, cargar_pesos

            modelo = MultimodalPhishingClassifier(cfg).to(device)
            ckpt = torch.load(puntos_control[ft.value], map_location=device)
            cargar_pesos(modelo, ckpt["model_state_dict"])
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
            # `rglob` y no `glob`: desde que cada arquitectura exporta a su propia
            # subcarpeta, `glob` devolvia directorios y respaldar un directorio
            # revienta al leerlo como bytes.
            for sub in ("onnx", "quantized"):
                d = MODEL_DIR / sub
                if d.exists():
                    for p in d.rglob("*"):
                        if p.is_file():
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
                    for p in d.rglob("*"):
                        if p.is_file():
                            registrar(p)
            return res

        r.ejecutar("7. Exportación a ONNX y cuantización", cuantizar, critico=False)

    # ---------- 8. Estructura de los pliegues por fuente ----------
    def pliegues():
        from phishing_model.train_loso import build_loso_folds
        from phishing_pipeline.config import PROCESSED_DIR

        # El corpus COMPLETO, no la partición de entrenamiento: la validación
        # por pliegue reparte todas las filas en cada uno —unas colecciones para
        # entrenar y las retenidas íntegras como prueba—, de modo que
        # comprobarlo sobre una partición no reflejaría el uso real y podría
        # ocultar un fallo que solo aparece con el corpus entero.
        completo = pd.read_parquet(corpus_del_trabajo())
        folds = build_loso_folds(completo)
        for f in folds:
            total = len(f["train_df"]) + len(f["val_df"]) + len(f["test_df"])
            if total != len(completo):
                raise AssertionError(
                    f"El pliegue que excluye {f['held_out_source']} suma {total} filas, "
                    f"pero el corpus tiene {len(completo)}: se están perdiendo o duplicando filas."
                )
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

    # ---------- 9. Ausencia de fugas por disponibilidad de campo ----------
    def sin_fugas_por_disponibilidad():
        """
        Comprueba que la PRESENCIA de un campo no revele la etiqueta dentro de su
        propia fuente.

        Es un invariante, no una comprobación puntual. El limpiador de PhishMMF
        extraía las cabeceras de autenticación solo de sus tres ficheros de
        phishing, de modo que `p(phishing | spf presente)` valía 1.0000 sobre 4,478
        correos, y el defecto sobrevivió meses porque nada lo vigilaba. Corregirlo
        resuelve el caso; comprobarlo aquí impide que vuelva por otra vía.
        """
        from phishing_pipeline.auditoria_fugas import auditar

        completo = pd.read_parquet(corpus_del_trabajo())
        informe = auditar(completo)
        if informe["veredicto"] != "SIN_FUGAS":
            detalle = "; ".join(
                f"{f['fuente']}·{f['campo']}={f['mi_presencia_etiqueta_nats']:.4f} nats"
                for f in informe["fugas_detectadas"]
            )
            raise AssertionError(
                f"{len(informe['fugas_detectadas'])} fuga(s) por disponibilidad de campo: {detalle}. "
                "La presencia del campo identifica la clase dentro de su fuente. "
                "Corregir en el limpiador: extraerlo de todas las sub-fuentes o de ninguna."
            )
        return {"campos_auditados": len(informe["campos_auditados"])}

    # No crítico mientras el corpus vigente no se haya regenerado con el limpiador
    # corregido: las cuatro fugas conocidas seguirán apareciendo hasta entonces, y
    # convertirlas en fallo bloqueante impediría usar la prueba para lo demás.
    # Pasar a crítico en cuanto el corpus se reconstruya.
    r.ejecutar("9. Sin fugas por disponibilidad de campo", sin_fugas_por_disponibilidad, critico=False)

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
