"""
E6 · Modelo final optimizado (R2.3)

No lo cubre ninguno de los cinco experimentos anteriores, que comparan
arquitecturas. Este responde al indicador de R2.3, que es de otra naturaleza: si
el modelo, una vez elegido, es utilizable en el punto de integración declarado
--una pasarela de correo-- y con qué tasa de falsos negativos.

Tres mediciones:

1. **Cuantización de FP32 a INT8** sobre el punto de control del modelo ganador.
2. **Latencia por muestra**, FP32 frente a INT8, que es la cifra que decide si
   el sistema es viable en línea.
3. **Tasa de falsos negativos** sobre datos de prueba reales, que es la que el
   indicador exige informar de forma expresa, sin presuponer que mejore respecto
   de la línea base.

El modelo se toma del ganador de E1 y E2. Cuantizar una variante que no es la
propuesta mediría la latencia de algo que no se va a desplegar.

    python scripts/experimentos/e6_modelo_optimizado.py [--variante ...]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import BASE, emitir


# Se miden TODAS las arquitecturas, no solo una elegida. La latencia depende de
# la arquitectura --la mezcla de expertos evalua cuatro expertos, la concatenacion
# tardia ninguno-- y presentar la de una sola obligaria a elegir ganadora antes de
# saber cuanto cuesta cada una, que es justo lo que R2.3 pregunta.
ARQUITECTURAS = ("atencion_cruzada_token", "atencion_cruzada_modalidad",
                 "mezcla_de_expertos", "fusor_mlp", "concatenacion_tardia",
                 "solo_texto")
REFERENCIA = "atencion_cruzada_token"


def localizar_punto(variante: str, semilla: int):
    """Punto de control ya entrenado de una variante, mirando E4, E2 y E1.

    Se prueba en ese orden porque E4 entrena sobre el corpus completo, que es el
    escenario de despliegue que R2.3 describe; E2 y E1 quedan como respaldo. El
    sufijo `_sc` --sin centinela-- forma parte del nombre desde que se arreglaron
    las colisiones, y E1, E2 y E4 entrenan sin centinela: sin contemplarlo aqui,
    E6 no encontraria ningun punto de control tras una reejecucion limpia.
    """
    d = BASE / "data" / "model" / "checkpoints"
    for exp, suf in (("e4", "_sc_agr"), ("e4", "_agr"), ("e2", "_sc"), ("e2", ""),
                     ("e1", "_sc"), ("e1", "")):
        for cola in ("_best.pt", ".pt"):
            c = d / f"{exp}_{variante}_s{semilla}{suf}{cola}"
            if c.exists():
                return c
    return None


def tasa_de_falsos_negativos(punto: Path) -> dict:
    """Falsos negativos sobre el CONJUNTO DE PRUEBA COMPLETO.

    El pipeline de cuantizacion trae una suya, pero sale de las diez muestras con
    que compara FP32 e INT8 --de las que solo cuatro eran phishing-- y una tasa
    sobre cuatro casos no es una tasa. El indicador de R2.3 exige informarla.
    """
    import pandas as pd

    from phishing_baseline.evaluation import DEFAULT_PREDICTIONS_DIR

    corrida = punto.name.replace("_best.pt", "").replace(".pt", "")
    hallados = sorted(DEFAULT_PREDICTIONS_DIR.glob(f"{corrida}_*_preds.parquet"))
    if not hallados:
        return {"tasa_falsos_negativos": None,
                "nota": f"no se hallaron predicciones para {corrida}"}
    pred = pd.read_parquet(hallados[0])
    reales = int((pred.y_true == 1).sum())
    fallados = int(((pred.y_true == 1) & (pred.y_pred == 0)).sum())
    return {"origen": hallados[0].name, "n_phishing_reales": reales,
            "falsos_negativos": fallados,
            "tasa_falsos_negativos": round(fallados / reales, 6) if reales else None,
            "recall_phishing": round(1 - fallados / reales, 6) if reales else None}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--muestras-latencia", type=int, default=50)
    args = p.parse_args()

    sys.path.insert(0, str(BASE / "src"))
    from phishing_model.config import FusionType
    from phishing_model.quantization import run_quantization_pipeline

    from entrenar import VARIANTES, configuracion

    medidas: dict[str, dict] = {}
    ausentes: list[str] = []
    for variante in ARQUITECTURAS:
        punto = localizar_punto(variante, args.semilla)
        if punto is None:
            ausentes.append(variante)
            print(f"  {variante:30s} SIN punto de control, se omite")
            continue
        modelo_cfg, _ = configuracion(variante, args.semilla)
        res = run_quantization_pipeline(
            checkpoint_path=punto,
            fusion_type=FusionType(VARIANTES[variante]["fusion_type"]),
            n_latency_samples=args.muestras_latencia,
            model_config=modelo_cfg,
        )
        lat, tam = res.get("latency_ms", {}), res.get("model_size_mb", {})
        medidas[variante] = {
            "punto_de_control": punto.name,
            "tamano_mb": tam,
            "latencia_ms": lat,
            "aceleracion_p50": res.get("latency_speedup_p50"),
            "acuerdo_fp32_int8": res.get("prediction_comparison", {}).get(
                "prediction_agreement_rate"),
            "falsos_negativos_prueba_completa": tasa_de_falsos_negativos(punto),
        }
        print(f"  {variante:30s} FP32 {lat.get('fp32', {}).get('mean_ms', '?')} ms  "
              f"INT8 {lat.get('int8', {}).get('mean_ms', '?')} ms  "
              f"{tam.get('fp32', '?')}->{tam.get('int8', '?')} MB")

    informe = {
        "experimento": "E6",
        "resultado_asociado": "R2.3",
        "pregunta": ("Es utilizable cada arquitectura en el punto de integracion "
                     "declarado, y con que tasa de falsos negativos?"),
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "arquitecturas_medidas": list(medidas),
        "arquitecturas_sin_punto_de_control": ausentes,
        "arquitectura_de_referencia": REFERENCIA,
        "medidas": medidas,
        "nota_sobre_la_tasa_de_falsos_negativos": (
            "Se informa sin presuponer mejora respecto de la línea base, conforme "
            "al enfoque de caracterización adoptado en R2.2, y se calcula sobre el "
            "conjunto de prueba completo y no sobre las muestras de comparación."),
        "nota_sobre_la_referencia": (
            "Se miden todas las arquitecturas. La referencia es la que el trabajo "
            "propone; no se selecciona ninguna por desempeno."),
    }
    emitir("e6", informe)
    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
