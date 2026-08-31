"""
R2.3 — Modelo final optimizado: exportación a ONNX + cuantización
post-entrenamiento (FP32 -> INT8) + análisis de latencia de inferencia.

Sigue el criterio de validación de R2.3 (matriz metodológica de la tesis,
sección 1.3): "Análisis de Latencia e Inferencia: medición del tiempo de
respuesta por muestra frente a datos no vistos, garantizando estabilidad
predictiva" -- por eso este módulo no solo mide latencia, también compara
predicciones FP32 vs. INT8 sobre las MISMAS filas para cuantificar cuánto
(si acaso) degrada la cuantización la exactitud, en vez de asumir que no.

Mismo patrón de verificación que el resto del proyecto (ver
`doc/INFORME_R1.3_R1.4_R2.2_R3.md`, §9): se construye y se prueba de punta a
punta contra el checkpoint de validación disponible; los NÚMEROS de latencia
reales (relevantes para la tesis) deben re-generarse contra el checkpoint
entrenado en GPU -- la latencia de un modelo sub-entrenado es la misma
arquitectura, así que el número en sí es válido de una vez que exista un
checkpoint de producción, no hace falta cambiar el código.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from phishing_model.config import BASE_DIR, FusionType, ModelConfig
from phishing_model.dataset import make_synthetic_batch
from phishing_model.model import MultimodalPhishingClassifier, cargar_pesos
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

ONNX_DIR = BASE_DIR / "data" / "model" / "onnx"
FP32_ONNX_PATH = ONNX_DIR / "model_fp32.onnx"
INT8_ONNX_PATH = ONNX_DIR / "model_int8.onnx"
QUANTIZATION_REPORT_PATH = BASE_DIR / "data" / "reports" / "quantization_latency_report.json"

INPUT_NAMES = [
    "input_ids",
    "attention_mask",
    "structural_continuous",
    "network_continuous",
    "network_categorical",
    "has_structure",
    "has_network",
]
OUTPUT_NAMES = ["logits"]

_INT64_INPUTS = {"input_ids", "attention_mask", "network_categorical"}
_FLOAT32_INPUTS = {"structural_continuous", "network_continuous", "has_structure", "has_network"}


class _ONNXExportWrapper(nn.Module):
    """
    Envuelve `MultimodalPhishingClassifier` para que su forward tome
    argumentos POSICIONALES (requisito de `torch.onnx.export`) en vez del
    dict que usan `train.py`/`evaluate.py` -- el dict es más ergonómico para
    entrenamiento, ONNX necesita una firma de tensores fija y ordenada.
    """

    def __init__(self, model: MultimodalPhishingClassifier) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        structural_continuous: torch.Tensor,
        network_continuous: torch.Tensor,
        network_categorical: torch.Tensor,
        has_structure: torch.Tensor,
        has_network: torch.Tensor,
    ) -> torch.Tensor:
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "structural_continuous": structural_continuous,
            "network_continuous": network_continuous,
            "network_categorical": network_categorical,
            "has_structure": has_structure,
            "has_network": has_network,
        }
        return self.model(batch)


def _batch_to_feed(batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Convierte un batch de tensores PyTorch al dict de numpy arrays con los
    dtypes exactos que espera la sesión ONNX Runtime (int64 para ids/máscaras/
    categóricas, float32 para el resto)."""
    feed: dict[str, np.ndarray] = {}
    for name in INPUT_NAMES:
        arr = batch[name].detach().cpu().numpy()
        if name in _INT64_INPUTS:
            feed[name] = arr.astype(np.int64)
        else:
            feed[name] = arr.astype(np.float32)
    return feed


def export_to_onnx(
    model: MultimodalPhishingClassifier, config: ModelConfig, output_path: Path | None = None
) -> Path:
    """
    Exporta el modelo (ya entrenado, en eval()) a ONNX en FP32.

    NOTA: fusion_type es un atributo FIJO del modelo instanciado (no un input
    del forward), así que `torch.onnx.export` traza únicamente la rama de
    código correspondiente a esa variante -- el grafo ONNX resultante es
    específico de UN `fusion_type`, hay que exportar cada variante por
    separado si se necesitan varias. El dropout de modalidad (activo solo en
    `model.training=True`) tampoco se traza: el modelo está en eval() al
    exportar, por diseño (ver `model._apply_modality_dropout`, que retorna
    temprano si `not self.training`).
    """
    output_path = output_path or FP32_ONNX_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # PyTorch usa internamente un "fast path" de nested tensors dentro de
    # nn.TransformerEncoder/Decoder cuando reciben key_padding_mask (nuestro
    # caso, ver fusion/{modality_encoder,cross_attention}.py) -- ese fast path
    # llama a `aten::_nested_tensor_from_mask`, un operador NO soportado por
    # ONNX (error real encontrado durante el desarrollo). Es un workaround
    # oficialmente documentado por PyTorch para exportación a ONNX de módulos
    # Transformer con máscaras de padding, no un hack propio de este proyecto.
    # No afecta la corrección numérica (el fast path es solo una optimización
    # de rendimiento en CPU/GPU, la matemática de atención es la misma).
    torch.backends.mha.set_fastpath_enabled(False)

    wrapper = _ONNXExportWrapper(model)
    dummy_batch = make_synthetic_batch(batch_size=1, config=config)
    dummy_inputs = tuple(dummy_batch[name] for name in INPUT_NAMES)

    dynamic_axes = {name: {0: "batch_size"} for name in INPUT_NAMES}
    dynamic_axes[OUTPUT_NAMES[0]] = {0: "batch_size"}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(output_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            # dynamo=False (exportador TorchScript "legado", no el nuevo basado en
            # torch.export/dynamo): error real encontrado durante el desarrollo --
            # el exportador dynamo (default en torch 2.x) produce un grafo válido
            # (la exportación en sí funciona), pero las herramientas de inferencia
            # de formas de ONNX Runtime (tanto la básica `onnx.shape_inference` como
            # la simbólica `onnxruntime.quantization.shape_inference`) tienen bugs
            # propios al procesar el estilo de grafo que el exportador dynamo genera
            # para arquitecturas Transformer con ejes de batch dinámicos -- ambas
            # crashean antes de llegar a `quantize_dynamic`. El exportador legado
            # produce un grafo de estilo más tradicional, compatible con el pipeline
            # de cuantización de ONNX Runtime tal como está documentado.
            dynamo=False,
        )
    logger.info("Modelo exportado a ONNX (FP32): %s", output_path)
    return output_path


def quantize_onnx_model(fp32_path: Path, int8_path: Path | None = None) -> Path:
    """
    Cuantización dinámica post-entrenamiento FP32 -> INT8 (ONNX Runtime).

    Se usa cuantización DINÁMICA (no estática) deliberadamente: no requiere
    un dataset de calibración representativo (que introduciría su propia
    complejidad de "¿qué muestra es representativa?"), es la técnica estándar
    de primera instancia para modelos basados en Transformer según la propia
    documentación de ONNX Runtime, y es la que describe Jacob et al. (2018)
    -- ya citado en el Cap. 1.3 de la tesis -- como suficiente para reducir
    la latencia de inferencia sin degradar significativamente la exactitud.

    Preprocesa con `quant_pre_process` (inferencia de formas simbólica, más
    robusta que `onnx.shape_inference` básico) antes de cuantizar -- sin este
    paso, `quantize_dynamic` falla en grafos con ejes dinámicos (batch_size
    dinámico, nuestro caso) con `InferenceError: Inferred shape and existing
    shape differ` (error real encontrado y corregido durante el desarrollo).
    Es el flujo recomendado por la propia documentación de ONNX Runtime, no
    un workaround ad-hoc.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    int8_path = int8_path or INT8_ONNX_PATH
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessed_path = fp32_path.with_name(fp32_path.stem + "_preprocessed.onnx")
    # skip_symbolic_shape=True: la inferencia de formas SIMBÓLICA de ONNX Runtime
    # (herramienta separada de la inferencia de formas básica de ONNX) tiene un
    # bug propio (AssertionError en su manejo de nodos Reshape) sobre el grafo
    # que produce este modelo -- error real encontrado durante el desarrollo,
    # no específico de este proyecto (rastreable a limitaciones conocidas de esa
    # herramienta con grafos de Transformers complejos). La inferencia de formas
    # BÁSICA (skip_onnx_shape=False, sí se ejecuta) es suficiente para que
    # `quantize_dynamic` pueda proceder.
    quant_pre_process(str(fp32_path), str(preprocessed_path), skip_symbolic_shape=True)

    quantize_dynamic(str(preprocessed_path), str(int8_path), weight_type=QuantType.QInt8)
    logger.info("Modelo cuantizado (INT8) guardado en: %s", int8_path)
    return int8_path


def load_real_test_batches(
    config: ModelConfig, n_examples: int, split: str = "test", seed: int = 42
) -> list[dict[str, torch.Tensor]]:
    """
    Carga batches de UNA fila desde el split REAL de evaluación (por defecto
    `test`), usando los scalers ajustados en entrenamiento.

    El IOV de R2.3 exige que el modelo optimizado sea "funcional para realizar
    inferencias sobre nuevos datos NO VISTOS, reportando su tasa de falsos
    negativos observada en el CONJUNTO DE PRUEBA" -- medir latencia y acuerdo
    FP32/INT8 sobre datos sintéticos no satisface ese criterio, por eso esta
    función existe y es la ruta por defecto del pipeline.
    """
    import pandas as pd
    from transformers import AutoTokenizer

    from phishing_model.dataset import MultimodalPhishingDataset, load_scalers
    from phishing_pipeline.config import PROCESSED_DIR

    split_path = PROCESSED_DIR / "splits_group_aware" / f"{split}.parquet"
    if not split_path.exists():
        raise FileNotFoundError(
            f"No existe {split_path}. R2.3 requiere evaluar sobre el conjunto de prueba real; "
            "genera los splits group-aware antes (ver phishing_pipeline.splits.create_group_aware_splits)."
        )

    df = pd.read_parquet(split_path)
    sample = df.sample(n=min(n_examples, len(df)), random_state=seed).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    structural_scaler, network_scaler, clip_bounds = load_scalers()
    dataset = MultimodalPhishingDataset(
        sample,
        tokenizer,
        structural_scaler=structural_scaler,
        network_scaler=network_scaler,
        max_token_length=config.max_token_length,
        fit_scalers=False,
        clip_bounds=clip_bounds,
        # Longitud FIJA: el grafo ONNX exportado traza la dimensión de secuencia
        # como constante (solo el lote es dinámico), así que la inferencia sobre
        # el modelo exportado exige secuencias de `max_token_length`. El relleno
        # dinámico por lote que usa el entrenamiento no aplica aquí.
        pad_to_max_length=True,
    )
    return [{k: v.unsqueeze(0) for k, v in dataset[i].items() if k != "email_id"} for i in range(len(dataset))]


def benchmark_latency(
    onnx_path: Path,
    config: ModelConfig,
    n_samples: int = 50,
    batch_size: int = 1,
    n_warmup: int = 5,
    real_batches: list[dict[str, torch.Tensor]] | None = None,
) -> dict[str, float]:
    """
    Mide latencia de inferencia por muestra (percentiles) de una sesión ONNX Runtime.

    Si se pasan `real_batches` (recomendado, ver `load_real_test_batches`), la
    medición rota entre correos REALES del conjunto de prueba -- más
    representativo que repetir un único batch sintético, porque la longitud
    efectiva del texto y la disponibilidad de modalidades varían entre correos
    reales y eso afecta el trabajo real del grafo.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    if real_batches:
        feeds = [_batch_to_feed(b) for b in real_batches]
        data_source = "test_real"
    else:
        feeds = [_batch_to_feed(make_synthetic_batch(batch_size=batch_size, config=config))]
        data_source = "sintetico"

    for i in range(n_warmup):
        session.run(None, feeds[i % len(feeds)])

    latencies_ms = []
    for i in range(n_samples):
        feed = feeds[i % len(feeds)]
        t0 = time.perf_counter()
        session.run(None, feed)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies_ms)
    return {
        "n_samples": n_samples,
        "batch_size": batch_size,
        "data_source": data_source,
        "mean_ms": round(float(arr.mean()), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "min_ms": round(float(arr.min()), 3),
        "max_ms": round(float(arr.max()), 3),
    }


def compare_fp32_vs_int8_predictions(
    fp32_path: Path, int8_path: Path, sample_batches: list[dict[str, torch.Tensor]]
) -> dict[str, Any]:
    """
    Compara predicciones FP32 vs. INT8 sobre las MISMAS filas -- verifica
    empíricamente que la cuantización no degrada significativamente la
    exactitud predictiva ("garantizando estabilidad predictiva", criterio de
    R2.3), en vez de asumirlo.
    """
    import onnxruntime as ort

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    agreements = 0
    total = 0
    max_logit_diff = 0.0
    preds_fp32_all: list[int] = []
    preds_int8_all: list[int] = []
    labels_all: list[int] = []

    for batch in sample_batches:
        feed = _batch_to_feed(batch)
        out_fp32 = sess_fp32.run(None, feed)[0]
        out_int8 = sess_int8.run(None, feed)[0]

        pred_fp32 = out_fp32.argmax(axis=-1)
        pred_int8 = out_int8.argmax(axis=-1)
        agreements += int((pred_fp32 == pred_int8).sum())
        total += pred_fp32.shape[0]
        max_logit_diff = max(max_logit_diff, float(np.abs(out_fp32 - out_int8).max()))

        preds_fp32_all.extend(pred_fp32.tolist())
        preds_int8_all.extend(pred_int8.tolist())
        if "label" in batch:
            labels_all.extend(batch["label"].detach().cpu().numpy().tolist())

    result: dict[str, Any] = {
        "n_examples": total,
        "prediction_agreement_rate": round(agreements / total, 4) if total else None,
        "max_logit_abs_diff": round(max_logit_diff, 4),
        "note": (
            "prediction_agreement_rate=1.0 significa que INT8 predice EXACTAMENTE lo mismo que "
            "FP32 en todos los ejemplos evaluados -- la cuantización no cambió ninguna decisión "
            "de clasificación en esta muestra."
        ),
    }

    # IOV de R2.3: "reportando su tasa de falsos negativos observada en el
    # conjunto de prueba". Un falso negativo aquí es un correo de phishing
    # (label=1) clasificado como legítimo (pred=0) -- el error operacionalmente
    # más costoso en detección de phishing, por eso el IOV lo pide explícito y
    # no se conforma con accuracy agregada.
    if labels_all:
        result["false_negative_analysis"] = {
            variant: _false_negative_stats(labels_all, preds)
            for variant, preds in (("fp32", preds_fp32_all), ("int8", preds_int8_all))
        }
        result["false_negative_analysis"]["note"] = (
            "Tasa de falsos negativos = FN / (FN + TP), es decir la proporción de correos de "
            "phishing REALES que el modelo dejó pasar como legítimos (equivale a 1 - recall de "
            "la clase phishing). Calculada sobre datos reales del conjunto de prueba, no vistos "
            "en entrenamiento."
        )
    else:
        result["false_negative_analysis"] = {
            "skipped": True,
            "reason": (
                "Los batches evaluados no traen 'label' (probablemente sintéticos) -- el IOV de "
                "R2.3 exige la tasa de FN sobre el conjunto de prueba real; usa "
                "load_real_test_batches() para satisfacerlo."
            ),
        }

    return result


def _false_negative_stats(labels: list[int], preds: list[int]) -> dict[str, Any]:
    """FN, TP y tasa de falsos negativos (= 1 - recall de la clase phishing)."""
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    positives = labels_arr == 1
    n_positives = int(positives.sum())
    false_negatives = int(((labels_arr == 1) & (preds_arr == 0)).sum())
    true_positives = int(((labels_arr == 1) & (preds_arr == 1)).sum())
    return {
        "n_phishing_reales": n_positives,
        "falsos_negativos": false_negatives,
        "verdaderos_positivos": true_positives,
        "tasa_falsos_negativos": round(false_negatives / n_positives, 4) if n_positives else None,
        "recall_phishing": round(true_positives / n_positives, 4) if n_positives else None,
    }


def run_quantization_pipeline(
    checkpoint_path: Path,
    fusion_type: FusionType,
    n_latency_samples: int = 50,
    n_comparison_examples: int = 10,
    is_validation_checkpoint: bool = True,
) -> dict[str, Any]:
    """Corre el pipeline completo: carga checkpoint -> exporta ONNX -> cuantiza -> mide
    latencia FP32 vs INT8 -> compara predicciones -> guarda reporte."""
    config = ModelConfig(fusion_type=fusion_type)
    model = MultimodalPhishingClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cargar_pesos(model, checkpoint["model_state_dict"])
    model.eval()

    fp32_path = export_to_onnx(model, config)
    int8_path = quantize_onnx_model(fp32_path)

    # Datos REALES del conjunto de prueba por defecto (IOV de R2.3). Si no están
    # disponibles (p.ej. splits aún no generados), se degrada a sintéticos pero
    # el reporte lo deja explícito en `data_source`/`false_negative_analysis`,
    # en vez de presentar números sintéticos como si fueran de test real.
    try:
        comparison_batches = load_real_test_batches(config, n_comparison_examples)
        used_real_data = True
    except Exception as exc:
        logger.warning(
            "No se pudieron cargar datos reales de test (%s) -- se degrada a batches sintéticos. "
            "El IOV de R2.3 NO queda satisfecho así; corrige esto antes de reportar resultados.",
            exc,
        )
        comparison_batches = [
            make_synthetic_batch(batch_size=1, config=config) for _ in range(n_comparison_examples)
        ]
        used_real_data = False

    latency_batches = comparison_batches if used_real_data else None
    latency_fp32 = benchmark_latency(
        fp32_path, config, n_samples=n_latency_samples, real_batches=latency_batches
    )
    latency_int8 = benchmark_latency(
        int8_path, config, n_samples=n_latency_samples, real_batches=latency_batches
    )

    prediction_comparison = compare_fp32_vs_int8_predictions(fp32_path, int8_path, comparison_batches)

    fp32_size_mb = fp32_path.stat().st_size / (1024 * 1024)
    int8_size_mb = int8_path.stat().st_size / (1024 * 1024)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "fusion_type": fusion_type.value,
        "IS_VALIDATION_CHECKPOINT_NOT_PRODUCTION": is_validation_checkpoint,
        "evaluated_on_real_test_data": used_real_data,
        "note": (
            "La arquitectura exportada/cuantizada es la misma sin importar qué tan entrenado "
            "esté el checkpoint -- los tamaños de modelo y la latencia relativa FP32 vs. INT8 "
            "son representativos ya con este checkpoint. Solo `prediction_agreement_rate` debe "
            "re-verificarse sobre el checkpoint de producción (un modelo apenas entrenado puede "
            "tener salidas menos estables numéricamente que uno convergido)."
        ),
        "model_size_mb": {
            "fp32": round(fp32_size_mb, 2),
            "int8": round(int8_size_mb, 2),
            "reduction_pct": round((1 - int8_size_mb / fp32_size_mb) * 100, 2) if fp32_size_mb else None,
        },
        "latency_ms": {"fp32": latency_fp32, "int8": latency_int8},
        "latency_speedup_p50": (
            round(latency_fp32["p50_ms"] / latency_int8["p50_ms"], 3) if latency_int8["p50_ms"] else None
        ),
        "prediction_comparison": prediction_comparison,
    }

    QUANTIZATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUANTIZATION_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info("Reporte de cuantización/latencia guardado en %s", QUANTIZATION_REPORT_PATH)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="R2.3 -- exportación ONNX + cuantización + latencia")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fusion-type", type=str, default=FusionType.CROSS_ATTENTION_TOKEN_LEVEL.value)
    parser.add_argument("--n-latency-samples", type=int, default=50)
    args = parser.parse_args()

    report = run_quantization_pipeline(
        Path(args.checkpoint), FusionType(args.fusion_type), n_latency_samples=args.n_latency_samples
    )
    print("=" * 70)
    print("R2.3 -- Cuantización y latencia")
    print("=" * 70)
    print(f"Tamaño modelo: FP32={report['model_size_mb']['fp32']}MB, INT8={report['model_size_mb']['int8']}MB "
          f"(-{report['model_size_mb']['reduction_pct']}%)")
    print(f"Latencia p50: FP32={report['latency_ms']['fp32']['p50_ms']}ms, "
          f"INT8={report['latency_ms']['int8']['p50_ms']}ms (speedup {report['latency_speedup_p50']}x)")
    print(f"Acuerdo de predicciones FP32 vs INT8: {report['prediction_comparison']['prediction_agreement_rate']}")
    print(f"Reporte completo: {QUANTIZATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
