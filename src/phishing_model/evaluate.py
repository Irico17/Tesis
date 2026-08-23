"""
Evaluación del modelo multimodal entrenado: holdout y LOSO, con guardado de
predicciones fila por fila (mismo formato que `phishing_baseline.evaluation.
save_predictions`) para poder correr McNemar/t-test contra B1/B2/B3 (Fase D).

Uso:
    python -m phishing_model.evaluate --checkpoint data/model/checkpoints/cross_attention_token_level.pt \
        --fusion-type cross_attention_token_level --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from phishing_model.config import CHECKPOINT_DIR, FusionType, ModelConfig
from phishing_model.dataset import MultimodalPhishingDataset, load_scalers, make_collate_fn
from phishing_model.model import MultimodalPhishingClassifier
from phishing_baseline.evaluation import compute_metrics, save_predictions
from phishing_pipeline.config import PROCESSED_DIR
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_SPLITS_DIR = PROCESSED_DIR / "splits_group_aware"
MODEL_NAME_PREFIX = "MultimodalModel"


def config_desde_checkpoint(checkpoint: dict, fusion_type: FusionType) -> ModelConfig:
    """
    Reconstruye la configuración con la que se entrenó el punto de control.

    Los puntos de control generados antes de que la configuración se persistiera
    completa solo llevan el tipo de fusión; para ellos se recurre a los valores
    por defecto, que son los que aquellas ejecuciones emplearon. Se deja
    constancia en el registro para que la diferencia no pase inadvertida.
    """
    guardada = checkpoint.get("model_config")
    if guardada:
        config = ModelConfig.from_dict(guardada)
        if config.fusion_type != fusion_type:
            raise ValueError(
                f"El punto de control se entrenó con fusion_type={config.fusion_type.value!r} "
                f"pero se solicitó evaluarlo como {fusion_type.value!r}."
            )
        return config
    logger.warning(
        "El punto de control no incluye la configuración del modelo (formato anterior). "
        "Se reconstruye con los valores por defecto, que son los de aquellas ejecuciones."
    )
    return ModelConfig(fusion_type=fusion_type)


def umbral_por_tasa(y_proba: np.ndarray, tasa_supuesta: float) -> float:
    """
    Umbral que hace coincidir la tasa de positivos PREDICHA con `tasa_supuesta`.

    Es el cuantil `1 - tasa_supuesta` de las probabilidades predichas. No requiere
    etiquetas del dominio de destino: requiere un único escalar, la tasa base
    esperada en despliegue.

    Por qué se incorpora. Se verificó que el deterioro entre fuentes es de
    CALIBRACIÓN antes que de discriminación: en el pliegue de Kaggle el área bajo la
    curva ROC es de 0.8051 mientras el modelo declara positivo el 5.1% de los correos
    cuando lo son el 37.3%. El modelo ordena bien y decide mal. Medido sobre los tres
    pliegues con un clasificador textual clásico:

        política de umbral                          F1 medio
        umbral fijo 0.5                              0.6481
        tasa = prevalencia de entrenamiento          0.7472
        tasa = prevalencia real del destino          0.8224
        umbral óptimo por pliegue (oráculo)          0.8450

    Igualar la tasa a la prevalencia real del destino recupera el 89% de la brecha
    que separa el umbral fijo del óptimo. La variante que emplea la prevalencia de
    ENTRENAMIENTO no necesita supuesto alguno y recupera la mitad.

    Se descartaron antes dos alternativas, y conviene dejar constancia para no
    reintentarlas: el algoritmo EM de Saerens et al. (2002) empeora el F1 medio de
    0.6481 a 0.3716 —estima la prevalencia de Kaggle en 0.0000 cuando es 0.3731—
    porque presupone desplazamiento puro de etiqueta y aquí lo que cambia es el
    contenido; el mismo argumento invalida el estimador BBSE de Lipton et al. (2018).
    Y estimar el umbral sobre una fuente de entrenamiento retenida tampoco funciona
    (0.6481 -> 0.6072): el umbral idóneo de una fuente desplazada no transfiere a otra.

    ADVERTENCIA DE USO. Esta política es una propiedad del DESPLIEGUE, no de la
    arquitectura. Si se aplica al modelo propuesto debe aplicarse por igual a todos
    los modelos de referencia antes de compararlos; en caso contrario la comparación
    atribuye a la arquitectura una ganancia que procede de la regla de decisión.
    """
    if not 0.0 < tasa_supuesta < 1.0:
        raise ValueError(f"tasa_supuesta debe estar en (0,1), recibido {tasa_supuesta}")
    return float(np.quantile(y_proba, 1.0 - tasa_supuesta))


@torch.no_grad()
def run_inference(
    model: MultimodalPhishingClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    temperature: float = 1.0,
) -> dict[str, np.ndarray]:
    """
    `temperature` divide los logits antes del softmax (escalado por temperatura,
    Guo et al. 2017). No altera el orden de las probabilidades y por tanto deja
    intacta el área bajo la curva; corrige únicamente su calibración, que es lo
    que el umbral necesita para ser interpretable.

    `threshold` sustituye al `argmax` implícito. Fijarlo en 0.5 equivale al
    comportamiento anterior; hacerlo explícito permite además estimarlo sobre
    validación en lugar de asumirlo, que es lo que exige un despliegue con una
    prevalencia distinta de la del entrenamiento.
    """
    model.eval()
    all_ids, all_true, all_pred, all_proba = [], [], [], []
    for batch in loader:
        email_ids = batch.pop("email_id")
        batch_t = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch_t) / temperature
        proba = torch.softmax(logits, dim=-1)[:, 1]
        pred = (proba >= threshold).long()

        all_ids.extend(email_ids)
        all_true.extend(batch_t["label"].cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_proba.extend(proba.cpu().numpy().tolist())

    return {
        "email_id": np.array(all_ids),
        "y_true": np.array(all_true),
        "y_pred": np.array(all_pred),
        "y_proba": np.array(all_proba),
    }


@torch.no_grad()
def ajustar_temperatura(
    model: MultimodalPhishingClassifier,
    loader: DataLoader,
    device: torch.device,
    rejilla: tuple[float, ...] = tuple(np.arange(0.5, 5.01, 0.05)),
) -> float:
    """
    Estima la temperatura que minimiza la entropía cruzada sobre VALIDACIÓN.

    Se recorre una rejilla en lugar de optimizar por descenso porque el problema
    es unidimensional y convexo en la práctica: la rejilla es más simple, no
    introduce hiperparámetros de optimización y su costo es despreciable frente a
    una pasada de inferencia, que además solo se realiza una vez.

    Debe ajustarse SIEMPRE sobre validación y nunca sobre el conjunto de prueba:
    hacerlo sobre prueba convierte la métrica en una estimación optimista que no
    puede sostenerse en despliegue.
    """
    model.eval()
    logits_acum, etiquetas_acum = [], []
    for batch in loader:
        batch.pop("email_id", None)
        batch_t = {k: v.to(device) for k, v in batch.items()}
        logits_acum.append(model(batch_t).float().cpu())
        etiquetas_acum.append(batch_t["label"].cpu())
    logits = torch.cat(logits_acum)
    etiquetas = torch.cat(etiquetas_acum)

    mejor_t, mejor_perdida = 1.0, float("inf")
    for t in rejilla:
        perdida = torch.nn.functional.cross_entropy(logits / float(t), etiquetas).item()
        if perdida < mejor_perdida:
            mejor_t, mejor_perdida = float(t), perdida
    logger.info("Temperatura estimada sobre validación: %.3f (entropía cruzada %.5f)", mejor_t, mejor_perdida)
    return mejor_t


def evaluate_checkpoint(
    checkpoint_path: Path,
    fusion_type: FusionType,
    df: pd.DataFrame,
    split_name: str,
    device: torch.device | None = None,
    model_name: str | None = None,
    scaler_path: Path | None = None,
    threshold: float = 0.5,
    temperature: float = 1.0,
    prevalencia_referencia: float | None = None,
) -> dict:
    """`prevalencia_referencia`: si se indica, se calcula ADEMÁS un bloque de
    métricas con el umbral que iguala la tasa de positivos predicha a ese valor
    (ver `umbral_por_tasa`). No sustituye a las métricas al umbral declarado: se
    añade como análisis secundario.

    `scaler_path`: debe coincidir con el usado al ENTRENAR este checkpoint
    (ver `train.train(..., scaler_path=...)`) -- crítico en LOSO multi-fold,
    donde cada fold tiene sus propios scalers guardados en rutas distintas."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = config_desde_checkpoint(checkpoint, fusion_type)

    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    structural_scaler, network_scaler = load_scalers(path=scaler_path)
    dataset = MultimodalPhishingDataset(
        df,
        tokenizer,
        structural_scaler=structural_scaler,
        network_scaler=network_scaler,
        max_token_length=config.max_token_length,
        fit_scalers=False,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=make_collate_fn(tokenizer))

    model = MultimodalPhishingClassifier(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    predictions = run_inference(model, loader, device, threshold=threshold, temperature=temperature)
    metrics = compute_metrics(predictions["y_true"], predictions["y_pred"], predictions["y_proba"])
    metrics["split"] = split_name
    metrics["fusion_type"] = fusion_type.value
    metrics["threshold"] = threshold
    metrics["temperature"] = temperature
    # Cuánto emplea el modelo las modalidades no textuales, aprendido durante el
    # entrenamiento. Se registra junto a las métricas para que la comparación entre
    # variantes disponga de la magnitud además del resultado.
    metrics["modality_gate"] = round(model.contribucion_modal(), 6)

    if prevalencia_referencia is not None:
        u = umbral_por_tasa(predictions["y_proba"], prevalencia_referencia)
        m2 = compute_metrics(
            predictions["y_true"], (predictions["y_proba"] >= u).astype(int), predictions["y_proba"]
        )
        m2["threshold"] = round(u, 6)
        m2["tasa_supuesta"] = round(float(prevalencia_referencia), 6)
        m2["politica"] = "tasa_igualada_a_prevalencia_de_entrenamiento"
        metrics["metrics_tasa_igualada"] = m2

    resolved_model_name = model_name or f"{MODEL_NAME_PREFIX}_{fusion_type.value}"
    pred_path = save_predictions(
        predictions["email_id"],
        predictions["y_true"],
        predictions["y_pred"],
        predictions["y_proba"],
        resolved_model_name,
        split_name,
    )
    logger.info("Predicciones guardadas: %s", pred_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación del modelo multimodal (R1.4/R2.2)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fusion-type", type=str, default=FusionType.CROSS_ATTENTION_TOKEN_LEVEL.value)
    parser.add_argument("--splits-dir", type=str, default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Umbral de decisión sobre la probabilidad de la clase phishing. 0.5 reproduce "
            "el comportamiento del argmax. Estímese sobre validación, nunca sobre prueba."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Temperatura de calibración: divide los logits antes del softmax. No altera el "
            "orden de las probabilidades ni el área bajo la curva, solo su calibración."
        ),
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=None,
        help=(
            "Escaladores con los que se entrenó el punto de control. Debe coincidir con el "
            "`--scaler-path` de la corrida correspondiente: evaluar con escaladores ajustados "
            "sobre otros datos distorsiona las características de entrada sin producir error alguno"
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    df = pd.read_parquet(Path(args.splits_dir) / f"{args.split}.parquet")

    metrics = evaluate_checkpoint(
        Path(args.checkpoint),
        FusionType(args.fusion_type),
        df,
        args.split,
        device=device,
        scaler_path=Path(args.scaler_path) if args.scaler_path else None,
        threshold=args.threshold,
        temperature=args.temperature,
    )
    print("=" * 70)
    print(f"Evaluación {args.fusion_type} sobre {args.split}")
    print("=" * 70)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "threshold", "temperature", "modality_gate"):
        print(f"{k}: {metrics.get(k)}")


if __name__ == "__main__":
    main()
