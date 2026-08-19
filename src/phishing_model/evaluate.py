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


@torch.no_grad()
def run_inference(
    model: MultimodalPhishingClassifier, loader: DataLoader, device: torch.device
) -> dict[str, np.ndarray]:
    model.eval()
    all_ids, all_true, all_pred, all_proba = [], [], [], []
    for batch in loader:
        email_ids = batch.pop("email_id")
        batch_t = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch_t)
        proba = torch.softmax(logits, dim=-1)[:, 1]
        pred = logits.argmax(dim=-1)

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


def evaluate_checkpoint(
    checkpoint_path: Path,
    fusion_type: FusionType,
    df: pd.DataFrame,
    split_name: str,
    device: torch.device | None = None,
    model_name: str | None = None,
    scaler_path: Path | None = None,
) -> dict:
    """`scaler_path`: debe coincidir con el usado al ENTRENAR este checkpoint
    (ver `train.train(..., scaler_path=...)`) -- crítico en LOSO multi-fold,
    donde cada fold tiene sus propios scalers guardados en rutas distintas."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = ModelConfig(fusion_type=fusion_type)

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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    predictions = run_inference(model, loader, device)
    metrics = compute_metrics(predictions["y_true"], predictions["y_pred"], predictions["y_proba"])
    metrics["split"] = split_name
    metrics["fusion_type"] = fusion_type.value

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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    df = pd.read_parquet(Path(args.splits_dir) / f"{args.split}.parquet")

    metrics = evaluate_checkpoint(
        Path(args.checkpoint), FusionType(args.fusion_type), df, args.split, device=device
    )
    print("=" * 70)
    print(f"Evaluación {args.fusion_type} sobre {args.split}")
    print("=" * 70)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{k}: {metrics.get(k)}")


if __name__ == "__main__":
    main()
