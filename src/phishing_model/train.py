"""
Loop de entrenamiento CLI para MultimodalPhishingClassifier.

El mismo script escala de prueba de humo en CPU (batch chico, --max-steps bajo,
--device cpu) a entrenamiento completo en GPU (laboratorio PUCP) -- ver plan,
checkpoints C1 (local) y C2 (GPU). Checkpoint/resume es requisito duro dado el
acceso intermitente al laboratorio (ver plan, Fase B).

Uso típico:
    # Checkpoint C1 (local, CPU, subset chico) -- ver plan
    python -m phishing_model.train --device cpu --max-steps 20 --batch-size 4 \
        --fusion-type cross_attention_token_level --max-rows 500

    # Checkpoint C2 (GPU laboratorio) -- comando completo documentado aparte
    python -m phishing_model.train --device cuda --fusion-type cross_attention_token_level \
        --epochs 3 --batch-size 16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from phishing_model.config import FusionType, ModelConfig, TrainConfig, get_checkpoint_path
from phishing_model.dataset import MultimodalPhishingDataset, save_scalers
from phishing_model.losses import build_loss_fn, compute_class_weights
from phishing_model.model import MultimodalPhishingClassifier
from phishing_pipeline.config import PROCESSED_DIR
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_SPLITS_DIR = PROCESSED_DIR / "splits_group_aware"


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_checkpoint(
    path: Path,
    model: MultimodalPhishingClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    config: ModelConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "fusion_type": config.fusion_type.value,
        },
        path,
    )
    logger.info("Checkpoint guardado: %s (epoch=%d, step=%d)", path, epoch, global_step)


def load_checkpoint(
    path: Path, model: MultimodalPhishingClassifier, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info(
        "Checkpoint cargado: %s (epoch=%d, step=%d)", path, checkpoint["epoch"], checkpoint["global_step"]
    )
    return checkpoint


@torch.no_grad()
def evaluate_loss_accuracy(
    model: MultimodalPhishingClassifier, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        batch_t = {k: v.to(device) for k, v in batch.items() if k != "email_id"}
        logits = model(batch_t)
        loss = loss_fn(logits, batch_t["label"])
        total_loss += loss.item() * len(batch_t["label"])
        total_correct += (logits.argmax(dim=-1) == batch_t["label"]).sum().item()
        total_n += len(batch_t["label"])
    return {"loss": total_loss / max(total_n, 1), "accuracy": total_correct / max(total_n, 1)}


def train(
    fusion_type: FusionType,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
    device: torch.device | None = None,
    max_steps: int | None = None,
    checkpoint_path: Path | None = None,
    scaler_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """
    Entrena el modelo y devuelve un resumen (curva de pérdida, métricas finales).

    `max_steps`, si se pasa, detiene el entrenamiento tras ese número de pasos
    de optimización sin importar cuántas épocas falten -- usado por el
    checkpoint local en CPU (Fase C1) y por pruebas rápidas.

    `scaler_path`, si se pasa, guarda los scalers ajustados en una ruta
    específica en vez de la ruta global por defecto -- IMPRESCINDIBLE en
    entrenamiento multi-fold (p. ej. `train_loso.py`): sin esto, cada fold
    sobrescribiría los scalers del fold anterior en la misma ruta antes de
    que ese fold se evalúe, filtrando el escalado de un fold hacia otro.
    """
    model_config = model_config or ModelConfig(fusion_type=fusion_type)
    model_config.fusion_type = fusion_type
    train_config = train_config or TrainConfig()
    device = device or _select_device("auto")

    torch.manual_seed(train_config.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_config.text_model_name)

    train_dataset = MultimodalPhishingDataset(
        train_df, tokenizer, max_token_length=model_config.max_token_length, fit_scalers=True
    )
    val_dataset = MultimodalPhishingDataset(
        val_df,
        tokenizer,
        structural_scaler=train_dataset.structural_scaler,
        network_scaler=train_dataset.network_scaler,
        max_token_length=model_config.max_token_length,
        fit_scalers=False,
    )
    save_scalers(train_dataset.structural_scaler, train_dataset.network_scaler, path=scaler_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.eval_batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
    )

    model = MultimodalPhishingClassifier(model_config).to(device)

    optimizer = torch.optim.AdamW(
        model.get_optimizer_param_groups(train_config.backbone_lr, train_config.head_lr),
        weight_decay=train_config.weight_decay,
    )

    class_weights = None
    if train_config.use_class_weights or train_config.use_focal_loss:
        labels_tensor = torch.tensor(train_dataset.labels, dtype=torch.long)
        class_weights = compute_class_weights(labels_tensor).to(device)
    loss_fn = build_loss_fn(
        use_class_weights=train_config.use_class_weights,
        use_focal_loss=train_config.use_focal_loss,
        focal_gamma=train_config.focal_loss_gamma,
        class_weights=class_weights,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(train_config.mixed_precision and device.type == "cuda"))

    checkpoint_path = checkpoint_path or get_checkpoint_path(fusion_type.value)
    start_epoch, global_step = 0, 0
    if resume and checkpoint_path.exists():
        ckpt = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch, global_step = ckpt["epoch"], ckpt["global_step"]

    loss_history: list[float] = []
    stopped_early = False

    for epoch in range(start_epoch, train_config.epochs):
        model.train()
        for batch in train_loader:
            batch_t = {k: v.to(device) for k, v in batch.items() if k != "email_id"}

            optimizer.zero_grad()
            with torch.amp.autocast(
                "cuda", enabled=(train_config.mixed_precision and device.type == "cuda")
            ):
                logits = model(batch_t)
                loss = loss_fn(logits, batch_t["label"])

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Pérdida no finita en epoch={epoch} step={global_step}: {loss.item()} "
                    "-- posible explosión de gradiente o bug numérico, deteniendo entrenamiento."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            loss_history.append(loss.item())

            if global_step % train_config.checkpoint_every_n_steps == 0:
                save_checkpoint(checkpoint_path, model, optimizer, epoch, global_step, model_config)

            if max_steps is not None and global_step >= max_steps:
                stopped_early = True
                break

        logger.info("Epoch %d completado (global_step=%d)", epoch, global_step)
        if stopped_early:
            break

    save_checkpoint(checkpoint_path, model, optimizer, start_epoch, global_step, model_config)
    val_metrics = evaluate_loss_accuracy(model, val_loader, loss_fn, device)

    return {
        "fusion_type": fusion_type.value,
        "device": str(device),
        "global_step": global_step,
        "loss_history": loss_history,
        "final_train_loss": loss_history[-1] if loss_history else None,
        "loss_decreased": (loss_history[0] > loss_history[-1]) if len(loss_history) >= 2 else None,
        "val_metrics": val_metrics,
        "checkpoint_path": str(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento del modelo multimodal (R1.4)")
    parser.add_argument("--fusion-type", type=str, default=FusionType.CROSS_ATTENTION_TOKEN_LEVEL.value)
    parser.add_argument("--splits-dir", type=str, default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None, help="Submuestreo de train/val (debug/CPU)")
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    val_df = pd.read_parquet(splits_dir / "val.parquet")
    if args.max_rows:
        train_df = train_df.sample(n=min(args.max_rows, len(train_df)), random_state=42).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(max(args.max_rows // 4, 1), len(val_df)), random_state=42
        ).reset_index(drop=True)

    model_config = ModelConfig(fusion_type=FusionType(args.fusion_type))
    train_config = TrainConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_class_weights=args.use_class_weights,
        use_focal_loss=args.use_focal_loss,
    )

    t0 = time.time()
    result = train(
        FusionType(args.fusion_type),
        train_df,
        val_df,
        model_config=model_config,
        train_config=train_config,
        device=_select_device(args.device),
        max_steps=args.max_steps,
        resume=args.resume,
    )
    elapsed = time.time() - t0

    print("=" * 70)
    print(f"Entrenamiento completado en {elapsed:.1f}s")
    print(f"fusion_type={result['fusion_type']} device={result['device']} steps={result['global_step']}")
    print(f"loss inicial={result['loss_history'][0] if result['loss_history'] else 'N/A'}")
    print(f"loss final={result['final_train_loss']} (bajó: {result['loss_decreased']})")
    print(f"val_metrics={result['val_metrics']}")
    print(f"checkpoint: {result['checkpoint_path']}")


if __name__ == "__main__":
    main()
