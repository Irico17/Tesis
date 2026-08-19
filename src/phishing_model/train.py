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
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from phishing_model.config import FusionType, ModelConfig, TrainConfig, get_checkpoint_path
from phishing_model.dataset import (
    LengthGroupedSampler,
    MultimodalPhishingDataset,
    make_collate_fn,
    save_scalers,
)
from phishing_model.losses import build_loss_fn, compute_class_weights
from phishing_model.model import MultimodalPhishingClassifier
from phishing_pipeline.config import PROCESSED_DIR, REPORTS_DIR
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_SPLITS_DIR = PROCESSED_DIR / "splits_group_aware"

# Artefactos de evidencia del medio de verificación de R1.4 ("logs de ejecución
# y curvas de aprendizaje"): el JSON es el artefacto CRÍTICO (fuente de verdad,
# regenerable a figura); el PNG es derivado y siempre reconstruible desde él.
TRAINING_LOGS_DIR = REPORTS_DIR / "training_logs"
FIGURES_DIR = REPORTS_DIR / "figures"


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _sanitize_run_name(run_name: str) -> str:
    """Normaliza `run_name` para usarlo como nombre de archivo en cualquier SO."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", run_name).strip("_") or "run"


def get_history_path(run_name: str) -> Path:
    """Ruta del JSON de historial de entrenamiento para una corrida nombrada."""
    return TRAINING_LOGS_DIR / f"{_sanitize_run_name(run_name)}_history.json"


def get_learning_curve_path(run_name: str) -> Path:
    """Ruta de la figura de curvas de aprendizaje para una corrida nombrada."""
    return FIGURES_DIR / f"learning_curve_{_sanitize_run_name(run_name)}.png"


def detect_overfitting(epoch_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Detecta el patrón clásico de sobreajuste sobre las métricas por época.

    Criterio aplicado: se marca la PRIMERA época en la que la pérdida de
    validación SUBE respecto de la época anterior mientras la pérdida media de
    entrenamiento sigue BAJANDO (divergencia train/val). Un aumento simultáneo
    de ambas pérdidas no es sobreajuste sino inestabilidad de optimización, y
    por eso no se marca aquí.

    Devuelve `{"val_loss_increased_after_epoch": None, ...}` si no se observa el
    patrón o si no hay suficientes épocas (se necesitan ≥2) para concluir.
    """
    usable = [
        m
        for m in epoch_metrics
        if m.get("val_loss") is not None and m.get("train_loss_mean") is not None
    ]
    if len(usable) < 2:
        return {
            "val_loss_increased_after_epoch": None,
            "note": (
                f"No evaluable: se registraron {len(usable)} época(s) con métricas completas y el "
                "criterio de sobreajuste requiere al menos 2 para comparar la tendencia de la "
                "pérdida de validación contra la de entrenamiento. Con una sola época no puede "
                "afirmarse ni descartarse sobreajuste."
            ),
        }

    for prev, curr in zip(usable, usable[1:]):
        if curr["val_loss"] > prev["val_loss"] and curr["train_loss_mean"] < prev["train_loss_mean"]:
            return {
                "val_loss_increased_after_epoch": prev["epoch"],
                "note": (
                    f"Posible sobreajuste: tras la época {prev['epoch']} la pérdida de validación "
                    f"subió ({prev['val_loss']:.4f} -> {curr['val_loss']:.4f}) mientras la pérdida "
                    f"media de entrenamiento siguió bajando ({prev['train_loss_mean']:.4f} -> "
                    f"{curr['train_loss_mean']:.4f}). Se recomienda usar el checkpoint de la época "
                    f"{prev['epoch']} (early stopping) o reforzar la regularización."
                ),
            }

    return {
        "val_loss_increased_after_epoch": None,
        "note": (
            f"Sin sobreajuste crítico detectado en {len(usable)} épocas: no se observó ninguna época "
            "en la que la pérdida de validación subiera mientras la de entrenamiento bajaba "
            f"(val_loss final = {usable[-1]['val_loss']:.4f} vs. inicial = {usable[0]['val_loss']:.4f})."
        ),
    }


def _loss_decreased(
    epoch_metrics: list[dict[str, Any]], loss_history: list[float]
) -> bool | None:
    """
    Indica si la pérdida de entrenamiento descendió a lo largo de la ejecución.

    Se compara la MEDIA POR ÉPOCA, no el primer paso contra el último: la
    pérdida de un paso individual depende del lote que le tocó y su varianza
    puede ocultar por completo una convergencia real (se observó una ejecución
    con medias por época de 0.400 -> 0.136 en la que, sin embargo, el último
    paso resultó mayor que el primero). Como este indicador forma parte de la
    evidencia de convergencia de R1.4, debe reflejar la tendencia y no el ruido.

    Con una sola época se recurre a comparar la media del primer y el último
    decil de pasos, que sigue siendo más estable que dos pasos sueltos.
    """
    if len(epoch_metrics) >= 2:
        return epoch_metrics[-1]["train_loss_mean"] < epoch_metrics[0]["train_loss_mean"]
    if len(loss_history) >= 10:
        k = max(len(loss_history) // 10, 1)
        return (sum(loss_history[-k:]) / k) < (sum(loss_history[:k]) / k)
    if len(loss_history) >= 2:
        return loss_history[-1] < loss_history[0]
    return None


def plot_learning_curve_from_history(
    history_path: str | Path, output_path: str | Path | None = None
) -> Path:
    """
    Regenera la figura de curvas de aprendizaje a partir del JSON de historial.

    Permite reconstruir la evidencia gráfica de R1.4 sin reentrenar: el
    entrenamiento real corre en el laboratorio GPU remoto y basta con traer el
    JSON (pocos KB) para producir la figura localmente.

    Devuelve la ruta del PNG generado. Propaga la excepción si matplotlib falla
    (el llamador dentro de `train()` la captura para no tumbar el entrenamiento).
    """
    import matplotlib

    matplotlib.use("Agg")  # backend sin display: servidores headless (laboratorio GPU)
    import matplotlib.pyplot as plt

    history_path = Path(history_path)
    history: dict[str, Any] = json.loads(history_path.read_text(encoding="utf-8"))

    run_name = history.get("run_name", history_path.stem)
    step_losses = history.get("step_losses") or []
    epoch_metrics = history.get("epoch_metrics") or []
    output_path = Path(output_path) if output_path else get_learning_curve_path(run_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Último paso global de cada época: ancla las métricas de validación (una por
    # época) sobre el mismo eje de pasos que la pérdida de entrenamiento.
    last_step_by_epoch: dict[int, int] = {}
    for record in step_losses:
        last_step_by_epoch[record["epoch"]] = max(
            last_step_by_epoch.get(record["epoch"], 0), record["step"]
        )

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    if step_losses:
        ax_loss.plot(
            [r["step"] for r in step_losses],
            [r["loss"] for r in step_losses],
            color="tab:blue",
            alpha=0.45,
            linewidth=1.0,
            label="Pérdida de entrenamiento (por paso)",
        )
    if epoch_metrics:
        epoch_steps = [last_step_by_epoch.get(m["epoch"], m["epoch"] + 1) for m in epoch_metrics]
        ax_loss.plot(
            epoch_steps,
            [m["train_loss_mean"] for m in epoch_metrics],
            color="tab:blue",
            marker="o",
            linewidth=2.0,
            label="Pérdida media de entrenamiento (por época)",
        )
        ax_loss.plot(
            epoch_steps,
            [m["val_loss"] for m in epoch_metrics],
            color="tab:red",
            marker="s",
            linewidth=2.0,
            label="Pérdida de validación (por época)",
        )

    ax_loss.set_xlabel("Paso de optimización")
    ax_loss.set_ylabel("Pérdida (loss)")
    ax_loss.set_title("(a) Convergencia: pérdida de entrenamiento y de validación")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="best", fontsize=9)

    if epoch_metrics:
        epochs = [m["epoch"] for m in epoch_metrics]
        ax_acc.plot(
            epochs,
            [m["val_accuracy"] for m in epoch_metrics],
            color="tab:green",
            marker="o",
            linewidth=2.0,
            label="Exactitud de validación",
        )
        ax_acc.set_xticks(epochs)
    ax_acc.set_xlabel("Época")
    ax_acc.set_ylabel("Exactitud (accuracy)")
    ax_acc.set_title("(b) Exactitud en validación por época")
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(loc="best", fontsize=9)

    overfitting = history.get("overfitting_check") or {}
    marked_epoch = overfitting.get("val_loss_increased_after_epoch")
    if marked_epoch is not None:
        anchor = last_step_by_epoch.get(marked_epoch)
        if anchor is not None:
            ax_loss.axvline(anchor, color="tab:orange", linestyle="--", linewidth=1.5)
            ax_loss.annotate(
                f"Sobreajuste desde época {marked_epoch}",
                xy=(anchor, ax_loss.get_ylim()[1]),
                xytext=(-5, -12),
                textcoords="offset points",
                ha="right",
                fontsize=9,
                color="tab:orange",
            )

    fig.suptitle(f"Curvas de aprendizaje — {run_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Curva de aprendizaje guardada: %s", output_path)
    return output_path


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
    run_name: str | None = None,
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

    `run_name`, si se pasa, nombra los artefactos de evidencia de R1.4
    (`data/reports/training_logs/{run_name}_history.json` y
    `data/reports/figures/learning_curve_{run_name}.png`); por defecto se deriva
    del `fusion_type`. Es obligatorio pasarlo en entrenamiento multi-fold para
    que un fold no sobrescriba el historial del anterior.

    La validación se evalúa al FINAL DE CADA ÉPOCA (no solo al final del
    entrenamiento): la curva train-loss vs. val-loss a lo largo de las épocas es
    lo que permite sustentar "ausencia de sobreajuste crítico" en R1.4, cosa
    imposible con un único punto final.
    """
    model_config = model_config or ModelConfig(fusion_type=fusion_type)
    model_config.fusion_type = fusion_type
    train_config = train_config or TrainConfig()
    device = device or _select_device("auto")

    # Reproducibilidad: la tesis declara la ejecución reproducible como criterio
    # metodológico, lo que exige fijar TODOS los generadores implicados -- no solo
    # el de la biblioteca tensorial (el muestreo del DataLoader y cualquier
    # operación de numpy también consumen aleatoriedad).
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)

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

    collate_fn = make_collate_fn(tokenizer)  # relleno dinámico por lote
    # Agrupamiento por longitud SOLO en entrenamiento: reduce el costo de atención
    # en un factor ~7.6x medido sobre este corpus (ver doc/REVISION_CODIGO_MODELO.md,
    # hallazgo 3). En validación no se usa: el orden allí es irrelevante para el
    # resultado y conviene mantener el recorrido secuencial simple.
    train_sampler = LengthGroupedSampler(
        train_dataset.approx_lengths,
        batch_size=train_config.batch_size,
        shuffle=True,
        seed=train_config.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.eval_batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        collate_fn=collate_fn,
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

    # Planificador de tasa de aprendizaje (calentamiento + decaimiento lineal).
    # El total de pasos se calcula a partir del número real de lotes por época,
    # acotado por `max_steps` cuando se usa: un planificador dimensionado sobre
    # un horizonte distinto del real dejaría la tasa a mitad de decaimiento.
    scheduler = None
    if train_config.use_lr_scheduler:
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * train_config.epochs
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)
        total_steps = max(total_steps, 1)
        warmup_steps = int(total_steps * train_config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        logger.info(
            "Planificador de LR activo: %d pasos totales, %d de calentamiento (%.0f%%)",
            total_steps,
            warmup_steps,
            train_config.warmup_ratio * 100,
        )

    checkpoint_path = checkpoint_path or get_checkpoint_path(fusion_type.value)
    best_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_best.pt")
    start_epoch, global_step = 0, 0
    if resume and checkpoint_path.exists():
        ckpt = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch, global_step = ckpt["epoch"], ckpt["global_step"]

    loss_history: list[float] = []
    step_losses: list[dict[str, Any]] = []
    epoch_metrics: list[dict[str, Any]] = []
    stopped_early = False
    # Seguimiento del MEJOR modelo, guardado aparte del punto de control de
    # reanudación: este último debe reflejar el estado más RECIENTE para poder
    # continuar la ejecución, mientras que la evaluación debe emplear el estado
    # ÓPTIMO. Son dos requisitos distintos y por eso son dos archivos distintos.
    best_val_loss = float("inf")
    best_epoch: int | None = None
    epochs_without_improvement = 0
    # Época desde la que debe continuar una reanudación posterior. Se actualiza
    # solo cuando una época se completa ENTERA: si el entrenamiento se corta a
    # mitad de época, la reanudación repite esa época desde su inicio (opción
    # conservadora, ya que el DataLoader no expone su posición interna).
    next_epoch = start_epoch

    for epoch in range(start_epoch, train_config.epochs):
        train_sampler.set_epoch(epoch)  # lotes distintos en cada época
        model.train()
        epoch_losses: list[float] = []
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
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            step_loss = loss.item()
            loss_history.append(step_loss)
            epoch_losses.append(step_loss)
            step_losses.append({"step": global_step, "epoch": epoch, "loss": round(step_loss, 6)})

            if global_step % train_config.checkpoint_every_n_steps == 0:
                save_checkpoint(checkpoint_path, model, optimizer, epoch, global_step, model_config)

            if max_steps is not None and global_step >= max_steps:
                stopped_early = True
                break

        # Evaluación de validación AL FINAL DE CADA ÉPOCA -- también cuando la
        # época se cortó por `max_steps`, para que las pruebas de humo produzcan
        # al menos un punto de curva.
        if epoch_losses:
            epoch_val = evaluate_loss_accuracy(model, val_loader, loss_fn, device)
            model.train()  # evaluate_loss_accuracy deja el modelo en eval()
            epoch_metrics.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss_mean": round(sum(epoch_losses) / len(epoch_losses), 6),
                    "val_loss": round(epoch_val["loss"], 6),
                    "val_accuracy": round(epoch_val["accuracy"], 6),
                }
            )
            logger.info(
                "Epoch %d completado (global_step=%d): train_loss_mean=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                global_step,
                epoch_metrics[-1]["train_loss_mean"],
                epoch_metrics[-1]["val_loss"],
                epoch_metrics[-1]["val_accuracy"],
            )

            # Selección del MEJOR modelo, no del último. Sin esto, una ejecución
            # que alcanza su óptimo en la época 2 de 3 conserva de todos modos los
            # pesos de la época 3, ya degradados: `detect_overfitting` advertía del
            # problema en el informe pero no impedía que ocurriera. El criterio es
            # la pérdida de validación, que es continua y detecta el deterioro
            # antes que la exactitud.
            current_val_loss = epoch_metrics[-1]["val_loss"]
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(
                    best_checkpoint_path, model, optimizer, epoch + 1, global_step, model_config
                )
                logger.info(
                    "Nuevo mejor modelo (val_loss=%.6f) guardado en %s",
                    best_val_loss,
                    best_checkpoint_path.name,
                )
            else:
                epochs_without_improvement += 1
                if (
                    train_config.early_stopping_patience > 0
                    and epochs_without_improvement >= train_config.early_stopping_patience
                ):
                    logger.info(
                        "Parada temprana: %d épocas sin mejora de val_loss (mejor: época %d, %.6f)",
                        epochs_without_improvement,
                        best_epoch,
                        best_val_loss,
                    )
                    stopped_early = True
        else:
            logger.info("Epoch %d completado sin pasos de optimización (global_step=%d)", epoch, global_step)

        if stopped_early:
            break
        next_epoch = epoch + 1  # época completada entera

    save_checkpoint(checkpoint_path, model, optimizer, next_epoch, global_step, model_config)
    if epoch_metrics:
        # Mismo estado del modelo y mismo conjunto que la última evaluación por
        # época: se reutiliza en vez de recomputar (evaluación determinista).
        val_metrics = {
            "loss": epoch_metrics[-1]["val_loss"],
            "accuracy": epoch_metrics[-1]["val_accuracy"],
        }
    else:
        val_metrics = evaluate_loss_accuracy(model, val_loader, loss_fn, device)

    run_name = run_name or fusion_type.value
    overfitting_check = detect_overfitting(epoch_metrics)
    # La evaluación debe usar el mejor modelo, no el último. Se informa su
    # ruta explícitamente para que `evaluate.py` y el plan de servidor apunten
    # al archivo correcto sin depender de una convención implícita.
    best_model_info = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss if best_epoch is not None else None,
        "best_checkpoint": str(best_checkpoint_path) if best_epoch is not None else None,
        "early_stopping_patience": train_config.early_stopping_patience,
    }
    history = {
        "run_name": run_name,
        "best_model": best_model_info,
        "fusion_type": fusion_type.value,
        "device": str(device),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "epochs": train_config.epochs,
            "batch_size": train_config.batch_size,
            "backbone_lr": train_config.backbone_lr,
            "head_lr": train_config.head_lr,
            "use_class_weights": train_config.use_class_weights,
            "use_focal_loss": train_config.use_focal_loss,
            "max_steps": max_steps,
        },
        "step_losses": step_losses,
        "epoch_metrics": epoch_metrics,
        "final": {
            "global_step": global_step,
            "final_train_loss": round(loss_history[-1], 6) if loss_history else None,
            "loss_decreased": _loss_decreased(epoch_metrics, loss_history),
            "stopped_early": stopped_early,
            "val_metrics": val_metrics,
            "checkpoint_path": str(checkpoint_path),
        },
        "overfitting_check": overfitting_check,
    }

    history_path = get_history_path(run_name)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Historial de entrenamiento guardado: %s", history_path)

    # El PNG es un derivado regenerable desde el JSON: un fallo de matplotlib
    # (backend sin display, versión, etc.) NO debe tumbar el entrenamiento.
    learning_curve_path: str | None = None
    try:
        learning_curve_path = str(plot_learning_curve_from_history(history_path))
    except Exception as exc:  # pragma: no cover -- degradación controlada
        logger.warning(
            "No se pudo generar la curva de aprendizaje para '%s' (%s: %s). El historial JSON sí se "
            "guardó en %s -- regenerar después con "
            "plot_learning_curve_from_history('%s').",
            run_name,
            type(exc).__name__,
            exc,
            history_path,
            history_path,
        )

    return {
        "run_name": run_name,
        "fusion_type": fusion_type.value,
        "device": str(device),
        "global_step": global_step,
        "loss_history": loss_history,
        "step_losses": step_losses,
        "epoch_metrics": epoch_metrics,
        "final_train_loss": loss_history[-1] if loss_history else None,
        "loss_decreased": _loss_decreased(epoch_metrics, loss_history),
        "val_metrics": val_metrics,
        "overfitting_check": overfitting_check,
        "checkpoint_path": str(checkpoint_path),
        "best_model": best_model_info,
        "history_path": str(history_path),
        "learning_curve_path": learning_curve_path,
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
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nombre de la corrida para los artefactos de evidencia (por defecto: el fusion_type)",
    )
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
        run_name=args.run_name,
    )
    elapsed = time.time() - t0

    print("=" * 70)
    print(f"Entrenamiento completado en {elapsed:.1f}s")
    print(f"fusion_type={result['fusion_type']} device={result['device']} steps={result['global_step']}")
    print(f"loss inicial={result['loss_history'][0] if result['loss_history'] else 'N/A'}")
    print(f"loss final={result['final_train_loss']} (bajó: {result['loss_decreased']})")
    print(f"val_metrics={result['val_metrics']}")
    for m in result["epoch_metrics"]:
        print(
            f"  época {m['epoch']}: train_loss_mean={m['train_loss_mean']} "
            f"val_loss={m['val_loss']} val_accuracy={m['val_accuracy']}"
        )
    print(f"overfitting_check: {result['overfitting_check']['note']}")
    print(f"checkpoint: {result['checkpoint_path']}")
    print(f"historial: {result['history_path']}")
    print(f"curva de aprendizaje: {result['learning_curve_path']}")


if __name__ == "__main__":
    main()
