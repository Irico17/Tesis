"""Funciones de pérdida — CrossEntropy con class weights opcional y FocalLoss.

La decisión de usar class weights / Focal Loss se toma empíricamente en la
Fase C (comparando recall de la clase minoritaria con y sin cada técnica sobre
los folds LOSO), no se asume aquí. Este módulo solo provee ambas opciones.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Pesos de clase inversamente proporcionales a la frecuencia (balanced),
    mismo criterio que `class_weight="balanced"` de scikit-learn:
    peso_c = n_muestras / (n_clases * n_muestras_c).
    """
    labels = labels.long()
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)  # evita división por cero si una clase no aparece en el batch
    n_samples = float(labels.numel())
    weights = n_samples / (num_classes * counts)
    return weights


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) para clasificación binaria/multiclase.

    Reduce el peso relativo de ejemplos fáciles (alta confianza, ya bien
    clasificados) y concentra el gradiente en ejemplos difíciles — alternativa
    a class weighting simple para desbalance de clases. `gamma=0` equivale a
    CrossEntropy estándar (con o sin `class_weights`).
    """

    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma debe ser >= 0, recibido {gamma}")
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError(f"logits debe ser [batch, num_classes], recibido {tuple(logits.shape)}")
        targets = targets.long()
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - target_probs).clamp(min=1e-6) ** self.gamma
        loss = -focal_term * target_log_probs

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)[targets]
            loss = loss * weights
            return loss.sum() / weights.sum().clamp(min=1e-8)

        return loss.mean()


def build_loss_fn(
    use_class_weights: bool = False,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """
    Construye la función de pérdida según la configuración de entrenamiento.

    `class_weights`, si se pasa, se usa tanto para CrossEntropy ponderado como
    para FocalLoss ponderado — normalmente viene de `compute_class_weights()`
    sobre el split de train (nunca sobre val/test, para no filtrar información).
    """
    if use_focal_loss:
        return FocalLoss(gamma=focal_gamma, class_weights=class_weights if use_class_weights else None)
    if use_class_weights:
        if class_weights is None:
            raise ValueError("use_class_weights=True requiere pasar class_weights")
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()
