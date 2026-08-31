"""Funciones de pérdida — CrossEntropy con class weights opcional y FocalLoss.

La decisión de usar class weights / Focal Loss se toma empíricamente en la
Fase C (comparando recall de la clase minoritaria con y sin cada técnica sobre
los folds LOSO), no se asume aquí. Este módulo solo provee ambas opciones.
"""

from __future__ import annotations

import numpy as np
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


class _InvertirGradiente(torch.autograd.Function):
    """Identidad hacia delante; invierte y escala el gradiente hacia atrás."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, escala: float) -> torch.Tensor:
        ctx.escala = escala
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        return -ctx.escala * grad, None


def invertir_gradiente(x: torch.Tensor, escala: float) -> torch.Tensor:
    return _InvertirGradiente.apply(x, escala)


class CabezaAdversariaDeFuente(nn.Module):
    """
    Cabeza adversaria que penaliza las representaciones de las que se puede
    predecir la FUENTE del correo (Ganin et al., 2016).

    Por qué aquí. Se midió que las tres fuentes de este corpus son identificables
    con una exactitud del 96.4% a partir del texto y del 94.9% a partir de las
    características tabulares. Si la representación que alimenta al clasificador
    conserva esa información, el modelo puede —y por economía va a— apoyarse en
    ella, porque dentro del entrenamiento resuelve la tarea casi perfectamente.
    Es el mecanismo del aprendizaje por atajos descrito por Geirhos et al. (2020).

    Se comprobó antes que la vía obvia no sirve: reformular las características
    absolutas como razones sin escala —enlaces por cada cien palabras, nodos del
    árbol por cada cien palabras— EMPEORA la situación en lugar de mejorarla, y la
    razón es que el denominador natural, `word_count`, es él mismo un identificador
    de la fuente, de modo que la división inyecta la información que pretendía
    eliminar. Medido: `nodos_por_100_pal` pasa de una razón clase/fuente de 1.0958
    a 0.6351 respecto de la característica absoluta de la que deriva. La ingeniería
    de características no puede resolverlo; hay que actuar sobre la representación.

    El mecanismo es la capa de inversión de gradiente. Hacia delante la
    representación pasa sin cambios y un clasificador auxiliar intenta predecir la
    fuente; hacia atrás el gradiente que vuelve al cuerpo del modelo se INVIERTE.
    El clasificador auxiliar mejora en identificar la fuente mientras el cuerpo
    aprende a dificultárselo, y el punto de equilibrio es una representación desde
    la que la fuente no es recuperable pero la clase sí.

    `escala` gobierna la intensidad. Ganin et al. la incrementan gradualmente desde
    cero, porque aplicarla al máximo desde el primer paso desestabiliza el
    entrenamiento cuando el cuerpo aún no ha aprendido nada útil; `escala_en` lo
    implementa con su misma programación.

    ADVERTENCIA. Este mecanismo elimina la información de fuente de la
    representación, no la asimetría del corpus. Si una fuente carece de señal real
    —como se verificó en Kaggle, donde ninguna característica tabular supera los
    0.0546 nats de información sobre la clase—, eliminar el atajo no crea la señal
    que falta: solo impide que el modelo la sustituya por el origen del dato.
    Debe reportarse como intervención evaluada, no como corrección garantizada.
    """

    def __init__(self, d_model: int, n_fuentes: int, oculto: int = 128) -> None:
        super().__init__()
        if n_fuentes < 2:
            raise ValueError(
                f"La cabeza adversaria requiere al menos 2 fuentes, recibió {n_fuentes}."
            )
        self.red = nn.Sequential(
            nn.Linear(d_model, oculto),
            nn.ReLU(),
            nn.Linear(oculto, n_fuentes),
        )

    @staticmethod
    def escala_en(progreso: float, gamma: float = 10.0) -> float:
        """
        Programación de la intensidad: 2/(1+exp(-gamma·p)) − 1, con p en [0, 1].

        Vale 0 al comenzar y tiende a 1 al final, de modo que la presión adversaria
        entra de forma gradual. Es la formulación original de Ganin et al.
        """
        p = min(max(float(progreso), 0.0), 1.0)
        return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)

    def forward(
        self, representacion: torch.Tensor, fuentes: torch.Tensor, escala: float
    ) -> torch.Tensor:
        logits = self.red(invertir_gradiente(representacion, escala))
        return F.cross_entropy(logits, fuentes.long())


class GroupDROLoss(nn.Module):
    """
    Minimización del riesgo del peor grupo (Sagawa et al., 2020), con la fuente
    de origen del correo como grupo.

    Por qué esta pérdida y no la ponderación de clases. Se verificó que el
    desbalance dominante de este corpus no es de clase sino de FUENTE: en el
    pliegue que retiene Kaggle, el 88.5% del entrenamiento procede de una sola
    fuente, y en el que retiene PhishMMF, el 82.4%. La entropía cruzada promedia
    sobre ejemplos, de modo que minimizarla equivale a minimizar el riesgo de la
    fuente mayoritaria y a ignorar las demás. Es exactamente el régimen para el
    que se formuló esta pérdida.

    Formulación en línea. Se mantiene un vector de pesos `q`, uno por grupo, que
    se actualiza de forma multiplicativa con la pérdida observada de cada grupo:

        q_g  ←  q_g · exp(η · pérdida_g)      y después se renormaliza

    y la pérdida devuelta es `Σ_g q_g · pérdida_g`. Los grupos que van peor ganan
    peso, así que el gradiente se concentra en ellos. La actualización es en línea
    y por lotes, de modo que **no exige que todos los grupos aparezcan en cada
    lote**: los ausentes conservan su peso. Esto importa aquí, porque el
    muestreador agrupa por longitud y la longitud está fuertemente asociada a la
    fuente —la mediana es de 18 palabras en una fuente y de 167-184 en las otras—,
    de modo que muchos lotes son casi puros en fuente.

    `q` se registra como buffer para que acompañe al modelo entre dispositivos y
    quede guardado en el punto de control, y así una reanudación continúe con los
    pesos que el entrenamiento ya había alcanzado en lugar de reiniciarlos.

    El parámetro `eta` gobierna la agresividad de la reponderación. Sagawa et al.
    advierten que el método requiere regularización fuerte para no sobreajustar el
    peor grupo; aquí esa regularización la aportan el decaimiento de peso y el
    descarte ya presentes en la arquitectura.
    """

    def __init__(
        self,
        n_grupos: int,
        eta: float = 0.01,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if n_grupos < 1:
            raise ValueError(f"n_grupos debe ser >= 1, recibido {n_grupos}")
        if eta <= 0:
            raise ValueError(f"eta debe ser > 0, recibido {eta}")
        self.n_grupos = n_grupos
        self.eta = eta
        self.register_buffer("q", torch.ones(n_grupos) / n_grupos, persistent=True)
        self.register_buffer("class_weights", class_weights, persistent=False)

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, grupos: torch.Tensor | None = None
    ) -> torch.Tensor:
        targets = targets.long()
        peso = self.class_weights.to(logits.device) if self.class_weights is not None else None
        por_ejemplo = F.cross_entropy(logits, targets, weight=peso, reduction="none")

        if grupos is None:
            return por_ejemplo.mean()

        grupos = grupos.long()
        q = self.q.to(logits.device)

        # Pérdida media de cada grupo presente en el lote.
        suma = torch.zeros(self.n_grupos, device=logits.device, dtype=por_ejemplo.dtype)
        cuenta = torch.zeros(self.n_grupos, device=logits.device, dtype=por_ejemplo.dtype)
        suma.index_add_(0, grupos, por_ejemplo)
        cuenta.index_add_(0, grupos, torch.ones_like(por_ejemplo))
        presentes = cuenta > 0
        media = torch.where(presentes, suma / cuenta.clamp(min=1.0), torch.zeros_like(suma))

        # Actualización multiplicativa de `q`, solo sobre los grupos presentes y
        # sin propagar gradiente a través de los pesos: `q` es un estado del
        # algoritmo, no un parámetro que el optimizador deba ajustar.
        with torch.no_grad():
            nuevo = q.clone()
            nuevo[presentes] = q[presentes] * torch.exp(self.eta * media[presentes].detach())
            self.q.copy_((nuevo / nuevo.sum().clamp(min=1e-12)).to(self.q.device))

        q = self.q.to(logits.device)
        # Se renormaliza sobre los grupos presentes para que la escala de la
        # pérdida no dependa de cuántos grupos toque el lote.
        q_lote = q * presentes.to(q.dtype)
        return (q_lote * media).sum() / q_lote.sum().clamp(min=1e-12)

    def pesos_actuales(self) -> list[float]:
        """Pesos por grupo, para registrarlos en el historial de la corrida."""
        return [round(float(v), 6) for v in self.q.detach().cpu()]


def build_loss_fn(
    use_class_weights: bool = False,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
    use_group_dro: bool = False,
    n_grupos: int = 1,
    group_dro_eta: float = 0.01,
) -> nn.Module:
    """
    Construye la función de pérdida según la configuración de entrenamiento.

    `class_weights`, si se pasa, se usa tanto para CrossEntropy ponderado como
    para FocalLoss ponderado — normalmente viene de `compute_class_weights()`
    sobre el split de train (nunca sobre val/test, para no filtrar información).

    `use_group_dro` tiene precedencia sobre la pérdida focal: son dos respuestas a
    desbalances distintos —de fuente la primera, de clase la segunda— y combinarlas
    haría indistinguible cuál produce el efecto observado, que es justamente lo
    que la ablación debe separar. La ponderación de clases sí se compone con
    ambas, porque actúa sobre el término por ejemplo y no sobre la agregación.
    """
    if use_group_dro:
        return GroupDROLoss(
            n_grupos=n_grupos,
            eta=group_dro_eta,
            class_weights=class_weights if use_class_weights else None,
        )
    if use_focal_loss:
        return FocalLoss(gamma=focal_gamma, class_weights=class_weights if use_class_weights else None)
    if use_class_weights:
        if class_weights is None:
            raise ValueError("use_class_weights=True requiere pasar class_weights")
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()
