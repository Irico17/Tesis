"""
Fidelidad de las atribuciones (criterio ausente en la matriz de decisión de R3.1).

Motivación metodológica. La matriz de decisión de R3.1 comparaba los tres
candidatos de explicabilidad según tres criterios: costo computacional,
estabilidad ante repetición y acuerdo entre métodos. Ninguno de los tres
responde a la pregunta que verdaderamente importa: **¿la explicación refleja lo
que el modelo realmente utiliza para decidir?**

El acuerdo entre métodos no acredita corrección —dos técnicas pueden coincidir
y estar ambas equivocadas—, y la estabilidad tampoco: un método que devuelve
siempre la misma explicación errónea es perfectamente estable. Seleccionar la
técnica de explicabilidad sin medir su fidelidad equivaldría a elegir un
clasificador por su rapidez y su determinismo, sin observar su exactitud.

Este módulo implementa las dos métricas establecidas para ello, siguiendo el
protocolo de DeYoung et al. (2020) en el banco de pruebas ERASER:

- **Exhaustividad** (*comprehensiveness*): se eliminan del mensaje las palabras
  a las que la técnica atribuyó mayor importancia y se mide cuánto desciende la
  probabilidad de la clase predicha. Si la explicación es fiel, retirar aquello
  que declaró decisivo debe degradar la predicción de forma pronunciada.
  **Valores altos son mejores.**

- **Suficiencia** (*sufficiency*): se conservan ÚNICAMENTE esas palabras y se
  descarta el resto. Si la explicación es fiel, lo conservado debe bastar para
  sostener la predicción y la probabilidad debe mantenerse.
  **Valores bajos son mejores** (mide la caída, de modo que menos caída es más
  suficiente).

Ambas se calculan mediante pasadas hacia adelante adicionales, sin gradientes ni
reentrenamiento, por lo que su costo es despreciable frente al de generar las
propias atribuciones.

Nota sobre el alcance. Se perturba exclusivamente la modalidad TEXTUAL; las
ramas estructural y de red permanecen intactas. Es lo correcto: lo que se está
midiendo es la fidelidad de la atribución sobre el texto, que es la que los tres
candidatos producen de forma comparable. La atribución a nivel de modalidad que
proporciona la atención cruzada se evalúa por separado.
"""

from __future__ import annotations

from typing import Any

import torch

DEFAULT_TOP_K_FRACTION = 0.2  # proporción de palabras consideradas "la explicación"


def _predict_proba(
    model: Any, row_batch: dict[str, torch.Tensor], target_class: int
) -> float:
    """Probabilidad que el modelo asigna a `target_class` para una fila."""
    model.eval()
    with torch.no_grad():
        logits = model(row_batch)
        return float(torch.softmax(logits, dim=-1)[0, target_class].item())


def _rebuild_batch_with_text(
    row_batch: dict[str, torch.Tensor],
    tokenizer: Any,
    new_text: str,
    max_token_length: int,
) -> dict[str, torch.Tensor]:
    """
    Devuelve una copia de la fila con el texto sustituido y el resto intacto.

    Las modalidades estructural y de red se conservan sin modificación porque la
    perturbación debe aislar el efecto del texto: alterarlas también impediría
    atribuir la caída de probabilidad a la explicación textual evaluada.
    """
    encoded = tokenizer(
        new_text if new_text.strip() else tokenizer.unk_token or "[UNK]",
        truncation=True,
        max_length=max_token_length,
        return_tensors="pt",
    )
    perturbed = dict(row_batch)
    # El tokenizador devuelve siempre tensores en CPU. Si el resto de la fila
    # vive en GPU —el caso normal al explicar un modelo ya cargado para
    # inferencia— mezclar ambos dispositivos aborta el paso hacia adelante con
    # "Expected all tensors to be on the same device". Se toma el dispositivo de
    # un tensor existente de la propia fila en vez de asumir uno.
    referencia = next(
        (v for v in row_batch.values() if isinstance(v, torch.Tensor)), None
    )
    device = referencia.device if referencia is not None else torch.device("cpu")
    perturbed["input_ids"] = encoded["input_ids"].to(device)
    perturbed["attention_mask"] = encoded["attention_mask"].to(device)
    return perturbed


def compute_faithfulness(
    model: Any,
    tokenizer: Any,
    row_batch: dict[str, torch.Tensor],
    word_scores: list[tuple[str, float]],
    original_text: str,
    target_class: int,
    top_k_fraction: float = DEFAULT_TOP_K_FRACTION,
    max_token_length: int = 512,
) -> dict[str, Any]:
    """
    Calcula exhaustividad y suficiencia de una atribución a nivel de palabra.

    `word_scores` es la lista (palabra, peso) que produce cualquiera de los tres
    candidatos; se emplea el valor absoluto del peso para ordenar, dado que una
    contribución fuertemente negativa es igualmente parte de la explicación.

    Devuelve además las probabilidades intermedias, de modo que el resultado sea
    auditable y no un único número sin trazabilidad.
    """
    words = original_text.split()
    if not words or not word_scores:
        return {
            "comprehensiveness": None,
            "sufficiency": None,
            "note": "Sin texto o sin atribuciones: la fidelidad no es evaluable en esta fila.",
        }

    ranked = sorted(word_scores, key=lambda pair: -abs(pair[1]))
    n_top = max(1, int(len(words) * top_k_fraction))
    top_words = {word.lower() for word, _ in ranked[:n_top]}

    kept_without_top = [w for w in words if w.lower() not in top_words]
    only_top = [w for w in words if w.lower() in top_words]

    p_original = _predict_proba(model, row_batch, target_class)
    p_removed = _predict_proba(
        model,
        _rebuild_batch_with_text(row_batch, tokenizer, " ".join(kept_without_top), max_token_length),
        target_class,
    )
    p_only_top = _predict_proba(
        model,
        _rebuild_batch_with_text(row_batch, tokenizer, " ".join(only_top), max_token_length),
        target_class,
    )

    return {
        # Exhaustividad: cuánto cae la probabilidad al RETIRAR la explicación.
        # Más alto es mejor.
        "comprehensiveness": round(p_original - p_removed, 6),
        # Suficiencia: cuánto cae al conservar SOLO la explicación.
        # Más bajo es mejor.
        "sufficiency": round(p_original - p_only_top, 6),
        "p_original": round(p_original, 6),
        "p_after_removing_top": round(p_removed, 6),
        "p_keeping_only_top": round(p_only_top, 6),
        "n_words_total": len(words),
        "n_words_in_explanation": len(only_top),
        "top_k_fraction": top_k_fraction,
    }
