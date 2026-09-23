"""Se niega a explicar un modelo que no es el que la tesis reporta.

Existe por un fallo real y silencioso. En el equipo local sobrevivía
`exp_atencion_cruzada_token_s42_best.pt`, que por su nombre y su tamaño parecía el
modelo de la investigación y en realidad era una prueba de humo: `epoch=1`,
`global_step=8`, porque la ruta rápida de la cola fija ocho pasos de optimización.
Medido sobre el conjunto de prueba daba F1 de 0.6505 frente al 0.9927 informado.
Nada en el fichero avisaba. Una matriz de decisión de R3.1 calculada sobre él
habría elegido una técnica de explicabilidad comparando atribuciones de un modelo
sin entrenar, y el capítulo habría descrito con detalle el comportamiento de un
objeto equivocado.

De ahí que la comprobación no sea optativa ni quede a criterio de quien ejecute:
los guiones de R3 la invocan antes de atribuir nada, y aborta.

La comprobación tiene dos niveles. El barato lee los metadatos del punto de
control y descarta lo que evidentemente no llegó a entrenarse. El caro evalúa
sobre el conjunto de prueba y contrasta contra la cifra publicada, que es lo único
que acredita que el punto de control es el que produjo la tabla del capítulo y no
otra corrida de la misma arquitectura.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

# Por debajo de esto no hay entrenamiento posible: una época del corpus completo
# son unos quinientos pasos con lote de treinta y dos.
PASOS_MINIMOS = 100

# Tolerancia al contrastar contra la cifra publicada. El informe promedia tres
# semillas y aquí se evalúa una sola, de modo que exigir igualdad exacta
# rechazaría el punto de control correcto; media centésima de F1 supera con
# holgura la dispersión entre semillas observada y sigue descartando un modelo
# que no es el que se informó.
TOLERANCIA_F1 = 0.005


class PuntoDeControlSospechoso(RuntimeError):
    """El punto de control no se corresponde con el modelo reportado."""


def metadatos(ruta: Path) -> dict[str, Any]:
    """Época, paso y configuración guardados dentro del punto de control."""
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    return {
        "ruta": str(ruta),
        "epoch": ck.get("epoch"),
        "global_step": ck.get("global_step"),
        "fusion_type": ck.get("fusion_type"),
        "model_config": ck.get("model_config"),
        "tiene_centinela": any("no_modality_token" in k for k in ck["model_state_dict"]),
    }


def comprobar_metadatos(ruta: Path) -> dict[str, Any]:
    """Descarta lo que ni siquiera llegó a entrenarse. No acredita nada más."""
    meta = metadatos(ruta)
    pasos = meta.get("global_step") or 0
    if pasos < PASOS_MINIMOS:
        raise PuntoDeControlSospechoso(
            f"El punto de control {ruta.name} registra {pasos} pasos de optimización. "
            "Es una prueba de humo, no un modelo entrenado, y explicarlo produciría "
            "afirmaciones sobre un objeto que no es el de la investigación."
        )
    return meta


def f1_reportado(informe_e4: Path, arquitectura: str = "atencion_cruzada_token") -> float | None:
    """La cifra publicada para esa arquitectura sobre el corpus completo."""
    if not informe_e4.exists():
        return None
    datos = json.loads(informe_e4.read_text(encoding="utf-8"))
    cuadro = datos.get("cuadro_comparativo_corpus_completo", {})
    return (cuadro.get(arquitectura) or {}).get("f1")


def comprobar_contra_lo_reportado(f1_medido: float, f1_publicado: float | None,
                                  ruta: Path) -> dict[str, Any]:
    """Contrasta el desempeño medido con el que el capítulo informa."""
    if f1_publicado is None:
        return {"contrastado": False,
                "motivo": "no se localizó el informe del experimento de referencia"}
    diferencia = abs(f1_medido - f1_publicado)
    if diferencia > TOLERANCIA_F1:
        raise PuntoDeControlSospechoso(
            f"El punto de control {ruta.name} alcanza F1 de {f1_medido:.4f} sobre el "
            f"conjunto de prueba, frente al {f1_publicado:.4f} que el capítulo informa "
            f"para esta arquitectura. La diferencia de {diferencia:.4f} supera la "
            f"tolerancia de {TOLERANCIA_F1}. No es el modelo reportado."
        )
    return {"contrastado": True, "f1_medido": round(f1_medido, 6),
            "f1_publicado": round(f1_publicado, 6), "diferencia": round(diferencia, 6),
            "tolerancia": TOLERANCIA_F1}
