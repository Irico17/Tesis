"""Carga el modelo que la tesis reporta, con su escalador y su comprobación.

Se aparta en un módulo propio porque los dos experimentos de R3 necesitan
exactamente lo mismo y porque el emparejamiento entre punto de control y
escalador es una fuente conocida de error silencioso: cada corrida guarda los
suyos en una ruta que lleva su nombre, y evaluar con los de otra corrida desplaza
el escalado sin que ninguna métrica se rompa de forma visible. Aquí el nombre de
la corrida determina ambos, de modo que no pueden divergir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from phishing_model.config import FusionType
from phishing_model.dataset import MultimodalPhishingDataset, load_scalers, make_collate_fn
from phishing_model.evaluate import cargar_pesos, config_desde_checkpoint
from phishing_model.model import MultimodalPhishingClassifier
from phishing_model.explicabilidad.procedencia import comprobar_metadatos

# La corrida cuyo desempeño recoge el cuadro comparativo del corpus completo. Es
# el modelo del que habla el capítulo, y por tanto el que R3 debe explicar: usar
# el de otro experimento produciría explicaciones de un modelo que no es aquel
# sobre el que se afirma nada.
CORRIDA_DE_REFERENCIA = "e4_atencion_cruzada_token_s42_agr"


@dataclass
class ModeloCargado:
    modelo: MultimodalPhishingClassifier
    tokenizador: Any
    escalador_estructura: Any
    escalador_red: Any
    limites: dict
    config: Any
    metadatos: dict


def rutas_de(base: Path, corrida: str = CORRIDA_DE_REFERENCIA) -> tuple[Path, Path]:
    """Punto de control y escalador de una corrida, derivados de su nombre."""
    punto = base / "data" / "model" / "checkpoints" / f"{corrida}_best.pt"
    escalador = base / "data" / "model" / "scalers" / f"{corrida}.joblib"
    return punto, escalador


def cargar(base: Path, corrida: str = CORRIDA_DE_REFERENCIA,
           dispositivo: str | None = None) -> ModeloCargado:
    """Modelo listo para explicar, tras comprobar que llegó a entrenarse."""
    from transformers import AutoTokenizer

    punto, escalador = rutas_de(base, corrida)
    if not punto.exists():
        raise SystemExit(
            f"No existe {punto}. Los puntos de control de las corridas reportadas viven "
            "en el servidor de cómputo; R3 debe ejecutarse allí o traerse el fichero."
        )
    if not escalador.exists():
        raise SystemExit(
            f"No existe {escalador}. Evaluar con el escalador de otra corrida desplaza "
            "el escalado de las características sin que ninguna métrica lo delate."
        )

    meta = comprobar_metadatos(punto)
    ck = torch.load(punto, map_location="cpu", weights_only=False)
    cfg = config_desde_checkpoint(ck, FusionType(ck["fusion_type"]))
    tok = AutoTokenizer.from_pretrained(cfg.text_model_name)
    se, ne, limites = load_scalers(path=escalador)

    modelo = MultimodalPhishingClassifier(cfg)
    cargar_pesos(modelo, ck["model_state_dict"])
    modelo.eval()
    if dispositivo:
        modelo = modelo.to(dispositivo)

    return ModeloCargado(modelo=modelo, tokenizador=tok, escalador_estructura=se,
                         escalador_red=ne, limites=limites, config=cfg, metadatos=meta)


def filas_como_lotes(cargado: ModeloCargado, df: pd.DataFrame,
                     dispositivo: str | None = None) -> list[dict[str, torch.Tensor]]:
    """Convierte un marco de datos en lotes de una fila, que es como se explica.

    La explicación es por correo: LIME perturba un mensaje concreto y la atención
    se reparte dentro de una secuencia concreta, de modo que agrupar filas en un
    lote mezclaría longitudes mediante relleno y cambiaría lo que se atribuye.
    """
    ds = MultimodalPhishingDataset(
        df, cargado.tokenizador,
        structural_scaler=cargado.escalador_estructura,
        network_scaler=cargado.escalador_red,
        max_token_length=cargado.config.max_token_length,
        fit_scalers=False, clip_bounds=cargado.limites,
    )
    collate = make_collate_fn(cargado.tokenizador)
    lotes = []
    for i in range(len(ds)):
        lote = collate([ds[i]])
        lote.pop("email_id", None)
        if dispositivo:
            lote = {k: (v.to(dispositivo) if hasattr(v, "to") else v) for k, v in lote.items()}
        lotes.append(lote)
    return lotes


def muestra_de_positivos(base: Path, corpus: pd.DataFrame, n: int | None,
                         semilla: int = 42,
                         corrida: str = CORRIDA_DE_REFERENCIA) -> pd.DataFrame:
    """Las filas que el modelo reportado clasificó como phishing.

    El indicador de R3.2 habla de la muestra de prueba de predicciones positivas,
    no de las etiquetas positivas: lo que se explica es lo que el sistema decide,
    incluidos sus falsos positivos, porque una explicación solo es útil si el
    analista la recibe antes de saber si el modelo acertó. Se leen de las
    predicciones ya guardadas para que la muestra sea exactamente la reportada y
    no dependa de volver a inferir.
    """
    ruta = (base / "data" / "predictions"
            / f"{corrida}_{corrida[len('e4_'):]}_preds.parquet")
    if not ruta.exists():
        candidatos = sorted((base / "data" / "predictions").glob(f"{corrida}*_preds.parquet"))
        if not candidatos:
            raise SystemExit(
                f"No se encontraron las predicciones de {corrida} en data/predictions. "
                "Sin ellas la muestra no sería la que el capítulo reporta."
            )
        ruta = candidatos[0]

    pred = pd.read_parquet(ruta)
    positivos = pred[pred["y_pred"] == 1]["email_id"]
    filas = corpus[corpus["email_id"].isin(set(positivos))]
    if n is not None and n < len(filas):
        filas = filas.sample(n=n, random_state=semilla)
    return filas
