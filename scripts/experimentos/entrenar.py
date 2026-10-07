"""
Puente entre los experimentos y el entrenador del modelo.

Los cinco experimentos difieren en qué datos usan y qué variantes comparan, no en
cómo se entrena. Concentrar aquí el ajuste evita que cada uno reimplemente el
bucle con diferencias accidentales, que es como una diferencia de método acaba
leyéndose como una diferencia de arquitectura.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))

from comun import metricas  # noqa: E402
from phishing_model.config import FusionType, ModelConfig, TrainConfig  # noqa: E402

# Variantes disponibles, con el nombre que aparecerá en tablas y figuras.
VARIANTES: dict[str, dict] = {
    "solo_texto": {"fusion_type": FusionType.TEXT_ONLY},
    # `sin_texto` es imprescindible: sin él estas dos variantes conservan la rama
    # textual --CONCAT_LATE_FUSION siempre concatena el texto agrupado-- y miden
    # "texto + estructura" mientras se llaman "solo estructura". Se comprobó: en la
    # primera corrida las tres daban F1 ~0.996 porque las tres veían el texto.
    "solo_estructura": {"fusion_type": FusionType.CONCAT_LATE_FUSION,
                        "ramas": ("estructura",), "sin_texto": True},
    "solo_red": {"fusion_type": FusionType.CONCAT_LATE_FUSION,
                 "ramas": ("red",), "sin_texto": True},
    "atencion_cruzada_token": {"fusion_type": FusionType.CROSS_ATTENTION_TOKEN_LEVEL},
    "atencion_cruzada_modalidad": {"fusion_type": FusionType.CROSS_ATTENTION_MODALITY_LEVEL},
    "concatenacion_tardia": {"fusion_type": FusionType.CONCAT_LATE_FUSION},
    "fusor_mlp": {"fusion_type": FusionType.CONCAT_LATE_FUSION, "cabeza_mlp": True},
    "mezcla_de_expertos": {"fusion_type": FusionType.MIXTURE_OF_EXPERTS},
}

# Nombre legible para las figuras.
ETIQUETAS = {
    "solo_texto": "Solo texto",
    "solo_estructura": "Solo estructura",
    "solo_red": "Solo red",
    "atencion_cruzada_token": "Atención cruzada\n(token)",
    "atencion_cruzada_modalidad": "Atención cruzada\n(modalidad)",
    "concatenacion_tardia": "Concatenación\ntardía",
    "fusor_mlp": "Fusor MLP",
    "mezcla_de_expertos": "Mezcla de\nexpertos",
}


def etiqueta_de(clave: str) -> str:
    """Nombre legible de una variante, para el rótulo de una figura.

    Las claves llevan guion bajo y van sin tildes porque nombran ficheros y puntos
    de control. Pasarlas a una lámina tal cual dejaba rótulos como «atencion
    cruzada token», que en una tesis no valen.
    """
    return ETIQUETAS.get(clave, clave.replace("_", " ").capitalize())


def configuracion(variante: str, semilla: int, sin_centinela: bool = False,
                  epocas: int = 3, lote: int = 32,
                  ponderar_clases: bool = False,
                  codificador: str | None = None) -> tuple[ModelConfig, TrainConfig]:
    """Configuración de modelo y entrenamiento para una variante.

    `sin_centinela` desactiva además el descarte de modalidad: las dos cosas van
    juntas porque son el mismo mecanismo de tolerancia a la ausencia, y separarlas
    produciría una condición que no corresponde a ninguna pregunta de E3.

    `codificador` sustituye el modelo preentrenado de la rama textual. Todo lo
    demás queda igual, de modo que la diferencia observada entre dos corridas que
    solo difieran en él es atribuible al codificador y a nada más.
    """
    if variante not in VARIANTES:
        raise KeyError(f"variante desconocida: {variante}. Disponibles: {sorted(VARIANTES)}")
    spec = dict(VARIANTES[variante])

    # El descarte de modalidad es propiedad del MODELO, no del entrenamiento:
    # ocurre dentro del paso hacia delante, no en el bucle. "none" desactiva el
    # mecanismo por completo, que es lo que E1 y E2 necesitan para que la
    # comparación mida la fusión y no el manejo de huecos.
    modelo = ModelConfig(
        fusion_type=spec["fusion_type"],
        modality_dropout_mode="none" if sin_centinela else "randomize",
    )
    if codificador:
        modelo.text_model_name = codificador
    if spec.get("cabeza_mlp"):
        # El fusor que sugirió el asesor: dos capas sobre las ramas concatenadas,
        # frente a la cabeza lineal que la concatenación tardía usa por defecto.
        modelo.cabeza_oculta = modelo.dim_feedforward

    entrenamiento = TrainConfig(
        seed=semilla,
        epochs=epocas,
        batch_size=lote,
        criterio_seleccion="val_auc",
        # La ponderación de clases es una DECISIÓN EMPÍRICA comprometida en el
        # plan (indicador de R1.4): no se activa por defecto ni se descarta por
        # defecto, se mide contrastando ambas condiciones sobre los mismos datos.
        use_class_weights=bool(ponderar_clases),
    )
    return modelo, entrenamiento


def ramas_activas(variante: str) -> tuple[str, ...]:
    """Qué ramas no textuales usa la variante; vacío si es unimodal de texto."""
    spec = VARIANTES[variante]
    if spec["fusion_type"] == FusionType.TEXT_ONLY:
        return ()
    return spec.get("ramas", ("estructura", "red"))


def usa_texto(variante: str) -> bool:
    """Si la variante recibe el cuerpo del correo.

    Las unimodales no textuales no lo reciben. No se retira la rama del modelo
    --eso cambiaría la capacidad y confundiría la comparación-- sino su entrada:
    el texto se sustituye por la cadena vacía, de modo que DistilBERT devuelve la
    misma representación constante para todos los ejemplos y no aporta ninguna
    información sobre la clase.
    """
    return not VARIANTES[variante].get("sin_texto", False)


def evaluar_predicciones(y_true, y_pred, y_proba, variante: str, semilla: int) -> dict:
    """Métricas de una corrida, con su identificación."""
    m = metricas(y_true, y_pred, y_proba)
    m["variante"] = variante
    m["semilla"] = int(semilla)
    return m


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Prueba exacta de McNemar entre dos modelos sobre las mismas observaciones.

    Solo intervienen los casos en que ambos discrepan: si uno fuera superior, esas
    discordancias se repartirían de forma asimétrica. Se usa la versión exacta y
    no la aproximación de ji cuadrado porque el número de discordancias puede ser
    reducido, y con pocos casos la aproximación no es fiable.
    """
    from scipy.stats import binomtest

    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    solo_a = int((a_ok & ~b_ok).sum())
    solo_b = int((~a_ok & b_ok).sum())
    n = solo_a + solo_b
    p = float(binomtest(solo_a, n, 0.5).pvalue) if n else 1.0
    return {
        "aciertos_solo_a": solo_a,
        "aciertos_solo_b": solo_b,
        "n_discordancias": n,
        "p_valor": round(p, 6),
        "significativo_005": bool(p < 0.05),
    }
