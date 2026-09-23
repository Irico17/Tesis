"""Atribución sobre las modalidades no textuales, que es la que esta tesis necesita.

Las tres familias de candidatos que R3.1 compromete atribuyen sobre objetos
distintos, y conviene no confundirlos:

  - LIME y los gradientes integrados atribuyen sobre PALABRAS del texto. Las ramas
    de estructura y de red quedan fijas mientras se perturba el mensaje, de modo
    que ninguno de los dos puede decir cuánto pesó que SPF fallara.
  - La atención cruzada y la ablación de rasgos atribuyen sobre las CARACTERÍSTICAS
    tabulares, que es lo que el enunciado de R3.2 pide de forma literal cuando
    habla de traducir las contribuciones de los vectores de características.

Este módulo cubre lo segundo. Existe aparte de `xai.attention_explainer` y no lo
modifica, por dos razones. La primera es que aquel declara 18 nombres y el modelo
entrega 19 posiciones: `model._con_centinela` antepone el token de "sin modalidad"
en la posición 0 SIEMPRE, en todas las variantes de atención por token. Conviene
subrayar que el argumento `sin_centinela` de `entrenar.configuracion` no desactiva
el centinela pese a su nombre, sino el descarte de modalidad, de manera que no
existe ningún punto de control entrenado cuya memoria tenga 18 posiciones. La
segunda es que la anchura no debe suponerse: se lee del tensor y se contrasta con
los nombres canónicos, y si no encaja en ninguno de los dos casos previstos se
aborta en lugar de repartir pesos sobre nombres equivocados.

El centinela no se descarta en silencio. Es una posición que el modelo puede
atender cuando ninguna modalidad le resulta útil, y cuánto la atiende es una
medida del aporte real de la fusión, de modo que se informa por separado en vez de
diluirla entre las características.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from phishing_model.config import NETWORK_CATEGORICAL_COLS
from phishing_model.xai.attention_explainer import extract_modality_attention
from phishing_pipeline.features.vectorizer import NETWORK_FEATURE_COLS, STRUCTURAL_FEATURE_COLS

NOMBRES_CANONICOS: list[str] = (
    list(STRUCTURAL_FEATURE_COLS) + list(NETWORK_FEATURE_COLS) + list(NETWORK_CATEGORICAL_COLS)
)
NOMBRE_DEL_CENTINELA = "sin_modalidad"

# Claves del lote que el modelo consume. Se listan para poder reconstruirlo al
# perturbar una rama sin arrastrar `label` ni identificadores, que el paso hacia
# delante no acepta.
CLAVES_DEL_MODELO = (
    "input_ids", "attention_mask", "structural_continuous", "network_continuous",
    "network_categorical", "has_structure", "has_network",
)


def lote_limpio(lote: dict[str, Any]) -> dict[str, torch.Tensor]:
    """El lote reducido a lo que el modelo consume."""
    return {k: v for k, v in lote.items() if k in CLAVES_DEL_MODELO}


def nombres_para(n_posiciones: int) -> tuple[list[str], bool]:
    """Nombres alineados con la anchura real de la memoria de atención.

    Devuelve además si la primera posición es el centinela. No se adivina por el
    nombre de la variante ni por la configuración: se deduce de la anchura, que es
    lo único que describe el tensor que se va a repartir.
    """
    n_base = len(NOMBRES_CANONICOS)
    if n_posiciones == n_base:
        return list(NOMBRES_CANONICOS), False
    if n_posiciones == n_base + 1:
        return [NOMBRE_DEL_CENTINELA] + list(NOMBRES_CANONICOS), True
    raise ValueError(
        f"La memoria de atención tiene {n_posiciones} posiciones y las características "
        f"canónicas son {n_base}. Solo se contemplan dos casos, con y sin el token "
        "centinela. Una anchura distinta significa que el esquema de características "
        "cambió y que los nombres ya no describen lo que el modelo atiende."
    )


def pesos_de_modalidad(modelo: torch.nn.Module, lote: dict[str, Any]) -> dict[str, Any]:
    """Reparto de la atención cruzada entre las características de modalidad.

    Se promedia sobre las posiciones de texto válidas, ponderando por la máscara,
    porque la pregunta no es a qué atendió un token concreto sino a qué atendió el
    mensaje. El reparto se normaliza para poder leerse como porcentaje.

    La entropía normalizada acompaña al reparto y no es decorativa: vale 1 cuando
    la atención se distribuye por igual entre todas las posiciones, caso en que la
    explicación no distingue nada, y se acerca a 0 cuando se concentra en una sola
    característica. Sin ella, un reparto casi uniforme se leería como una
    explicación legítima con diferencias de centésimas entre rasgos.
    """
    entrada = lote_limpio(lote)
    attn = extract_modality_attention(modelo, entrada)  # [lote, texto, memoria]
    nombres, hay_centinela = nombres_para(attn.shape[-1])

    mascara = entrada["attention_mask"].to(attn.dtype)
    pesado = (attn * mascara.unsqueeze(-1)).sum(dim=1)
    pesado = pesado / mascara.sum(dim=1, keepdim=True).clamp(min=1.0)
    pesado = pesado / pesado.sum(dim=1, keepdim=True).clamp(min=1e-8)

    salida = []
    for fila in pesado:
        valores = fila.tolist()
        completo = dict(zip(nombres, valores))
        centinela = completo.pop(NOMBRE_DEL_CENTINELA, 0.0) if hay_centinela else 0.0

        # Las características se renormalizan entre ellas: el lector de la
        # narrativa pregunta qué pesó dentro de lo que el modelo sí miró, y dejar
        # el centinela dentro del reparto desplazaría todos los porcentajes sin
        # que ninguno de ellos se refiera a una característica del correo.
        resto = sum(completo.values())
        if resto > 0:
            completo = {k: v / resto for k, v in completo.items()}

        entropia = -sum(v * math.log(v) for v in valores if v > 0) / math.log(len(valores))
        salida.append({
            "por_caracteristica": completo,
            "peso_del_centinela": centinela,
            "entropia_normalizada": entropia,
            "uniforme_de_referencia": 1.0 / len(valores),
        })
    return {"filas": salida, "n_posiciones": attn.shape[-1], "con_centinela": hay_centinela}


def _funcion_hacia_delante(modelo: torch.nn.Module):
    """Paso hacia delante con las dos ramas tabulares como entradas atribuibles.

    Captum necesita que lo atribuido sean los primeros argumentos posicionales y
    que el resto viaje como argumentos adicionales. Las categóricas de red quedan
    fuera de la atribución por gradiente: son índices enteros que alimentan una
    tabla de embeddings, y perturbarlos con ruido continuo produce el mismo fallo
    que impide aplicar GradientSHAP sobre los tokens de texto. Su contribución se
    obtiene por ablación, que sí admite entradas discretas.
    """

    def hacia_delante(estructura, red, input_ids, attention_mask, red_categorica,
                      tiene_estructura, tiene_red):
        logits = modelo({
            "input_ids": input_ids, "attention_mask": attention_mask,
            "structural_continuous": estructura, "network_continuous": red,
            "network_categorical": red_categorica,
            "has_structure": tiene_estructura, "has_network": tiene_red,
        })
        return torch.softmax(logits, dim=-1)

    return hacia_delante


def _argumentos(lote: dict[str, torch.Tensor]):
    e = lote_limpio(lote)
    entradas = (e["structural_continuous"], e["network_continuous"])
    adicionales = (e["input_ids"], e["attention_mask"], e["network_categorical"],
                   e["has_structure"], e["has_network"])
    return entradas, adicionales


def _emparejar(atribuciones) -> dict[str, float]:
    valores = torch.cat([atribuciones[0].squeeze(0), atribuciones[1].squeeze(0)]).tolist()
    nombres = list(STRUCTURAL_FEATURE_COLS) + list(NETWORK_FEATURE_COLS)
    return dict(zip(nombres, valores))


def atribucion_por_ablacion(modelo: torch.nn.Module, lote: dict[str, torch.Tensor],
                            clase: int = 1) -> dict[str, float]:
    """Cuánto cambia la probabilidad al anular cada característica, una por una.

    Es el patrón de referencia contra el que se juzgan los demás: no estima la
    contribución, la mide sobre el propio modelo. Su coste crece con el número de
    características, que aquí son quince y por tanto quince pasos hacia delante,
    despreciable frente a las perturbaciones que necesita LIME.
    """
    from captum.attr import FeatureAblation

    modelo.eval()
    entradas, adicionales = _argumentos(lote)
    at = FeatureAblation(_funcion_hacia_delante(modelo)).attribute(
        inputs=entradas, additional_forward_args=adicionales, target=clase)
    return _emparejar(at)


def atribucion_gradient_shap(modelo: torch.nn.Module, lote: dict[str, torch.Tensor],
                             clase: int = 1, n_muestras: int = 16,
                             referencia: tuple[torch.Tensor, torch.Tensor] | None = None,
                             ) -> dict[str, float]:
    """GradientSHAP sobre los vectores de características, que es lo comprometido.

    La matriz de objetivos compromete GradientSHAP mediante Captum sobre los
    vectores de entrada. Sobre el texto no es aplicable, porque el método perturba
    la entrada con ruido gaussiano y los identificadores de token son enteros que
    indexan una tabla de embeddings; sobre los vectores tabulares, que son
    continuos, es su terreno natural.

    `referencia` es la distribución de partida. Si no se indica se usa el vector
    nulo repetido, que corresponde al correo sin ninguna señal estructural ni de
    red. Conviene pasar una muestra real de entrenamiento cuando se quiera que la
    referencia sea el correo típico y no el correo vacío.
    """
    from captum.attr import GradientShap

    modelo.eval()
    entradas, adicionales = _argumentos(lote)

    # La referencia se construye en el MISMO dispositivo que las entradas. Sin
    # esto, en el servidor el modelo vive en la tarjeta y los tensores nulos se
    # creaban en el procesador, de modo que Captum abortaba con "tensores en
    # cuda:0 y cpu". En el equipo local no se manifestaba, porque allí todo está
    # en el procesador: es un defecto que solo aparece donde el experimento se
    # ejecuta de verdad. Una referencia recibida desde fuera se traslada también,
    # porque quien la construya a partir del corpus la tendrá en el procesador.
    dispositivo = entradas[0].device
    if referencia is None:
        referencia = tuple(
            torch.zeros(n_muestras, e.shape[-1], dtype=e.dtype, device=dispositivo)
            for e in entradas
        )
    else:
        referencia = tuple(r.to(dispositivo) for r in referencia)
    at = GradientShap(_funcion_hacia_delante(modelo)).attribute(
        inputs=entradas, baselines=referencia, additional_forward_args=adicionales,
        target=clase, n_samples=n_muestras)
    return _emparejar(at)
