"""
Ejecución de una variante sobre una partición dada.

Aísla el único punto donde los experimentos tocan el entrenador, de modo que
cambiar cómo se entrena no obligue a tocar cinco guiones ni pueda introducir
diferencias accidentales entre ellos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import Particion, metricas  # noqa: E402
from entrenar import configuracion, ramas_activas, usa_texto  # noqa: E402


def _anular_ramas(df: pd.DataFrame, conservar: tuple[str, ...],
                  con_texto: bool = True) -> pd.DataFrame:
    """Pone a cero las entradas que la variante no usa.

    Es como se construyen los unimodales: no con una arquitectura distinta, sino
    con la misma a la que se le retira la entrada. Así la comparación mide la
    información de la modalidad y no una diferencia de capacidad entre
    arquitecturas, que sería otro experimento.

    `con_texto=False` retira además el cuerpo del correo. Hace falta porque la
    fusión por concatenación agrupa e incorpora el texto SIEMPRE: sin esto, la
    variante llamada "solo estructura" recibe texto y estructura, y su cifra no
    responde a la pregunta que le da nombre.
    """
    from phishing_model.config import NETWORK_CATEGORICAL_COLS
    from phishing_pipeline.features.vectorizer import (
        NETWORK_FEATURE_COLS, STRUCTURAL_FEATURE_COLS)

    df = df.copy()
    if not con_texto:
        # La cadena vacía se tokeniza como [CLS][SEP]: representación constante
        # para todos los ejemplos, sin riesgo de secuencia vacía ni de NaN.
        df["clean_text"] = ""
    if "estructura" not in conservar:
        for c in STRUCTURAL_FEATURE_COLS:
            if c in df.columns:
                df[c] = 0
        df["has_structure_modality"] = False
    if "red" not in conservar:
        for c in NETWORK_FEATURE_COLS:
            if c in df.columns:
                df[c] = 0
        for c in NETWORK_CATEGORICAL_COLS:
            if c in df.columns:
                df[c] = None
        df["has_network_modality"] = False
    return df


def _leer_predicciones(nombre: str, division: str) -> dict | None:
    """Predicciones de una corrida anterior, si existen."""
    from phishing_baseline.evaluation import DEFAULT_PREDICTIONS_DIR

    ruta = DEFAULT_PREDICTIONS_DIR / f"{nombre}_{division}_preds.parquet"
    if not ruta.exists():
        return None
    pred = pd.read_parquet(ruta)
    y_true, y_pred = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
    y_proba = pred["y_proba"].to_numpy() if "y_proba" in pred.columns else None
    # Se devuelven tambien las rutas del punto de control y de los escaladores: la
    # rejilla de degradacion reevalua el modelo entrenado sobre textos corrompidos,
    # y sin ellas el modo de reejecucion no le serviria.
    punto = BASE / "data" / "model" / "checkpoints" / f"{nombre}_best.pt"
    escaladores = BASE / "data" / "model" / "scalers" / f"{nombre}.joblib"
    return {"metricas": metricas(y_true, y_pred, y_proba), "y_true": y_true,
            "y_pred": y_pred, "y_proba": y_proba,
            "punto_de_control": str(punto) if punto.exists() else "",
            "escaladores": str(escaladores) if escaladores.exists() else "",
            "resumen_entrenamiento": {}}


def ejecutar_variante(variante: str, corpus: pd.DataFrame, particion: Particion,
                      semilla: int, epocas: int = 3, sin_centinela: bool = False,
                      rapido: bool = False, enmascarar: tuple[str, ...] | None = None,
                      reusar: bool = False, experimento: str = "exp",
                      sufijo: str = "", ponderar_clases: bool = False,
                      codificador: str | None = None) -> dict:
    """
    Entrena una variante y devuelve sus métricas y predicciones sobre la prueba.

    `enmascarar` oculta ramas SOLO en evaluación, que es la manipulación de E3:
    mide la tolerancia a que falte información en despliegue, sobre un modelo que
    durante el ajuste sí la tuvo.
    """
    from phishing_model.evaluate import evaluate_checkpoint
    from phishing_model.train import train

    modelo_cfg, train_cfg = configuracion(
        variante, semilla, sin_centinela=sin_centinela,
        epocas=1 if rapido else epocas, ponderar_clases=ponderar_clases,
        codificador=codificador,
    )
    if rapido:
        train_cfg.max_steps = 8

    conservar = ramas_activas(variante)
    con_texto = usa_texto(variante)
    intacto = conservar == ("estructura", "red") and con_texto
    datos = corpus if intacto else _anular_ramas(corpus, conservar, con_texto=con_texto)

    prueba = datos.loc[particion.prueba]
    if enmascarar:
        # Se retiran las ramas indicadas únicamente del conjunto de prueba.
        quedan = tuple(r for r in ("estructura", "red") if r not in enmascarar)
        prueba = _anular_ramas(prueba, quedan)

    # El nombre lleva el experimento por delante. Sin eso, E1 y E2 --que entrenan
    # la misma variante sobre la misma particion-- escriben el MISMO fichero de
    # predicciones y el mismo punto de control, y el segundo pisa al primero sin
    # avisar. Se comprobo: E2 sobrescribio las predicciones de E1 quince minutos
    # despues de que E1 terminara, y `--reusar` devolvia entonces las de E2
    # creyendo que eran las de E1.
    # El sufijo distingue CONDICIONES dentro de un mismo experimento. Sin el, las
    # cuatro fracciones de la curva de aprendizaje de E4 --misma variante, misma
    # semilla, distinto subconjunto de entrenamiento-- escribian todas el mismo
    # fichero de predicciones y se pisaban. Las metricas informadas eran correctas
    # porque se calculan al vuelo, pero `--reusar` habria devuelto la ultima
    # condicion para todas, y cualquier analisis posterior sobre los parquet
    # habria comparado un modelo consigo mismo.
    # `sin_centinela` entra en el nombre AUTOMATICAMENTE, no por que cada
    # experimento se acuerde de pasar un sufijo. Es la tercera vez que dos
    # corridas distintas comparten fichero en este proyecto --E1 con E2, las
    # fracciones de E4 entre si, y los dos mecanismos de E3-- y el patron es
    # siempre el mismo: dos configuraciones que difieren en algo que no aparece
    # en la ruta. Cualquier campo que cambie el modelo entrenado tiene que estar
    # aqui, porque confiar en que quien anada una condicion se acuerde de
    # distinguirla es exactamente lo que ya fallo tres veces.
    marca = f"{experimento}_{variante}_s{semilla}"
    if sin_centinela:
        marca += "_sc"
    # Igual que el centinela: la ponderación cambia el modelo entrenado, así que
    # tiene que aparecer en la ruta o las dos condiciones compartirían fichero.
    if ponderar_clases:
        marca += "_pond"
    # El codificador cambia los pesos entrenados: sin aparecer en la marca, las
    # dos condiciones compartirian punto de control y la segunda sobrescribiria a
    # la primera. Es el mismo defecto que ya se corrigio con el centinela.
    if codificador:
        marca += "_" + codificador.split("/")[-1].replace("-", "")[:18]
    if sufijo:
        marca += f"_{sufijo}"
    nombre = marca
    division = marca[len(experimento) + 1:]
    if reusar:
        # Reejecución sin tarjeta: se leen las predicciones ya guardadas en vez
        # de volver a ajustar. Sirve para regenerar todos los informes tras
        # cambiar el formato o añadir una métrica, que de otro modo costaría
        # repetir entrenamientos idénticos durante horas. Si no existen, se
        # entrena: el modo nunca inventa un resultado que no se produjo.
        previo = _leer_predicciones(nombre, division)
        # Si faltase el punto de control, reutilizar dejaria a la rejilla sin nada
        # que reevaluar. En ese caso se entrena: el modo nunca devuelve a medias.
        if previo is not None and previo["punto_de_control"]:
            previo["metricas"]["variante"] = variante
            previo["metricas"]["semilla"] = int(semilla)
            previo["reusado"] = True
            return previo
    # Cada corrida escribe sus escaladores en su propia ruta: compartirla haría
    # que una variante sobrescribiera los de otra antes de evaluarla, filtrando
    # el escalado entre condiciones del mismo experimento.
    escaladores = BASE / "data" / "model" / "scalers" / f"{nombre}.joblib"
    escaladores.parent.mkdir(parents=True, exist_ok=True)
    resumen = train(
        fusion_type=modelo_cfg.fusion_type,
        train_df=datos.loc[particion.entrenamiento],
        val_df=datos.loc[particion.validacion],
        model_config=modelo_cfg,
        train_config=train_cfg,
        max_steps=8 if rapido else None,
        run_name=nombre,
        scaler_path=escaladores,
    )
    punto = Path(resumen.get("best_checkpoint") or resumen["checkpoint_path"])
    salida = evaluate_checkpoint(
        checkpoint_path=punto,
        fusion_type=modelo_cfg.fusion_type,
        df=prueba,
        split_name=division,
        model_name=nombre,
        scaler_path=escaladores,
    )
    # `evaluate_checkpoint` no devuelve las predicciones: las persiste en un
    # parquet, que es la convención del proyecto y lo que permite rehacer
    # cualquier prueba pareada sin volver a inferir. Se leen de ahí.
    from phishing_baseline.evaluation import DEFAULT_PREDICTIONS_DIR

    ruta_pred = DEFAULT_PREDICTIONS_DIR / f"{nombre}_{division}_preds.parquet"
    pred = pd.read_parquet(ruta_pred)
    y_true = pred["y_true"].to_numpy()
    y_pred = pred["y_pred"].to_numpy()
    y_proba = pred["y_proba"].to_numpy() if "y_proba" in pred.columns else None

    m = metricas(y_true, y_pred, y_proba)
    m["variante"] = variante
    m["semilla"] = int(semilla)
    if resumen.get("modality_gate") is not None:
        m["compuerta_modal"] = round(float(resumen["modality_gate"]), 6)
    return {"metricas": m, "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba,
            "punto_de_control": str(punto), "escaladores": str(escaladores),
            "resumen_entrenamiento": resumen}


def evaluar_punto_de_control(variante: str, semilla: int, prueba: pd.DataFrame,
                             etiqueta: str, punto: str, escaladores: str,
                             sin_centinela: bool = False,
                             experimento: str = "exp") -> dict:
    """Evalúa un modelo YA entrenado sobre un conjunto de prueba distinto.

    Existe por una razón de coste que cambia qué experimentos son posibles: la
    rejilla de degradación son cuatro operadores por seis intensidades por tres
    semillas. Reentrenando serían 72 ajustes, unas diecisiete horas de tarjeta,
    y el experimento no se haría. Entrenando una vez por semilla y reevaluando,
    son 3 ajustes y 72 inferencias.

    No es solo más barato: es MÁS CORRECTO. La pregunta es qué le pasa a un
    modelo desplegado cuando le llega texto degradado, y un modelo desplegado es
    uno solo. Reentrenar en cada celda mediría otra cosa --cómo aprende de datos
    degradados-- que es el experimento de entrenamiento adversario, no este.
    """
    from phishing_model.evaluate import evaluate_checkpoint
    from phishing_baseline.evaluation import DEFAULT_PREDICTIONS_DIR

    modelo_cfg, _ = configuracion(variante, semilla, sin_centinela=sin_centinela)
    # El nombre lleva el experimento por delante. Sin eso, E1 y E2 --que entrenan
    # la misma variante sobre la misma particion-- escriben el MISMO fichero de
    # predicciones y el mismo punto de control, y el segundo pisa al primero sin
    # avisar. Se comprobo: E2 sobrescribio las predicciones de E1 quince minutos
    # despues de que E1 terminara, y `--reusar` devolvia entonces las de E2
    # creyendo que eran las de E1.
    nombre = f"{experimento}_{variante}_s{semilla}"
    division = f"{variante}_s{semilla}_{etiqueta}"
    evaluate_checkpoint(
        checkpoint_path=Path(punto), fusion_type=modelo_cfg.fusion_type,
        df=prueba, split_name=division, model_name=nombre,
        scaler_path=Path(escaladores),
    )
    pred = pd.read_parquet(DEFAULT_PREDICTIONS_DIR / f"{nombre}_{division}_preds.parquet")
    y_true, y_pred = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
    y_proba = pred["y_proba"].to_numpy() if "y_proba" in pred.columns else None
    return {"metricas": metricas(y_true, y_pred, y_proba),
            "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba}
