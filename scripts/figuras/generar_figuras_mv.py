"""
Figuras que los medios de verificación exigen y que los experimentos no emiten.

La tabla de medios de verificación del plan pide, para R2.2, «gráficos de
rendimiento (curvas ROC, matriz de confusión)», y para R1.4 «curvas de
aprendizaje (Loss/Accuracy)». Los siete experimentos emiten sus propias figuras
comparativas, pero no esas: este guion las construye a partir de los artefactos
que ya dejó la cola, sin reentrenar ni volver a inferir.

De dónde sale cada figura:

  Matrices de confusión   de `e4.json`, que guarda la matriz de cada modelo y
                          semilla sobre el corpus completo. No hacen falta las
                          predicciones fila por fila.
  Curvas ROC              de `data/predictions/e4_*_s42_agr_*_preds.parquet`,
                          que conservan la probabilidad asignada a cada correo.
  Curvas de aprendizaje   de los historiales de entrenamiento, que registran la
                          pérdida por paso y las métricas de validación por
                          época.
  Diagrama                se dibuja aquí, conforme a lo implementado en
                          `src/phishing_model/model.py`.

    python scripts/figuras/generar_figuras_mv.py
    python scripts/figuras/generar_figuras_mv.py --solo curvas_roc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import formato  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
EXP = BASE / "medios_de_verificacion" / "experimentos"
PREDICCIONES = BASE / "data" / "predictions"
FIGURAS_ARQ = BASE / "data" / "reports" / "figures"

# Destinos, uno por resultado comprometido.
SALIDA_R1_3 = BASE / "medios_de_verificacion" / "R1.3_arquitectura"
SALIDA_R1_4 = BASE / "medios_de_verificacion" / "R1.4_entrenamiento" / "curvas_de_aprendizaje"
SALIDA_R2_2 = BASE / "medios_de_verificacion" / "R2.2_comparacion"

NOMBRES = {
    "atencion_cruzada_token": "Atención cruzada por token (propuesta)",
    "atencion_cruzada_modalidad": "Atención cruzada por modalidad",
    "mezcla_de_expertos": "Mezcla de expertos",
    "fusor_mlp": "Fusor MLP",
    "concatenacion_tardia": "Concatenación tardía",
    "solo_texto": "Unimodal de texto",
    "solo_estructura": "Unimodal de estructura",
    "solo_red": "Unimodal de red",
    "b1_tfidf_lr": "B1: TF-IDF con regresión logística",
    "b3_rf_estructura": "B3: bosque aleatorio sobre estructura",
    "b4_red": "B4: bosque aleatorio sobre red",
    "b5_multimodal_clasico": "B5: fusión clásica de las tres modalidades",
}


def _cuadro_de_e4() -> dict:
    ruta = EXP / "e4" / "e4.json"
    if not ruta.exists():
        raise SystemExit(f"Falta {ruta}. Ejecutar la cola de experimentos.")
    return json.loads(ruta.read_text(encoding="utf-8"))[
        "cuadro_comparativo_corpus_completo"]


# ------------------------------------------------------- matrices de confusión

def matrices_de_confusion() -> int:
    """Una matriz por modelo, con la primera semilla del cuadro comparativo."""
    destino = SALIDA_R2_2 / "matrices_de_confusion"
    destino.mkdir(parents=True, exist_ok=True)
    hechas = 0
    for clave, datos in _cuadro_de_e4().items():
        por_semilla = datos.get("por_semilla") or []
        if not por_semilla or not por_semilla[0].get("confusion_matrix"):
            continue
        cm = np.array(por_semilla[0]["confusion_matrix"])
        semilla = por_semilla[0].get("semilla", "—")

        fig, ax = plt.subplots(figsize=(formato.ANCHO_VERTICAL * 0.72,
                                        formato.ANCHO_VERTICAL * 0.62))
        im = ax.imshow(cm, cmap="Blues")
        total = cm.sum()
        umbral = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:,}\n({100 * cm[i, j] / total:.2f}%)",
                        ha="center", va="center", fontsize=formato.CUERPO,
                        color="white" if cm[i, j] > umbral else "black")
        ax.set_xticks([0, 1], ["Legítimo", "Phishing"])
        ax.set_yticks([0, 1], ["Legítimo", "Phishing"])
        ax.set_xlabel("Predicción del modelo")
        ax.set_ylabel("Clase real")
        ax.set_title(f"{NOMBRES.get(clave, clave)}\nsemilla {semilla}, corpus completo")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(destino / f"matriz_confusion_{clave}.png")
        plt.close(fig)
        hechas += 1
    print(f"  matrices de confusión: {hechas} en {destino.relative_to(BASE)}")
    return hechas


# -------------------------------------------------------------- curvas ROC

def _puntos_roc(y, p):
    """Curva ROC sin depender de scikit-learn, que aquí no hace falta."""
    orden = np.argsort(-p)
    y = np.asarray(y)[orden]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    positivos, negativos = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    tpr = np.concatenate([[0.0], tp / positivos, [1.0]])
    fpr = np.concatenate([[0.0], fp / negativos, [1.0]])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def curvas_roc() -> int:
    """Una figura por familia, con todas sus arquitecturas superpuestas."""
    SALIDA_R2_2.mkdir(parents=True, exist_ok=True)
    curvas: dict[str, tuple] = {}
    for clave in NOMBRES:
        candidatos = sorted(PREDICCIONES.glob(f"e4_{clave}_s42_agr*_preds.parquet"))
        # La misma arquitectura aparece en E4 bajo varias condiciones (ponderada,
        # sin agrupar, por fracción de datos, con otra semilla de partición).
        # Solo interesa la del cuadro comparativo, que es la agrupada sin más.
        candidatos = [c for c in candidatos
                      if not any(m in c.name for m in ("_pond", "_sinagr", "_frac",
                                                       "_psem", "_adv_"))]
        if not candidatos:
            continue
        d = pd.read_parquet(candidatos[0])
        curvas[clave] = _puntos_roc(d["y_true"].to_numpy(), d["y_proba"].to_numpy())

    if not curvas:
        print("  curvas ROC: no hay predicciones en data/predictions/; se omiten")
        return 0

    cuadro = _cuadro_de_e4()
    for etiqueta, familia in (("neuronales", "neuronal"), ("clasicas", "clasica")):
        presentes = [k for k in curvas if cuadro.get(k, {}).get("familia") == familia]
        if not presentes:
            continue
        fig, ax = plt.subplots(figsize=(formato.ANCHO_VERTICAL,
                                        formato.ANCHO_VERTICAL * 0.82))
        for k in presentes:
            fpr, tpr, auc = curvas[k]
            ax.plot(fpr, tpr, lw=1.6, label=f"{NOMBRES.get(k, k)} (AUC {auc:.4f})")
        ax.plot([0, 1], [0, 1], "--", lw=1, color="#999999", label="Azar")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title(f"Curvas ROC sobre el corpus completo\nmodelos de familia {familia}")
        ax.legend(fontsize=formato.MINIMO_LEGIBLE, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(SALIDA_R2_2 / f"curvas_roc_{etiqueta}.png")
        plt.close(fig)

    # Vista logarítmica: al saturar la tarea, la esquina superior izquierda es la
    # única región donde las curvas se distinguen, y es además la que describe el
    # punto de operación de una pasarela de correo.
    fig, ax = plt.subplots(figsize=(formato.ANCHO_VERTICAL,
                                    formato.ANCHO_VERTICAL * 0.82))
    for k, (fpr, tpr, auc) in curvas.items():
        ax.plot(np.clip(fpr, 1e-4, 1), tpr, lw=1.5, label=NOMBRES.get(k, k))
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("Tasa de falsos positivos (escala logarítmica)")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC en la región de operación\ncorpus completo, todas las familias")
    ax.legend(fontsize=formato.MINIMO_LEGIBLE, loc="lower right", ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(SALIDA_R2_2 / "curvas_roc_region_de_operacion.png")
    plt.close(fig)
    print(f"  curvas ROC: {len(curvas)} modelos en {SALIDA_R2_2.relative_to(BASE)}")
    return len(curvas)


# ------------------------------------------------- curvas de aprendizaje

def _nombre_de_corrida(clave: str) -> str:
    """Traduce el identificador de una corrida a una frase legible.

    Las claves son del tipo `e4_atencion_cruzada_token_s42_agr`: experimento,
    arquitectura, semilla y condición de partición. Impresas tal cual llevan al
    documento guiones bajos y jerga del código.
    """
    partes = clave.split("_")
    experimento = partes[0].upper() if partes and partes[0].startswith("e") else ""
    semilla = next((p[1:] for p in partes if p.startswith("s") and p[1:].isdigit()), None)
    condiciones = {
        "agr": "partición agrupada por campaña",
        "sinagr": "partición sin agrupar",
        "sc": "subconjunto trimodal",
        "pond": "con ponderación de clases",
    }
    sufijos = [condiciones[p] for p in partes if p in condiciones]
    fraccion = next((p[4:] for p in partes if p.startswith("frac")), None)
    psem = next((p[4:] for p in partes if p.startswith("psem")), None)
    # El filtro comprueba el patrón completo y no solo la inicial. Con
    # `startswith("s")`, la palabra «solo» de `solo_texto` se tomaba por una
    # semilla y la arquitectura quedaba reducida a «Texto».
    def es_marca(p: str) -> bool:
        return ((p.startswith("s") and p[1:].isdigit())
                or (p.startswith("frac") and p[4:].isdigit())
                or (p.startswith("psem") and p[4:].isdigit())
                or p in condiciones)

    nucleo = [p for p in partes[1:] if not es_marca(p)]
    frase = _rotulo("_".join(nucleo)) if nucleo else clave
    if sufijos:
        frase += ", " + " y ".join(sufijos)
    if fraccion:
        frase += f", con el {fraccion} por ciento de los datos de entrenamiento"
    if psem:
        frase += f", repartición de semilla {psem}"
    if semilla:
        frase += f", semilla de modelo {semilla}"
    return (experimento + ". " + frase) if experimento else frase


def curvas_de_aprendizaje() -> int:
    """Pérdida por paso y métricas de validación por época, corrida a corrida.

    Es lo que el medio de verificación de R1.4 pide de forma literal, y lo que
    permite comprobar la convergencia sin creerse la palabra del informe.
    """
    formato.estilo()
    # Los historiales viven en la carpeta de R1.4, que es donde su medio de
    # verificacion los pide; consolidar_iov.py los traslada alli desde el
    # directorio de trabajo de la cola.
    origen = SALIDA_R1_4.parent / "historiales"
    if not origen.exists():
        print("  curvas de aprendizaje: no hay historiales; se omiten")
        return 0
    SALIDA_R1_4.mkdir(parents=True, exist_ok=True)
    hechas = 0
    for ruta in sorted(origen.glob("*_history.json")):
        h = json.loads(ruta.read_text(encoding="utf-8"))
        # `step_losses` es una lista de registros {step, epoch, loss} y no de
        # numeros sueltos: dibujarla tal cual reventaba el eje.
        registros = h.get("step_losses") or []
        pasos = [r_.get("step", i + 1) for i, r_ in enumerate(registros)]
        perdidas = [r_.get("loss") for r_ in registros]
        epocas = h.get("epoch_metrics") or []
        if not perdidas and not epocas:
            continue
        nombre = ruta.stem.replace("_history", "")

        fig, (izq, der) = plt.subplots(1, 2, figsize=(formato.ANCHO_VERTICAL, 2.7))
        if perdidas:
            izq.plot(pasos, perdidas, lw=0.7, color="#1f77b4", alpha=0.5,
                     label="pérdida por paso")
            if len(perdidas) >= 50:
                k = max(len(perdidas) // 50, 1)
                suave = pd.Series(perdidas).rolling(k, min_periods=1).mean()
                izq.plot(pasos, suave, lw=1.8, color="#08306b",
                         label=f"media móvil ({k} pasos)")
            if epocas:
                # La pérdida de validación se marca al final de cada época, que es
                # cuando se calcula: superponerla sobre la de entrenamiento es lo
                # que permite ver si el ajuste empieza a memorizar.
                px = [e.get("global_step") for e in epocas]
                val = [e.get("val_loss") for e in epocas]
                if all(p is not None for p in px) and any(v is not None for v in val):
                    izq.plot(px, val, marker="o", lw=1.6, color="#d62728",
                             label="pérdida de validación")
            izq.set_xlabel("Paso de entrenamiento")
            izq.set_ylabel("Pérdida")
            izq.set_title("Pérdida")
            izq.legend(fontsize=formato.MINIMO_LEGIBLE)
            izq.grid(alpha=0.3)

        if epocas:
            x = [e.get("epoch", i) + 1 for i, e in enumerate(epocas)]
            for campo, etiqueta, color in (
                    ("val_accuracy", "Exactitud de validación", "#ff7f0e"),
                    ("val_roc_auc", "ROC-AUC de validación", "#2ca02c")):
                y = [e.get(campo) for e in epocas]
                if any(v is not None for v in y):
                    der.plot(x, y, marker="o", lw=1.6, color=color, label=etiqueta)
            der.set_xlabel("Época")
            der.set_ylabel("Métrica de validación")
            der.set_title("Validación por época")
            der.set_xticks(x)
            der.legend(fontsize=formato.MINIMO_LEGIBLE)
            der.grid(alpha=0.3)

        # El titulo se escribe en lenguaje natural: estas laminas pueden acabar
        # dentro del documento, donde ni un identificador con guiones bajos ni el
        # punto medio como separador tienen sitio.
        aviso = (h.get("overfitting_check") or {}).get("note")
        titulo = _nombre_de_corrida(nombre)
        if aviso and "sobreajuste" in aviso.lower():
            titulo += ("; la pérdida de validación sube tras la primera época")
        # El titulo se parte a mano: en una sola linea desbordaba el lienzo y el
        # recorte ajustado lo ensanchaba hasta diez pulgadas, de modo que la
        # lamina volvia a comprimirse al encajarla en la caja de texto.
        import textwrap as _tw

        fig.suptitle("\n".join(_tw.wrap(titulo, 78)), fontsize=formato.TITULO)
        fig.tight_layout()
        fig.savefig(SALIDA_R1_4 / f"curva_{nombre}.png")
        plt.close(fig)
        hechas += 1
    print(f"  curvas de aprendizaje: {hechas} en {SALIDA_R1_4.relative_to(BASE)}")
    return hechas


# ------------------------------------------------------------- arquitectura

def diagrama_arquitectura() -> None:
    """Esquema de la arquitectura, conforme a lo implementado en `model.py`."""
    # El lienzo mide lo que mide la caja de texto de la pagina apaisada que lo
    # aloja, de modo que un rotulo de ocho puntos se imprime a ocho puntos. Antes
    # se dibujaba a 11.5 pulgadas y el documento lo encajaba en 15.9 cm: factor
    # 0.53, con lo que el texto de las cajas salia por debajo de cinco puntos.
    formato.estilo()
    # El diagrama se dibuja al ancho de la caja de texto vertical, que es donde
    # el capitulo lo coloca. Se estira en altura en lugar de en anchura: las
    # cajas mantienen su disposicion y ganan el sitio que necesita el texto sin
    # que la lamina tenga que comprimirse al encajarla.
    escala = 0.78
    fig, ax = plt.subplots(figsize=(formato.ANCHO_VERTICAL,
                                    formato.ANCHO_VERTICAL * 0.80))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def caja(x, y, w, h, texto, color, tamano=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    facecolor=color, edgecolor="#333333", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
                fontsize=tamano * escala, wrap=True)

    def flecha(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.4))

    caja(0.3, 6.9, 9.4, 0.7, "Correo electrónico (RFC 5322)", "#e8eaf6", 11)
    caja(0.3, 5.2, 2.9, 1.2,
         "Rama textual\nDistilBERT\n(secuencia completa de tokens)", "#c5e1a5")
    caja(3.5, 5.2, 2.9, 1.2,
         "Rama estructural\nTokenización por característica\n(6 vectores)", "#ffe0b2")
    caja(6.8, 5.2, 2.9, 1.2,
         "Rama de red\nURL y autenticación\n(12 vectores)", "#b3e5fc")
    flecha(2.0, 6.9, 1.75, 6.4)
    flecha(5.0, 6.9, 4.95, 6.4)
    flecha(8.0, 6.9, 8.25, 6.4)

    caja(3.5, 4.25, 6.2, 0.6,
         "Máscaras de disponibilidad y descarte de modalidad en entrenamiento",
         "#fff9c4", 9)
    flecha(4.95, 5.2, 4.95, 4.85)
    flecha(8.25, 5.2, 8.25, 4.85)

    caja(3.5, 3.05, 6.2, 0.9,
         "Primera etapa de fusión (jerárquica)\n"
         "Codificador Transformer sobre 18 vectores más el centinela de ausencia",
         "#d1c4e9")
    flecha(6.6, 4.25, 6.6, 3.95)

    caja(0.3, 1.75, 9.4, 0.95,
         "Segunda etapa de fusión: atención cruzada\n"
         "el texto actúa como consulta y las modalidades como clave y valor",
         "#f8bbd0", 10)
    flecha(1.75, 5.2, 1.75, 2.7)
    flecha(6.6, 3.05, 6.6, 2.7)

    caja(3.1, 0.45, 3.8, 0.85,
         "Promediado enmascarado\ny clasificación binaria", "#cfd8dc", 10)
    flecha(5.0, 1.75, 5.0, 1.3)
    ax.text(5.0, 0.12, "Legítimo  o  phishing", ha="center",
            fontsize=10 * escala, style="italic")

    fig.tight_layout()
    for destino in (FIGURAS_ARQ, SALIDA_R1_3):
        destino.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino / "arquitectura_multimodal.png")
    plt.close(fig)
    print(f"  diagrama de arquitectura en {SALIDA_R1_3.relative_to(BASE)}")


def _rotulo(clave: str) -> str:
    """Nombre legible de una serie, a partir de la clave con que viaja en el informe."""
    if clave in NOMBRES:
        return NOMBRES[clave]
    legible = clave.replace("_", " ").replace("[control]", "").strip()
    reemplazos = {
        "atencion cruzada (token)": "Atención cruzada por token (propuesta)",
        "atencion cruzada (modalidad)": "Atención cruzada por modalidad",
        "concatenacion tardia": "Concatenación tardía",
        "fusor MLP": "Fusor MLP",
        "mezcla de expertos": "Mezcla de expertos",
        "solo texto": "Unimodal de texto",
        "solo estructura": "Unimodal de estructura (control)",
        "solo red": "Unimodal de red (control)",
        "las tres": "Las tres modalidades",
        "sin estructura": "Sin estructura",
        "sin red": "Sin red",
        "centinela + descarte": "Centinela y descarte",
        "sin centinela": "Descarte sin centinela",
    }
    return reemplazos.get(legible, legible[:1].upper() + legible[1:])


def _barras(destino: Path, nombre: str, titulo: str, etiquetas, medias,
            desviaciones, resaltar=None, etiqueta_resaltada="", ylabel="F1",
            copias=()) -> None:
    """Lámina de barras rehecha desde el informe, sin reentrenar.

    Delega en `formato.barras`, que es la misma función que usa la cola: tener
    dos implementaciones del mismo gráfico las hizo divergir, y las láminas de un
    mismo capítulo llegaron a rotular de forma distinta la misma barra.
    """
    formato.barras(((destino, nombre + ".png"), *copias), nombre, titulo,
                   etiquetas, medias, desviaciones, ylabel=ylabel,
                   resaltar=resaltar, etiqueta_resaltada=etiqueta_resaltada,
                   envolver=0)


def laminas_de_barras() -> int:
    """Rehace las láminas de barras de E2 y E3 con rótulos y leyendas correctos.

    Se rehacen aquí y no en sus experimentos porque hacerlo allí exigiría
    reentrenar, y las cifras no cambian: lo que cambia es lo que la lámina dice de
    ellas. La de E2 rotulaba «arquitectura propuesta» la barra del fusor por
    perceptrón multicapa, que es su línea base de comparación, y la de E3 rotulaba
    así una condición de evaluación. La leyenda, además, tapaba el valor impreso
    sobre las últimas barras.
    """
    print("Láminas de barras de E2 y E3:")
    e2 = json.loads((EXP / "e2" / "e2.json").read_text(encoding="utf-8"))
    res = e2.get("resultados", {})
    base = e2.get("linea_base")
    claves = list(res)
    _barras(EXP / "e2", "e2_mecanismo_fusion",
            "E2. Desempeño por mecanismo de fusión\n"
            "subconjunto trimodal, media de tres semillas",
            [_rotulo(k) for k in claves],
            [res[k].get("f1", 0.0) for k in claves],
            [res[k].get("f1_desv", 0.0) for k in claves],
            resaltar=claves.index(base) if base in claves else None,
            etiqueta_resaltada="línea base de comparación",
            copias=((SALIDA_R2_2, "mecanismos_de_fusion.png"),))

    e3 = json.loads((EXP / "e3" / "e3.json").read_text(encoding="utf-8"))
    deg = e3.get("degradacion", {})
    claves3 = list(deg)
    # Aquí las claves nombran CONDICIONES de evaluación y no arquitecturas: «solo
    # texto» es el mismo modelo con las modalidades no textuales ocultas, no el
    # unimodal de texto, que es otro modelo entrenado sin ellas.
    condiciones = {"las tres": "Las tres disponibles", "sin estructura": "Sin estructura",
                   "sin red": "Sin red", "solo texto": "Solo texto"}
    _barras(EXP / "e3", "e3_degradacion",
            "E3. Degradación al ocultar modalidades en evaluación\n"
            "mecanismo: " + _rotulo(str(e3.get("mejor_mecanismo", ""))),
            [condiciones.get(k, _rotulo(k)) for k in claves3],
            [deg[k].get("f1", 0.0) for k in claves3],
            [deg[k].get("f1_desv", 0.0) for k in claves3],
            resaltar=claves3.index("solo texto") if "solo texto" in claves3 else None,
            etiqueta_resaltada="condición sin modalidades no textuales",
            copias=((SALIDA_R1_3, "degradacion_por_ausencia.png"),))
    return 0


def lamina_adversaria() -> int:
    """Rehace la rejilla adversaria en dos filas y con rótulos legibles.

    La versión anterior ponía la leyenda dentro de los ejes y con las claves del
    código, guiones bajos incluidos, y disponía los cuatro operadores en una sola
    fila de 2944 píxeles de ancho: impresa a la anchura de la columna, esa leyenda
    quedaba por debajo de cinco puntos.
    """
    print("Rejilla de degradación adversaria:")
    e3 = json.loads((EXP / "e3" / "e3.json").read_text(encoding="utf-8"))
    rejilla = e3.get("degradacion_adversaria_del_texto", {})
    if not rejilla:
        print("  sin datos de degradación adversaria")
        return 0
    formato.estilo()
    operadores = list(rejilla)
    filas = 2 if len(operadores) > 2 else 1
    columnas = (len(operadores) + filas - 1) // filas
    # El lienzo mide lo que mide la caja de texto. Con 5.2 pulgadas por columna
    # la rejilla llegaba a 10.3 y el documento la comprimia a 0.61, de modo que
    # la leyenda de ocho puntos se imprimia por debajo de cinco.
    fig, ejes = plt.subplots(filas, columnas, sharey=True,
                             figsize=(formato.ANCHO_VERTICAL,
                                      formato.ANCHO_VERTICAL * 0.46 * filas))
    ejes = np.atleast_1d(ejes).ravel()
    colores = {
        "atencion cruzada (token)": "#C44E52",
        "atencion cruzada (modalidad)": "#4C72B0",
        "fusor MLP": "#55A868", "concatenacion tardia": "#937860",
        "mezcla de expertos": "#8172B3", "solo texto": "#DA8BC3",
        "solo estructura [control]": "#9AA5B1", "solo red [control]": "#B7C0C9",
        "b1_tfidf_lr": "#DD8452", "b5_multimodal_clasico": "#CCB974",
        "b3_rf_estructura": "#9AA5B1", "b4_red": "#B7C0C9",
    }
    marcas = {"solo estructura [control]": ":", "solo red [control]": ":",
              "b3_rf_estructura": ":", "b4_red": ":",
              "b1_tfidf_lr": "s--", "b5_multimodal_clasico": "s--"}
    # `propuesta` y `b5_fusion_clasica` son ALIAS que el experimento añade a cada
    # celda para que otros guiones no dependan del nombre del mecanismo elegido.
    # Dibujarlos junto a sus originales trazaba dos veces la misma serie y dejaba
    # dos entradas de leyenda para una sola curva.
    ALIAS = {"propuesta": "atencion cruzada (token)",
             "b5_fusion_clasica": "b5_multimodal_clasico"}
    todas = list(next(iter(next(iter(rejilla.values())).values())))
    series = [s for s in todas
              if not (s in ALIAS and ALIAS[s] in todas)]
    manejadores: dict[str, object] = {}
    for eje, operador in zip(ejes, operadores):
        celdas = rejilla[operador]
        xs = [float(k) for k in celdas]
        for serie in series:
            ys = [c[serie]["f1"] for c in celdas.values() if serie in c]
            if len(ys) != len(xs):
                continue
            linea, = eje.plot(xs, ys, marcas.get(serie, "o-"),
                              color=colores.get(serie), markersize=4, linewidth=1.4)
            manejadores.setdefault(_rotulo(serie), linea)
        eje.set_title(operador.replace("_", " "))
        eje.set_xlabel("intensidad")
        eje.grid(alpha=0.3)
        eje.set_ylim(0, 1.05)
    for eje in ejes[len(operadores):]:
        eje.axis("off")
    for i in range(0, len(ejes), columnas):
        ejes[i].set_ylabel("F1")
    fig.suptitle("E3. Degradación adversaria del texto, aplicada solo en evaluación")
    # Se reserva la banda inferior ANTES de colocar la leyenda. Con `tight_layout`
    # posterior, matplotlib no cuenta la leyenda de figura y esta caía sobre los
    # rótulos del eje inferior.
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    fig.legend(list(manejadores.values()), list(manejadores), loc="lower center",
               ncol=3, fontsize=formato.MINIMO_LEGIBLE, frameon=False)
    formato.guardar(fig, ((EXP / "e3", "e3_degradacion_adversaria.png"),
                          (SALIDA_R1_3, "degradacion_adversaria.png")),
                    "e3_degradacion_adversaria")
    return 0


from estructura_del_trabajo import (  # noqa: E402
    FASES, PAQUETES, RAIZ)




def edt() -> int:
    """Dibuja la estructura de descomposición del trabajo.

    La lámina anterior trazaba los enlaces desde la raíz en diagonal hasta cada
    caja, de modo que varios cruzaban por encima de otras cajas y uno atravesaba
    una barra de título. Aquí el trazado es ortogonal: la raíz baja a un eje
    horizontal y de ese eje caen las verticales, que por construcción no pueden
    cruzarse. Se retiran además el punto medio y la flecha, que el proyecto
    prohíbe, y los rótulos de fase dejan de ser una palabra dentro del título para
    pasar a una banda que agrupa los paquetes que de verdad las componen.

    La lámina NO enumera las tareas. Enumerarlas obligaba a un lienzo de dieciséis
    pulgadas que la página encajaba con un factor de 0.60, con lo que el texto de
    las tareas se imprimía a 4.6 puntos y no había manera de leerlo. Las tareas
    están en la tabla de paquetes, que va inmediatamente debajo y en la misma
    página apaisada, de modo que no se pierde información: se reparte entre la
    figura, que muestra la jerarquía, y la tabla, que muestra el detalle.
    """
    import textwrap

    from matplotlib.patches import FancyBboxPatch as Caja

    print("Estructura de descomposición del trabajo:")
    formato.estilo()
    n = len(PAQUETES)
    ancho_caja, hueco = 1.0, 0.16
    paso = ancho_caja + hueco
    total = n * paso - hueco

    fig, ax = plt.subplots(figsize=(formato.ANCHO_APAISADO,
                                    formato.ANCHO_APAISADO * 0.40))
    ax.set_xlim(-0.25, total + 0.25)
    ax.set_ylim(0, 10)
    ax.axis("off")

    azul, gris, borde = "#1F3A5F", "#F2F4F7", "#9AA5B1"

    # Raíz
    centro = total / 2
    ax.add_patch(Caja((centro - 3.1, 8.55), 6.2, 1.15,
                      boxstyle="round,pad=0.02,rounding_size=0.08",
                      facecolor=azul, edgecolor=azul))
    ax.text(centro, 9.12, "\n".join(textwrap.wrap(RAIZ, 46)), ha="center",
            va="center", color="white", fontsize=formato.TITULO,
            fontweight="bold", linespacing=1.35)

    # Eje ortogonal: una vertical desde la raíz y una horizontal de la que cuelgan
    # las verticales de cada paquete. Ningún segmento diagonal, ningún cruce.
    eje_y = 7.85
    ax.plot([centro, centro], [8.55, eje_y], color=borde, lw=1.3,
            solid_capstyle="butt")
    x_centros = [i * paso + ancho_caja / 2 for i in range(n)]
    ax.plot([x_centros[0], x_centros[-1]], [eje_y, eje_y], color=borde, lw=1.3)

    # La cabecera se dimensiona por el título MÁS largo: con una altura fija, el
    # paquete de nombre más extenso derramaba su última línea fuera de la caja.
    def _lineas(texto: str, ancho: int) -> int:
        return len(textwrap.wrap(texto, ancho))

    alto_linea = 0.62
    alto_cabecera = max(_lineas(t, 17) for t, _, _, _ in PAQUETES) * alto_linea + 0.42
    alto_pie = max(_lineas(r, 15) for _, _, r, _ in PAQUETES) * alto_linea + 0.75
    cima = 7.0
    for x, (titulo, _tareas, resultado, _fase) in zip(x_centros, PAQUETES):
        izq = x - ancho_caja / 2
        ax.plot([x, x], [eje_y, cima], color=borde, lw=1.3, solid_capstyle="butt")

        ax.add_patch(Caja((izq, cima - alto_cabecera), ancho_caja, alto_cabecera,
                          boxstyle="square,pad=0", facecolor=azul, edgecolor=azul))
        ax.text(x, cima - alto_cabecera / 2, "\n".join(textwrap.wrap(titulo, 17)),
                ha="center", va="center", color="white", fontsize=formato.ROTULO,
                fontweight="bold", linespacing=1.3)

        pie_y = cima - alto_cabecera - alto_pie
        ax.add_patch(Caja((izq, pie_y), ancho_caja, alto_pie,
                          boxstyle="square,pad=0", facecolor=gris, edgecolor=borde,
                          linewidth=0.9))
        ax.text(x, pie_y + alto_pie / 2,
                "Entrega\n" + "\n".join(textwrap.wrap(resultado, 15)),
                ha="center", va="center", fontsize=formato.MINIMO_LEGIBLE,
                color="#14202E", linespacing=1.35)

    # Bandas de fase, bajo los paquetes que realmente las componen.
    banda_y = pie_y - 1.15
    alto_banda = 0.78
    inicio = 0
    for i in range(1, n + 1):
        if i == n or PAQUETES[i][3] != PAQUETES[inicio][3]:
            izq = inicio * paso
            der = (i - 1) * paso + ancho_caja
            ax.add_patch(Caja((izq, banda_y), der - izq, alto_banda,
                              boxstyle="square,pad=0", facecolor="white",
                              edgecolor=azul, linewidth=1.1))
            ax.text((izq + der) / 2, banda_y + alto_banda / 2,
                    FASES[PAQUETES[inicio][3]], ha="center", va="center",
                    fontsize=formato.ROTULO, color=azul, fontweight="bold")
            inicio = i
    ax.set_ylim(banda_y - 0.35, 10)

    fig.tight_layout(pad=0.3)
    FIGURAS_ARQ.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_ARQ / "edt.png", facecolor="white")
    plt.close(fig)
    print("  edt.png")
    return 0


CARACTERISTICAS = {
    "num_images": "Número de imágenes",
    "total_nodos_dom": "Nodos del árbol del documento",
    "url_has_shortener": "Emplea un acortador de enlaces",
    "num_links": "Número de enlaces",
    "profundidad_dom": "Profundidad del árbol del documento",
    "has_html": "El cuerpo trae marcado",
    "has_ip_link": "Enlace escrito como dirección numérica",
    "url_has_at_symbol": "Enlace con arroba",
    "url_max_digit_ratio": "Proporción de dígitos en el enlace",
    "num_urls_metadata": "Número de enlaces en las cabeceras",
    "url_domain_max_entropy": "Entropía del dominio",
    "url_max_length": "Longitud del enlace más largo",
    "word_count": "Número de palabras",
    "url_max_subdomain_count": "Número de subdominios",
    "url_has_suspicious_tld": "Dominio de primer nivel infrecuente",
}


def auditoria_de_caracteristicas() -> int:
    """Qué informa cada característica no textual: la clase o su procedencia.

    Es la lámina que sostiene la limitación central del trabajo, y la sección de
    validez no tenía ninguna. Se rehace aquí, desde el informe, porque la que emite
    el experimento rotula las barras con los nombres de columna del corpus, guiones
    bajos incluidos, y omite una de las quince características.
    """
    print("Auditoría de características:")
    e5 = json.loads((EXP / "e5" / "e5.json").read_text(encoding="utf-8"))
    filas = e5.get("auditoria_de_caracteristicas") or []
    if not filas:
        print("  sin auditoría de características")
        return 0
    filas = sorted(filas, key=lambda f: f.get("mi_coleccion") or 0)
    nombres = [CARACTERISTICAS.get(f["caracteristica"],
                                   f["caracteristica"].replace("_", " "))
               for f in filas]
    coleccion = [f.get("mi_coleccion") or 0 for f in filas]
    clase = [f.get("mi_clase_dada_la_coleccion") or 0 for f in filas]
    formato.estilo()
    y = np.arange(len(filas))
    fig, ax = plt.subplots(figsize=(formato.ANCHO_VERTICAL,
                                    formato.alto_por_filas(len(filas), 0.34, 1.5)))
    ax.barh(y + 0.20, coleccion, height=0.38, color="#C44E52",
            edgecolor="#7A2E31", linewidth=0.5,
            label="sobre la colección de procedencia")
    # El trazado distingue las dos series sin depender del color: impresa en
    # blanco y negro, la luminancia del rojo y la del azul son casi la misma y la
    # lámina dejaba de decir nada.
    ax.barh(y - 0.20, clase, height=0.38, color="#4C72B0", hatch="///",
            edgecolor="#1A2733", linewidth=0.5,
            label="sobre la clase, fijada la colección")
    # Siete características informan CERO sobre la clase, y una barra de longitud
    # cero es indistinguible de una barra ausente: quien lea la lámina no sabe si
    # el valor es nulo o si falta el dato. Se marca el cero explícitamente.
    for pos, v in zip(y, clase):
        if v == 0:
            ax.plot([0], [pos - 0.20], marker="|", markersize=6, color="#1A2733")
            ax.text(max(coleccion) * 0.012, pos - 0.20, "0", va="center",
                    ha="left", fontsize=formato.MINIMO_LEGIBLE - 1,
                    color="#1A2733")
    ax.set_yticks(y)
    ax.set_yticklabels(nombres)
    ax.set_xlim(0, max(coleccion) * 1.06)
    ax.set_xlabel("Información mutua (nats)")
    ax.set_title("Qué informa cada característica no textual", pad=10)
    ax.grid(axis="x", alpha=0.3)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95,
              fontsize=formato.MINIMO_LEGIBLE)
    fig.tight_layout()
    formato.guardar(fig, ((EXP / "e5", "e5_auditoria_caracteristicas.png"),
                          (SALIDA_R2_2, "auditoria_de_caracteristicas.png")),
                    "e5_auditoria_caracteristicas")
    print(f"  {len(filas)} características")
    return 0


def comparativas() -> int:
    """Rehace desde los informes las láminas de E0, E1 y E4 que el capítulo usa.

    Las emite la cola en la misma ejecución que produce el resultado, y así debe
    ser: es lo que impide que una lámina describa una corrida y la tabla de al
    lado describa otra. Pero las que quedaron en el repositorio se dibujaron con
    lienzos de nueve a trece pulgadas, y el documento las encaja en 15.9 cm: la
    del cuadro comparativo se imprimía con un factor de 0.48. Se rehacen aquí, sin
    reentrenar y desde el mismo informe del que salen las tablas, con la geometría
    de impresión que fija `formato`. Cuando la cola vuelva a correr las emitirá ya
    con esa geometría, porque la comparten.
    """
    print("Láminas comparativas de E0, E1 y E4:")
    formato.estilo()

    # E0. Cobertura de las modalidades no textuales, por colección.
    e0 = json.loads((EXP / "e0" / "e0.json").read_text(encoding="utf-8"))
    por_coleccion = (e0.get("cobertura_modal_por_coleccion") or {}).get("por_coleccion", {})
    if por_coleccion:
        fuentes = sorted(por_coleccion)
        x = np.arange(len(fuentes))
        fig, ax = plt.subplots(figsize=(formato.ANCHO_VERTICAL,
                                        formato.ANCHO_VERTICAL * 0.46))
        for desplazamiento, campo, etiqueta, color, trama in (
                (-0.27, "pct_estructura", "Estructura (HTML/DOM)", "#4C72B0", ""),
                (0.0, "pct_red", "Red (URL)", "#DD8452", "///"),
                (0.27, "pct_trimodal", "Ambas", "#55A868", "...")):
            ax.bar(x + desplazamiento, [por_coleccion[f][campo] for f in fuentes],
                   0.27, label=etiqueta, color=color, hatch=trama,
                   edgecolor="#22303F", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(fuentes, rotation=20, ha="right")
        ax.set_ylabel("Cobertura (%)")
        ax.set_ylim(0, 108)
        ax.set_title("Disponibilidad de las modalidades no textuales, por colección")
        ax.legend(fontsize=formato.MINIMO_LEGIBLE, ncol=3, loc="upper center")
        ax.grid(axis="y", alpha=0.3)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        fig.tight_layout()
        formato.guardar(fig, EXP / "e0", "e0_cobertura_modal")

    # E1. La multimodalidad frente a los unimodales, sobre el subconjunto trimodal.
    e1 = json.loads((EXP / "e1" / "e1.json").read_text(encoding="utf-8"))
    res1 = e1.get("resultados", {})
    claves1 = list(res1)
    if claves1:
        _barras(EXP / "e1", "e1_multimodalidad",
                "E1. La multimodalidad frente a los unimodales y al piso clásico\n"
                "F1 sobre el subconjunto trimodal, media de tres semillas",
                [NOMBRES.get(k, _rotulo(k)) for k in claves1],
                [res1[k].get("f1", 0.0) for k in claves1],
                [res1[k].get("f1_desv", 0.0) for k in claves1],
                resaltar=0, etiqueta_resaltada="arquitectura propuesta")

    # E4. Las doce arquitecturas sobre el corpus completo, y la curva de aprendizaje.
    e4 = json.loads((EXP / "e4" / "e4.json").read_text(encoding="utf-8"))
    cuadro = e4.get("cuadro_comparativo_corpus_completo", {})
    referencia = e4.get("arquitectura_de_referencia", "atencion_cruzada_token")
    if cuadro:
        orden = sorted(cuadro, key=lambda k: -cuadro[k]["f1"])
        _barras(EXP / "e4", "e4_cuadro_comparativo",
                "E4. Todas las arquitecturas sobre el corpus completo\n"
                "F1 con partición agrupada por campaña, media de tres semillas",
                [NOMBRES.get(k, _rotulo(k)) for k in orden],
                [cuadro[k]["f1"] for k in orden],
                [cuadro[k].get("f1_desv", 0.0) for k in orden],
                resaltar=orden.index(referencia) if referencia in orden else None,
                etiqueta_resaltada="arquitectura propuesta")
    curva = e4.get("curva_de_aprendizaje") or {}
    if curva:
        _barras(EXP / "e4", "e4_curva_aprendizaje",
                "E4. Curva de aprendizaje: ¿limitado por datos o saturado?",
                [f"{k} del corpus\n({v['n_entrenamiento']:,} correos)".replace(",", " ")
                 for k, v in curva.items()],
                [v["f1"] for v in curva.values()], [0.0] * len(curva))
    return 0


TAREAS = {
    "comparativas": comparativas,
    "matrices": matrices_de_confusion,
    "curvas_roc": curvas_roc,
    "curvas_aprendizaje": curvas_de_aprendizaje,
    "arquitectura": diagrama_arquitectura,
    "barras": laminas_de_barras,
    "adversaria": lamina_adversaria,
    "caracteristicas": auditoria_de_caracteristicas,
    "edt": edt,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solo", choices=sorted(TAREAS), default=None,
                   help="generar una sola familia de figuras")
    args = p.parse_args()

    print("Figuras de los medios de verificación:")
    for nombre, tarea in TAREAS.items():
        if args.solo in (None, nombre):
            tarea()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
