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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

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

        fig, ax = plt.subplots(figsize=(5.0, 4.3))
        im = ax.imshow(cm, cmap="Blues")
        total = cm.sum()
        umbral = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:,}\n({100 * cm[i, j] / total:.2f}%)",
                        ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > umbral else "black")
        ax.set_xticks([0, 1], ["Legítimo", "Phishing"])
        ax.set_yticks([0, 1], ["Legítimo", "Phishing"])
        ax.set_xlabel("Predicción del modelo")
        ax.set_ylabel("Clase real")
        ax.set_title(f"{NOMBRES.get(clave, clave)}\nsemilla {semilla}, corpus completo",
                     fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(destino / f"matriz_confusion_{clave}.png", dpi=150,
                    bbox_inches="tight")
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
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        for k in presentes:
            fpr, tpr, auc = curvas[k]
            ax.plot(fpr, tpr, lw=1.6, label=f"{NOMBRES.get(k, k)} (AUC {auc:.4f})")
        ax.plot([0, 1], [0, 1], "--", lw=1, color="#999999", label="Azar")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title(f"Curvas ROC sobre el corpus completo\nmodelos de familia {familia}",
                     fontsize=11)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(SALIDA_R2_2 / f"curvas_roc_{etiqueta}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # Vista logarítmica: al saturar la tarea, la esquina superior izquierda es la
    # única región donde las curvas se distinguen, y es además la que describe el
    # punto de operación de una pasarela de correo.
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for k, (fpr, tpr, auc) in curvas.items():
        ax.plot(np.clip(fpr, 1e-4, 1), tpr, lw=1.5, label=NOMBRES.get(k, k))
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("Tasa de falsos positivos (escala logarítmica)")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC en la región de operación\ncorpus completo, todas las familias",
                 fontsize=11)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(SALIDA_R2_2 / "curvas_roc_region_de_operacion.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  curvas ROC: {len(curvas)} modelos en {SALIDA_R2_2.relative_to(BASE)}")
    return len(curvas)


# ------------------------------------------------- curvas de aprendizaje

def curvas_de_aprendizaje() -> int:
    """Pérdida por paso y métricas de validación por época, corrida a corrida.

    Es lo que el medio de verificación de R1.4 pide de forma literal, y lo que
    permite comprobar la convergencia sin creerse la palabra del informe.
    """
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

        fig, (izq, der) = plt.subplots(1, 2, figsize=(10.4, 3.8))
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
            izq.set_title("Pérdida", fontsize=10)
            izq.legend(fontsize=8)
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
            der.set_title("Validación por época", fontsize=10)
            der.set_xticks(x)
            der.legend(fontsize=8)
            der.grid(alpha=0.3)

        aviso = (h.get("overfitting_check") or {}).get("note")
        titulo = nombre.replace("_", " ")
        if aviso and "sobreajuste" in aviso.lower():
            titulo += "  ·  la pérdida de validación sube tras la primera época"
        fig.suptitle(titulo, fontsize=10)
        fig.tight_layout()
        fig.savefig(SALIDA_R1_4 / f"curva_{nombre}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        hechas += 1
    print(f"  curvas de aprendizaje: {hechas} en {SALIDA_R1_4.relative_to(BASE)}")
    return hechas


# ------------------------------------------------------------- arquitectura

def diagrama_arquitectura() -> None:
    """Esquema de la arquitectura, conforme a lo implementado en `model.py`."""
    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def caja(x, y, w, h, texto, color, tamano=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    facecolor=color, edgecolor="#333333", linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
                fontsize=tamano, wrap=True)

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
    ax.text(5.0, 0.12, "Legítimo  o  phishing", ha="center", fontsize=10, style="italic")

    fig.tight_layout()
    for destino in (FIGURAS_ARQ, SALIDA_R1_3):
        destino.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino / "arquitectura_multimodal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  diagrama de arquitectura en {SALIDA_R1_3.relative_to(BASE)}")


TAREAS = {
    "matrices": matrices_de_confusion,
    "curvas_roc": curvas_roc,
    "curvas_aprendizaje": curvas_de_aprendizaje,
    "arquitectura": diagrama_arquitectura,
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
