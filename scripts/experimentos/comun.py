"""
Piezas compartidas por los cinco experimentos.

Existe para que la partición, las métricas y la emisión de artefactos sean
literalmente el mismo código en todos. La revisión del pipeline encontró que las
asimetrías de preprocesamiento entre fuentes se convierten en huellas del origen;
lo mismo vale para las asimetrías de protocolo entre experimentos, que convierten
una diferencia de método en una diferencia aparente de arquitectura.

Cada experimento emite su JSON y su figura **en la misma ejecución** que produce
el resultado. No se generan después con guiones aparte: así fue como una tabla del
capítulo llegó a declarar seis mediciones cuando ya había nueve.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src"))

from phishing_baseline.evaluation import compute_metrics  # noqa: E402
from phishing_pipeline.features.vectorizer import compute_modality_availability  # noqa: E402

CORPUS = BASE / "data" / "Datasets_Procesados" / "Dataset_Real.parquet"
SALIDA = BASE / "medios_de_verificacion" / "experimentos"

# Semillas de modelo. Tres, porque las capas de fusión se inicializan al azar y
# sin conocer la dispersión que introduce esa inicialización no puede decidirse si
# una diferencia entre arquitecturas es real. Se comprobó que importa: con dos
# semillas, un efecto parecía firme en los tres pliegues y la tercera lo desmintió.
SEMILLAS = (42, 43, 44)

# Semilla de partición, distinta de las de modelo y fija en todos los
# experimentos: E1 y E2 se comparan entre sí y eso exige el mismo reparto.
SEMILLA_PARTICION = 7


@dataclass
class Particion:
    """Índices de entrenamiento, validación y prueba sobre un mismo marco."""

    entrenamiento: np.ndarray
    validacion: np.ndarray
    prueba: np.ndarray

    def resumen(self) -> dict:
        return {
            "entrenamiento": int(len(self.entrenamiento)),
            "validacion": int(len(self.validacion)),
            "prueba": int(len(self.prueba)),
        }


def cargar_corpus() -> pd.DataFrame:
    """Corpus con las banderas de disponibilidad ya calculadas."""
    if not CORPUS.exists():
        raise SystemExit(
            f"No existe {CORPUS}. Reconstruir con "
            "`python -m phishing_pipeline.corpus_real --sin-descargar`."
        )
    df = pd.read_parquet(CORPUS)
    disp = compute_modality_availability(df)
    return pd.concat([df.reset_index(drop=True), disp.reset_index(drop=True)], axis=1)


def subconjunto_trimodal(df: pd.DataFrame) -> pd.DataFrame:
    """Filas con las tres modalidades, que es donde se responde si la fusión aporta.

    Restringir aquí es lo que pidió el asesor y tiene una razón precisa: sobre el
    corpus completo, la comparación entre multimodal y unimodal mezcla dos efectos
    --si la fusión aprovecha las modalidades, y si el modelo tolera que falten--
    y no permite atribuir el resultado a ninguno de los dos.
    """
    return df[df["has_structure_modality"] & df["has_network_modality"]].copy()


def particion_agrupada(df: pd.DataFrame, semilla: int = SEMILLA_PARTICION,
                       proporciones: tuple[float, float, float] = (0.8, 0.1, 0.1),
                       agrupar: bool = True) -> Particion:
    """
    Reparto 80/10/10 estratificado por clase y agrupado por campaña.

    Agrupar NO reduce el corpus: es una restricción sobre el reparto, no un
    subconjunto. Lo único que impide es que una misma campaña caiga a ambos lados.
    Sin ella, los casi-duplicados de una campaña quedan en entrenamiento y en
    prueba, el modelo memoriza la plantilla y la métrica se infla. Con 29,564
    conglomerados para 44,100 correos, hay casi-duplicado de sobra para que
    importe.

    `agrupar=False` existe para poder MEDIR cuánto infla no agrupar, que es lo que
    hace la mayor parte de la literatura del área.
    """
    rng = np.random.default_rng(semilla)
    tr, va, te = [], [], []

    clave = "template_cluster_id" if agrupar else None
    for etiqueta in sorted(df["label"].unique()):
        sub = df[df["label"] == etiqueta]
        if clave and clave in sub.columns:
            # Se reparten conglomerados, pero contando FILAS: los tamaños son
            # muy dispares --hay conglomerados de 64 correos y 235 de uno solo--
            # y repartir por número de conglomerados dejaba 58 filas de
            # validación donde debía haber unas 2,000.
            # Relleno defensivo: hoy la cobertura es del 100%, pero una fila sin
            # conglomerado debe formar el suyo propio y no compartir uno con
            # todas las demás sin asignar, que las metería en el mismo lado.
            grupos = sub[clave].astype(object).where(
                sub[clave].notna(),
                pd.Series([f"_solo_{i}" for i in sub.index], index=sub.index),
            ).to_numpy()
            unicos, cuentas = np.unique(grupos, return_counts=True)
            orden = rng.permutation(len(unicos))
            unicos, cuentas = unicos[orden], cuentas[orden]
            objetivo = np.cumsum(cuentas) / cuentas.sum()
            destino_grupo = np.where(
                objetivo <= proporciones[0], 0,
                np.where(objetivo <= proporciones[0] + proporciones[1], 1, 2))
            reparto = dict(zip(unicos, destino_grupo))
            destino = np.array([reparto[g] for g in grupos])
        else:
            destino = rng.choice(
                [0, 1, 2], size=len(sub), p=list(proporciones)
            )
        idx = sub.index.to_numpy()
        tr.append(idx[destino == 0]); va.append(idx[destino == 1]); te.append(idx[destino == 2])

    return Particion(np.concatenate(tr), np.concatenate(va), np.concatenate(te))


def metricas(y_true, y_pred, y_proba=None) -> dict:
    """Todas las métricas del proyecto, para no tener que reelegir después.

    Se calculan todas siempre --incluidas calibración y tasa de falsos positivos a
    detección fija-- porque volver a ejecutar un entrenamiento para obtener una
    métrica que no se pidió cuesta horas de GPU, y porque el capítulo necesita
    poder responder preguntas que aún no se han formulado.
    """
    return compute_metrics(np.asarray(y_true), np.asarray(y_pred),
                           None if y_proba is None else np.asarray(y_proba))


# Puntos de operación declarados A PRIORI, por requisito de despliegue y no por
# lo que favorezca a ningún modelo. Una pasarela procesa mucho más correo legítimo
# que fraudulento: un 1% de falsos positivos significa bloquear correo real todos
# los días. Barrer umbrales y quedarse con el mejor sería elegir el número después
# de verlo; por eso se fijan aquí y se informan siempre los dos.
FPR_OBJETIVO = (0.01, 0.001)


def tpr_a_fpr(y_true, y_proba, objetivos=FPR_OBJETIVO) -> dict:
    """Detección alcanzable sin pasar de una tasa de falsos positivos dada.

    Informa además la RESOLUCIÓN del conjunto de prueba, que es un límite que se
    suele callar: con N negativos, el menor FPR distinto de cero que puede
    medirse es 1/N. Pedir un FPR por debajo de eso no da un número conservador,
    da un número sin sentido, y conviene que lo diga el informe y no el tribunal.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true)
    n_neg = int((y_true == 0).sum())
    salida: dict = {"n_negativos": n_neg,
                    "fpr_minimo_medible": round(1.0 / n_neg, 6) if n_neg else None}
    if y_proba is None or n_neg == 0:
        return salida
    fpr, tpr, umbrales = roc_curve(y_true, np.asarray(y_proba))
    for obj in objetivos:
        clave = f"tpr_a_fpr_{obj:g}"
        if obj < 1.0 / n_neg:
            salida[clave] = None
            salida[f"{clave}_nota"] = (
                f"no estimable: el conjunto de prueba solo resuelve hasta "
                f"FPR={1.0 / n_neg:.5f}")
            continue
        viables = np.where(fpr <= obj)[0]
        i = viables[-1]
        salida[clave] = round(float(tpr[i]), 4)
        salida[f"{clave}_umbral"] = round(float(umbrales[i]), 6)
    return salida


def _metrica_rapida(y_true: np.ndarray, y_pred: np.ndarray, clave: str) -> float:
    """La métrica pedida y solo esa, con aritmética de conteos.

    El bootstrap la evalúa miles de veces. Llamar al cálculo completo --que
    incluye ROC, Brier y calibración-- multiplicaba el coste por veinte y hacía
    que reconstruir un informe tardara más que el entrenamiento que evita.
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    if clave == "f1":
        d = 2 * tp + fp + fn
        return (2 * tp / d) if d else 0.0
    if clave == "accuracy":
        return (tp + tn) / len(y_true) if len(y_true) else 0.0
    if clave == "precision":
        return tp / (tp + fp) if (tp + fp) else 0.0
    if clave == "recall":
        return tp / (tp + fn) if (tp + fn) else 0.0
    raise KeyError(f"metrica no soportada en bootstrap: {clave}")


def intervalo_bootstrap(y_true, y_pred, y_proba=None, clave: str = "f1",
                        n: int = 2000, semilla: int = 7) -> dict:
    """Intervalo de confianza al 95% remuestreando el CONJUNTO DE PRUEBA.

    Es una fuente de incertidumbre distinta de la dispersión entre semillas y no
    se sustituyen: las semillas miden el ruido de optimización --qué tan estable
    es el ajuste-- y el bootstrap mide el ruido de muestreo --qué tan bien
    representa este conjunto de prueba a la población--. Con pocos errores el
    segundo domina, y omitirlo hace parecer resueltas diferencias que no lo son.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    y_proba = None if y_proba is None else np.asarray(y_proba)
    rng = np.random.default_rng(semilla)
    n_obs = len(y_true)
    valores = []
    for _ in range(n):
        i = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y_true[i])) < 2:
            continue
        valores.append(_metrica_rapida(y_true[i], y_pred[i], clave))
    if not valores:
        return {"metrica": clave, "ic95": None}
    v = np.asarray(valores, dtype=float)
    return {
        "metrica": clave, "n_remuestreos": len(v),
        # El valor sobre el que se centra el intervalo. Hace falta declararlo
        # porque el intervalo se calcula sobre las predicciones de UNA corrida
        # mientras que el F1 de la fila es la media de las semillas. Se comprobo
        # que la diferencia importa: en `solo_estructura` las semillas van de
        # 0.7703 a 0.8371, y sin este campo el punto de la tabla caia fuera de su
        # propio intervalo sin que nada explicara por que.
        "valor_observado": round(_metrica_rapida(y_true, y_pred, clave), 4),
        "corresponde_a": "una sola corrida, no la media entre semillas",
        "ic95_inferior": round(float(np.percentile(v, 2.5)), 4),
        "ic95_superior": round(float(np.percentile(v, 97.5)), 4),
        "amplitud": round(float(np.percentile(v, 97.5) - np.percentile(v, 2.5)), 4),
    }


def equivalencia(y_true, pred_a, pred_b, delta: float = 0.005,
                 clave: str = "f1", n: int = 2000, semilla: int = 7) -> dict:
    """¿Son A y B equivalentes dentro de un margen `delta` declarado?

    Es la prueba que corresponde cuando la tarea satura. Preguntar «¿A supera a
    B?» da no significativo y se lee como que no se supo medir. Preguntar «¿es
    |A-B| < delta?» permite AFIRMAR equivalencia, que es una conclusión positiva
    y falsable. El margen se declara antes de mirar: aquí 0.005 de F1, que sobre
    ~2,000 filas de prueba son unos diez correos.

    Remuestreo pareado --las mismas filas para los dos modelos-- porque comparar
    remuestreos independientes inflaría la varianza de la diferencia.
    """
    y_true = np.asarray(y_true)
    pred_a, pred_b = np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(semilla)
    n_obs = len(y_true)
    difs = []
    for _ in range(n):
        i = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y_true[i])) < 2:
            continue
        difs.append(_metrica_rapida(y_true[i], pred_a[i], clave)
                    - _metrica_rapida(y_true[i], pred_b[i], clave))
    if not difs:
        return {"equivalentes": None}
    d = np.asarray(difs, dtype=float)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {
        "metrica": clave, "margen_delta": delta,
        "diferencia_media": round(float(d.mean()), 4),
        "ic95_de_la_diferencia": [round(lo, 4), round(hi, 4)],
        # Equivalencia por intervalo de confianza: si TODO el intervalo cabe
        # dentro de (-delta, +delta), se declara equivalencia al 95%.
        "equivalentes": bool(lo > -delta and hi < delta),
        "interpretacion": (
            f"equivalentes dentro de +/-{delta} con 95% de confianza"
            if lo > -delta and hi < delta else
            "no se puede declarar equivalencia: el intervalo excede el margen"),
    }


def agregar(por_semilla: list[dict], claves: tuple[str, ...] = (
        "f1", "accuracy", "precision", "recall", "roc_auc", "pr_auc",
        "mcc", "balanced_accuracy", "specificity", "brier", "ece", "fpr_a_tpr95")) -> dict:
    """Media y desviación entre semillas. La desviación es ENTRE SEMILLAS."""
    salida = {}
    for k in claves:
        vals = [d[k] for d in por_semilla if d.get(k) is not None]
        if vals:
            salida[k] = round(float(np.mean(vals)), 4)
            salida[f"{k}_desv"] = round(float(np.std(vals)), 4)
    return salida


def huella(valor) -> str:
    """Huella corta y estable de un arreglo o un marco, para comparar entre corridas."""
    import hashlib

    if isinstance(valor, pd.DataFrame):
        material = pd.util.hash_pandas_object(valor, index=False).to_numpy().tobytes()
    else:
        material = np.ascontiguousarray(np.asarray(valor)).tobytes()
    return hashlib.sha256(material).hexdigest()[:16]


def procedencia(corpus: pd.DataFrame | None = None,
                particion: "Particion | None" = None) -> dict:
    """Todo lo que hace falta para volver a producir el número, y nada más.

    Se adjunta a TODOS los informes desde `emitir`, no se pide a cada experimento
    que se acuerde. Un experimento que deba recordar documentarse acaba sin
    documentar, y en una tesis el revisor no puede distinguir «no se registró» de
    «se registró y coincide».

    Las dos huellas son el punto: la del corpus detecta que un informe describe
    una versión del corpus distinta de la de sus vecinos, y la de la partición
    demuestra que las líneas base clásicas y las neuronales se evaluaron sobre
    exactamente las mismas filas. Sin ellas, que las tablas sean comparables entre
    sí es una afirmación de confianza y no una comprobación.
    """
    import platform
    import subprocess

    def version(modulo: str) -> str:
        try:
            return __import__(modulo).__version__
        except Exception:
            return "ausente"

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        sucio = bool(subprocess.run(["git", "status", "--porcelain"], cwd=BASE,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
    except Exception:
        commit, sucio = "desconocido", None
    if not commit:
        # El servidor tiene copia de trabajo, no clon: sin esto el campo sale
        # vacio y no se distingue de un fallo al leerlo.
        commit = "sin repositorio git en esta maquina"

    reg: dict = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "git": {"commit": commit, "arbol_con_cambios_sin_confirmar": sucio},
        "semillas_de_modelo": list(SEMILLAS),
        "semilla_de_particion": SEMILLA_PARTICION,
        "entorno": {
            "python": platform.python_version(),
            "sistema": f"{platform.system()} {platform.release()}",
            "numpy": version("numpy"), "pandas": version("pandas"),
            "sklearn": version("sklearn"), "torch": version("torch"),
            "transformers": version("transformers"),
        },
    }
    try:
        import torch

        reg["entorno"]["gpu"] = (torch.cuda.get_device_name(0)
                                 if torch.cuda.is_available() else "sin GPU")
    except Exception:
        reg["entorno"]["gpu"] = "sin torch"

    if corpus is not None:
        reg["corpus"] = {
            "n": int(len(corpus)),
            "prevalencia": round(float(corpus["label"].mean()), 6),
            "huella": huella(corpus[["email_id", "label"]]
                             if "email_id" in corpus.columns else corpus[["label"]]),
        }
    if particion is not None:
        reg["particion"] = {
            **particion.resumen(),
            "huella": huella(np.concatenate([
                np.asarray(particion.entrenamiento), np.asarray(particion.validacion),
                np.asarray(particion.prueba)])),
        }
    return reg


def emitir(experimento: str, informe: dict,
           figura: Callable[[Path], None] | None = None,
           corpus: pd.DataFrame | None = None,
           particion: "Particion | None" = None) -> Path:
    """Escribe el JSON del experimento y, si se da, su figura, en el mismo paso.

    Adjunta la procedencia siempre: sin ella el JSON dice qué salió pero no sobre
    qué datos ni con qué versiones, que es la mitad que un tribunal necesita.
    """
    informe = {**informe, "procedencia": procedencia(corpus, particion)}
    destino = SALIDA / experimento
    destino.mkdir(parents=True, exist_ok=True)
    ruta = destino / f"{experimento}.json"
    ruta.write_text(json.dumps(informe, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    if figura is not None:
        figura(destino)
    print(f"  artefactos en {destino}")
    return ruta


def barras_con_error(destino: Path, nombre: str, titulo: str,
                     etiquetas: list[str], medias: list[float],
                     desviaciones: list[float], ylabel: str = "F1",
                     resaltar: int | None = None) -> None:
    """Gráfico de barras con dispersión entre semillas, eje acotado al rango útil.

    El eje no arranca en cero de forma deliberada: cuando todas las barras están
    entre 0.85 y 0.95, un eje de 0 a 1 las vuelve indistinguibles y oculta
    justamente lo que la lámina debe mostrar. Se anota el mínimo del eje para que
    la compresión quede declarada y no engañe.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_lineas = 1 + titulo.count(chr(10))
    fig, ax = plt.subplots(figsize=(max(6, 1.55 * len(etiquetas)),
                                    4.4 + 0.35 * (n_lineas - 1)))
    colores = ["#4C72B0"] * len(etiquetas)
    if resaltar is not None:
        colores[resaltar] = "#C44E52"
    ax.bar(etiquetas, medias, yerr=desviaciones, capsize=5, color=colores,
           edgecolor="#22303F", linewidth=0.6)
    bajo = max(0.0, min(m - d for m, d in zip(medias, desviaciones)) - 0.05)
    alto = min(1.0, max(m + d for m, d in zip(medias, desviaciones)) + 0.03)
    ax.set_ylim(bajo, alto)
    ax.set_ylabel(ylabel)
    # `pad` reserva sitio: con titulo de dos lineas la segunda caia sobre las
    # barras mas altas y tapaba su valor, que es justo el dato que se mira.
    ax.set_title(titulo, fontsize=11, pad=14)
    ax.grid(axis="y", alpha=0.3)
    for i, (m, d) in enumerate(zip(medias, desviaciones)):
        ax.text(i, m + d + (alto - bajo) * 0.02, f"{m:.4f}",
                ha="center", fontsize=9)
    ax.text(0.99, 0.02, f"eje recortado desde {bajo:.2f}", transform=ax.transAxes,
            ha="right", fontsize=8, style="italic", color="#6E7F8D")
    plt.xticks(rotation=12, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(destino / f"{nombre}.png", dpi=160)
    plt.close(fig)
