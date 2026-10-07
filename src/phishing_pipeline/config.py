"""Rutas, constantes y configuraciÃ³n del pipeline."""

import logging
from pathlib import Path

import pandas as pd

import os

_kaggle_config_dir = Path.home() / ".kaggle"
if "KAGGLE_CONFIG_DIR" not in os.environ and _kaggle_config_dir.is_dir():
    os.environ["KAGGLE_CONFIG_DIR"] = str(_kaggle_config_dir)
# Directorio raÃ­z del proyecto (Tesis/)
BASE_DIR = Path(__file__).resolve().parents[2]

# Carpetas de datos
RAW_DIR = BASE_DIR / "data" / "Datasets_Originales"
PROCESSED_DIR = BASE_DIR / "data" / "Datasets_Procesados"
REPORTS_DIR = BASE_DIR / "data" / "reports"
FEATURES_DIR = BASE_DIR / "data" / "features"

# Subcarpetas por fuente
KAGGLE_PHISHING_DIR = RAW_DIR / "Kaggle_Phishing_Email"
SPAM_GENUINE_DIR = RAW_DIR / "Spam_Genuine_Mail"
PHISH_MMF_DIR = RAW_DIR / "PhishMMF"
PHISH_MMF_EXTRACTED = PHISH_MMF_DIR / "extracted"
PER_SOURCE_DIR = PROCESSED_DIR / "por_fuente"

# Archivos de salida
UNIFIED_CSV = PROCESSED_DIR / "Dataset_Unificado.csv"
UNIFIED_PARQUET = PROCESSED_DIR / "Dataset_Unificado.parquet"

# Reproducibilidad
RANDOM_STATE = 42

# Datasets Kaggle
KAGGLE_PHISHING_SLUG = "mohammadaoalhija/phishing-email"
KAGGLE_PHISHING_FILE = "Phishing_Email.csv"
SPAM_GENUINE_SLUG = "isuranga/spam-genuine-mail-contents-dataset"
SPAM_GENUINE_FILE = "email_dataset_100k.csv"

# PhishMMF
PHISH_MMF_GITHUB_URL = "https://github.com/12345677876/PhishMMF.git"
# Commit fijado para reproducibilidad: PhishMMF es un repositorio de
# terceros sin versionado formal; un `git clone --depth 1` sin referencia fija
# trae lo que sea que esté en HEAD el día de la ejecución, que puede diferir
# de lo usado para construir el corpus actual. Este es el commit real bajo el
# que se generó `Dataset_Unificado.parquet` (verificado contra el clon local
# ya presente en data/Datasets_Originales/PhishMMF/repo/, 2026-08-18) --
# fijarlo garantiza que una re-ejecución del pipeline en otra máquina/fecha
# reproduzca exactamente los mismos datos crudos.
PHISH_MMF_PINNED_COMMIT = "4887966166f1f19a72f41ac7ca26f98064ce3248"
PHISH_MMF_ZIP_FILES = [
    "all.zip",
    "phishing_pot.zip",
    "datacon2023_1.zip",
    "datacon2023_2.zip",
    "SpamAssasin_0.zip",
    "CEAS_08_0.zip",
]

# Indicadores R1.2
MIN_SUCCESS_RATE = 0.90
SMOKE_MULTIMODAL_MIN = 0.85
TOKENIZER_MODEL = "distilbert-base-uncased"
MAX_TOKEN_LENGTH = 512

# Splits estratificados
SPLITS_DIR = PROCESSED_DIR / "splits"
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Baselines R2.1
BASELINES_RESULTS_PATH = REPORTS_DIR / "baselines_results.json"

# Umbrales mÃ­nimos para detectar placeholders / descargas incompletas
MIN_KAGGLE_PHISHING_ROWS = 15_000
MIN_SPAM_GENUINE_ROWS = 90_000
MIN_PHISH_MMF_JSONL_LINES = 1_000

# Pandas kwargs para Kaggle phishing (celda 0 del notebook)
KAGGLE_PHISHING_PANDAS_KWARGS = {
    "sep": ",",
    "quotechar": '"',
    "escapechar": "\\",
    "on_bad_lines": "skip",
    "index_col": 0,
}

# Etiquetas canÃ³nicas
LABEL_SAFE = 0
LABEL_PHISHING = 1
LABEL_TEXT_SAFE = "Safe Email"
LABEL_TEXT_PHISHING = "Phishing Email"

# Nombres de fuente
SOURCE_KAGGLE = "Kaggle_Phishing_Email"
SOURCE_SPAM_GENUINE = "Spam_Genuine_Mail"
SOURCE_PHISH_MMF = "PhishMMF"

# Splits group-aware (Fase A1 — StratifiedGroupKFold por template_cluster_id)
SPLITS_DIR_GROUP_AWARE = PROCESSED_DIR / "splits_group_aware"


_logger = logging.getLogger(__name__)

# Fuentes del corpus de correo real. Cada una es su propia familia: se nombran por
# procedencia, no por fichero, que es la unidad sobre la que se componen los pliegues.
FUENTES_CORREO_REAL = frozenset(
    {"phishing_pot", "Nazario", "SpamAssassin", "CEAS_08", "datacon2023", "Kaggle"}
)


def get_source_family(source_dataset: str) -> str:
    """
    Familia de fuente canónica para agrupamiento (GroupKFold LOSO, dedup por fuente).

    Unifica las sub-fuentes de PhishMMF (p.ej. "PhishMMF_CEAS_08_0.jsonl") bajo
    "PhishMMF", y mapea Kaggle/Spam_Genuine a sus nombres completos de columna
    `source_dataset` (no abreviados) — estos son los mismos valores usados como
    `held_out_source` en data/reports/baselines_results.json. Antes existían dos
    definiciones divergentes de esta lógica (splits.py y phishing_baseline/group_cv.py);
    esta es la única fuente de verdad.

    Las fuentes del corpus de correo real (`corpus_real.py`) se devuelven tal cual:
    ya se nombran por su procedencia y no necesitan unificación.

    **Cuidado con el orden de las comprobaciones.** `SpamAssassin` empieza por
    «Spam», de modo que la condición `startswith("Spam")` lo capturaba y lo asignaba
    a la familia del corpus sintético `Spam_Genuine_Mail`. Como esta función decide
    la composición de los pliegues, el efecto habría sido silencioso y grave: una
    fuente legítima de correo real contabilizada como si fuese el generador de
    plantillas. Las coincidencias exactas se comprueban por tanto ANTES que los
    prefijos.
    """
    if not isinstance(source_dataset, str):
        return "Unknown"

    # Coincidencia exacta primero: las fuentes de correo real ya son su propia familia.
    if source_dataset in FUENTES_CORREO_REAL:
        return source_dataset

    if source_dataset.startswith("PhishMMF"):
        return SOURCE_PHISH_MMF
    if source_dataset.startswith("Kaggle"):
        return SOURCE_KAGGLE
    if source_dataset.startswith("Spam"):
        return SOURCE_SPAM_GENUINE
    return source_dataset


# Composición de pliegues para el corpus de correo real.
#
# **Por qué hace falta componerlos y no basta con retener una fuente.** Cinco de
# las seis fuentes de `Dataset_Real.parquet` son de CLASE ÚNICA: CEAS_08 y
# SpamAssassin solo aportan correo legítimo; Nazario, datacon2023 y phishing_pot
# solo phishing. Retener cualquiera de ellas produce un conjunto de prueba con una
# sola clase, sobre el que ni F1 ni el área bajo la curva ROC están definidos. El
# protocolo, sencillamente, no se puede calcular.
#
# No es un defecto de este corpus sino de cómo se publica el correo en el área:
# nadie difunde una colección que contenga ambas clases extraídas del mismo flujo,
# porque el correo legítimo real es dato privado. La única excepción del corpus es
# Kaggle, y por eso forma pliegue propio.
#
# El emparejamiento junta una recolección de phishing con una de correo legítimo
# obtenidas por procedimientos independientes, que es lo que da sentido a medir
# transferencia: retener el pliegue completo equivale a preguntar «¿generaliza a
# una pareja de recolecciones que nunca vio?».
#
# Se declara como constante y no se infiere automáticamente porque es una decisión
# de diseño experimental —qué se empareja con qué cambia lo que el experimento
# mide— y debe quedar explícita y revisable, no escondida en una heurística.
COMPOSICION_PLIEGUES: dict[str, tuple[str, ...]] = {
    "A_pot_vs_spamassassin": ("phishing_pot", "SpamAssassin"),
    "B_nazario_datacon_vs_ceas": ("Nazario", "datacon2023", "CEAS_08"),
    "C_kaggle": ("Kaggle",),
}


def asignar_pliegues(df: pd.DataFrame) -> pd.Series:
    """
    Asigna a cada fila el pliegue al que pertenece.

    Si todas las fuentes del corpus figuran en `COMPOSICION_PLIEGUES` se emplea esa
    composición; en cualquier otro caso se recurre a la familia de fuente, que es el
    comportamiento histórico y el que reproduce los resultados del corpus anterior.

    La condición es de cobertura TOTAL y no parcial a propósito: una composición que
    cubriera solo algunas fuentes dejaría al resto sin pliegue asignado, y mezclar
    ambos criterios produciría pliegues de naturaleza distinta dentro del mismo
    protocolo.
    """
    familias = df["source_dataset"].apply(get_source_family)
    presentes = set(familias.unique())
    declaradas = {f for fuentes in COMPOSICION_PLIEGUES.values() for f in fuentes}

    if not presentes.issubset(declaradas):
        return familias

    de_fuente_a_pliegue = {
        fuente: pliegue for pliegue, fuentes in COMPOSICION_PLIEGUES.items() for fuente in fuentes
    }
    _logger.info("Composición de pliegues activa: %s", sorted(COMPOSICION_PLIEGUES))
    return familias.map(de_fuente_a_pliegue)
