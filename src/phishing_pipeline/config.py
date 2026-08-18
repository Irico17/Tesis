"""Rutas, constantes y configuraciÃ³n del pipeline."""

from pathlib import Path

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
# Commit fijado para reproducibilidad (mejora de Fase I, ver
# doc/INFORME_R1.3_R1.4_R2.2_R3.md §11): PhishMMF es un repositorio de
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


def get_source_family(source_dataset: str) -> str:
    """
    Familia de fuente canónica para agrupamiento (GroupKFold LOSO, dedup por fuente).

    Unifica las sub-fuentes de PhishMMF (p.ej. "PhishMMF_CEAS_08_0.jsonl") bajo
    "PhishMMF", y mapea Kaggle/Spam_Genuine a sus nombres completos de columna
    `source_dataset` (no abreviados) — estos son los mismos valores usados como
    `held_out_source` en data/reports/baselines_results.json. Antes existían dos
    definiciones divergentes de esta lógica (splits.py y phishing_baseline/group_cv.py);
    esta es la única fuente de verdad.
    """
    if not isinstance(source_dataset, str):
        return "Unknown"
    if source_dataset.startswith("PhishMMF"):
        return SOURCE_PHISH_MMF
    if source_dataset.startswith("Kaggle"):
        return SOURCE_KAGGLE
    if source_dataset.startswith("Spam"):
        return SOURCE_SPAM_GENUINE
    return source_dataset
