"""
Construcción reproducible del corpus de correo real.

Sustituye a `Spam_Genuine_Mail` —texto generado por plantilla, cuyo 100% de
plantillas pertenecía a una sola clase— por correo real en formato crudo. El
resultado se escribe en `Dataset_Real.parquet` y **no toca**
`Dataset_Unificado.parquet`, que se conserva para poder reproducir los resultados
anteriores.

Se implementa aparte de `pipeline.py` y no dentro de él por una razón concreta: ese
orquestador tiene las tres fuentes originales cableadas en su cuerpo, y añadir allí
las nuevas obligaría a reestructurarlo con riesgo de alterar el camino que produce
el corpus vigente. Este módulo replica su contrato —descargar, limpiar, unificar,
deduplicar, informar— sobre el conjunto nuevo de fuentes.

Composición:

| Fuente | Origen | Clase |
|---|---|---|
| phishing_pot | `.eml`, honeypot, recolección continua | phishing |
| Nazario | `mbox`, 2005-2025 | phishing |
| datacon2023 | JSONL de PhishMMF | phishing |
| SpamAssassin | `.eml`, 2002-2003 | legítimo |
| CEAS_08 | JSONL de PhishMMF | legítimo |
| Kaggle | CSV | ambas |

Las versiones de phishing_pot y SpamAssassin que llegaban por el JSONL de PhishMMF
quedan excluidas y se sustituyen por su formato original: se comprobó que esa
conversión descartaba el cuerpo HTML y con él la modalidad estructural completa.

Uso:

    python -m phishing_pipeline.corpus_real
    python -m phishing_pipeline.corpus_real --sin-descargar
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from phishing_pipeline.cleaners.correo_crudo import limpiar_correo_crudo
from phishing_pipeline.config import (
    KAGGLE_PHISHING_DIR,
    KAGGLE_PHISHING_FILE,
    PHISH_MMF_EXTRACTED,
    PROCESSED_DIR,
    RAW_DIR,
    REPORTS_DIR,
)
from phishing_pipeline.features.dom_parser import ETIQUETA_HTML
from phishing_pipeline.features.dom_stats import compute_dom_stats
from phishing_pipeline.features.network import URL_PATTERN, extract_lexical_url_features
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

SALIDA = PROCESSED_DIR / "Dataset_Real.parquet"

# Las seis fuentes que debe contener el corpus completo. Si falta alguna, el
# resultado describe otro corpus y no debe sobrescribir al vigente sin decirlo.
FUENTES_ESPERADAS = (
    "phishing_pot",
    "Nazario",
    "SpamAssassin",
    "CEAS_08",
    "datacon2023",
    "Kaggle",
)
INFORME = REPORTS_DIR / "corpus_real_report.json"

# Sub-fuentes del JSONL de PhishMMF que SÍ se conservan. Las otras dos
# —phishing_pot y SpamAssassin— se reemplazan por su formato crudo.
MMF_CONSERVADAS = {
    "PhishMMF_CEAS_08_0.jsonl": "CEAS_08",
    "PhishMMF_datacon2023_1.jsonl": "datacon2023",
    "PhishMMF_datacon2023_2.jsonl": "datacon2023",
}

FUENTES_CRUDAS = (
    ("phishing_pot", "phishing_pot/email", 1),
    ("Nazario", "Nazario", 1),
    ("SpamAssassin", "SpamAssassin", 0),
)


def _huella(texto: str) -> str:
    return hashlib.blake2b(
        " ".join(str(texto).split()).lower().encode("utf-8"), digest_size=16
    ).hexdigest()


def recalcular_caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcula las características derivadas del cuerpo sobre TODAS las filas.

    Es necesario aunque los limpiadores ya las produzcan, porque las filas que
    proceden del corpus anterior traen valores calculados con los extractores en su
    versión defectuosa. Recalcular aquí garantiza que todo el corpus se haya medido
    con el mismo código, que es la propiedad que hace comparables las fuentes entre
    sí: la revisión del pipeline encontró que las asimetrías de preprocesamiento
    entre fuentes se convierten en huellas del origen.
    """
    cuerpos = df["body_raw"].fillna("").astype(str)

    df = df.copy()
    df["has_html"] = cuerpos.map(lambda b: 1 if ETIQUETA_HTML.search(b) else 0)

    estructura = pd.DataFrame([compute_dom_stats(b) for b in cuerpos], index=df.index)
    df["total_nodos_dom"] = estructura["total_nodos_dom"]
    df["profundidad_dom"] = estructura["profundidad_dom"]

    lexicas = pd.DataFrame([extract_lexical_url_features(b) for b in cuerpos], index=df.index)
    for columna in lexicas.columns:
        df[columna] = lexicas[columna]

    df["num_urls_metadata"] = cuerpos.map(lambda b: len(URL_PATTERN.findall(b)))
    return df


def _cargar_crudas(raw_dir: Path) -> list[pd.DataFrame]:
    partes = []
    for nombre, subruta, etiqueta in FUENTES_CRUDAS:
        ruta = raw_dir / subruta
        if not ruta.exists():
            logger.warning("No existe %s; se omite la fuente %s", ruta, nombre)
            continue
        partes.append(limpiar_correo_crudo(ruta, nombre, etiqueta))
    return partes


def _heredadas_desde_unificado() -> list[pd.DataFrame]:
    """
    Recupera Kaggle, CEAS_08 y datacon2023 del corpus ya consolidado.

    Es la vía normal, no un caso de reserva. `data/Datasets_Originales/` no está
    versionado —son gigabytes de datos crudos regenerables— mientras que
    `Dataset_Unificado.parquet` sí contiene esas tres fuentes ya limpiadas. Sin
    esta ruta, construir el corpus en una máquina recién clonada produce en
    silencio un corpus **incompleto**: se observó una ejecución que devolvió 14,234
    filas con prevalencia 0.711 en lugar de 35,723 con prevalencia 0.496, sin más
    aviso que dos advertencias en el registro.

    El riesgo de esa degradación silenciosa es alto, porque el resultado parece
    válido: un parquet bien formado, con las columnas correctas, que simplemente
    describe otro corpus.
    """
    ruta = PROCESSED_DIR / "Dataset_Unificado.parquet"
    if not ruta.exists():
        return []

    unificado = pd.read_parquet(ruta)
    mmf = unificado[unificado["source_dataset"].isin(MMF_CONSERVADAS)].copy()
    mmf["source_dataset"] = mmf["source_dataset"].map(MMF_CONSERVADAS)

    kaggle = unificado[unificado["source_dataset"].str.startswith("Kaggle")].copy()
    kaggle["source_dataset"] = "Kaggle"

    partes = [p for p in (kaggle, mmf) if len(p)]
    for p in partes:
        logger.info(
            "Desde Dataset_Unificado: %s (%d filas)",
            sorted(p["source_dataset"].unique()),
            len(p),
        )
    return partes


def _cargar_heredadas() -> list[pd.DataFrame]:
    """Kaggle y las dos sub-fuentes de PhishMMF que se conservan."""
    partes = []

    ruta_kaggle = KAGGLE_PHISHING_DIR / KAGGLE_PHISHING_FILE
    if ruta_kaggle.exists():
        from phishing_pipeline.cleaners.kaggle_phishing import clean_kaggle_phishing
        from phishing_pipeline.config import KAGGLE_PHISHING_PANDAS_KWARGS

        kwargs = {k: v for k, v in KAGGLE_PHISHING_PANDAS_KWARGS.items() if k != "index_col"}
        limpio = clean_kaggle_phishing(pd.read_csv(ruta_kaggle, **kwargs))
        limpio["source_dataset"] = "Kaggle"
        partes.append(limpio)
        logger.info("Kaggle: %d filas", len(limpio))
    else:
        logger.warning("No existe %s; se omite Kaggle", ruta_kaggle)

    if PHISH_MMF_EXTRACTED.exists():
        from phishing_pipeline.cleaners.phish_mmf import clean_phish_mmf

        mmf = clean_phish_mmf(PHISH_MMF_EXTRACTED)
        mmf = mmf[mmf["source_dataset"].isin(MMF_CONSERVADAS)].copy()
        mmf["source_dataset"] = mmf["source_dataset"].map(MMF_CONSERVADAS)
        if len(mmf):
            partes.append(mmf)
            logger.info("PhishMMF (solo CEAS_08 y datacon2023): %d filas", len(mmf))
    else:
        logger.warning("No existe %s; se omiten CEAS_08 y datacon2023", PHISH_MMF_EXTRACTED)

    presentes = {f for p in partes for f in p["source_dataset"].unique()}
    esperadas = {"Kaggle", "CEAS_08", "datacon2023"}
    if not esperadas.issubset(presentes):
        logger.info(
            "Faltan fuentes heredadas %s en datos crudos; se recuperan del corpus consolidado",
            sorted(esperadas - presentes),
        )
        for extra in _heredadas_desde_unificado():
            nuevas = set(extra["source_dataset"].unique()) - presentes
            if nuevas:
                partes.append(extra[extra["source_dataset"].isin(nuevas)])
                presentes |= nuevas

    return partes


def construir(raw_dir: Path | None = None, descargar: bool = True) -> tuple[pd.DataFrame, dict]:
    """Descarga si procede, limpia, unifica, deduplica y devuelve corpus e informe."""
    raw_dir = raw_dir or RAW_DIR

    if descargar:
        from phishing_pipeline.downloaders.correo_real import descargar_todo

        descargar_todo(raw_dir)

    partes = _cargar_crudas(raw_dir) + _cargar_heredadas()
    if not partes:
        raise RuntimeError(
            "Ninguna fuente disponible. Ejecute primero "
            "`python -m phishing_pipeline.downloaders.correo_real`."
        )

    columnas = partes[0].columns
    df = pd.concat([p.reindex(columns=columnas) for p in partes], ignore_index=True)
    df = recalcular_caracteristicas(df)

    # Deduplicación exacta GLOBAL, no por fuente. Las colecciones de phishing se
    # solapan entre sí —Nazario y phishing_pot comparten mensajes—, y un correo
    # presente en dos fuentes caería a ambos lados de la partición bajo el
    # protocolo por fuente no observada.
    antes = len(df)
    df["_huella"] = df["clean_text"].map(_huella)
    df = df.drop_duplicates("_huella", keep="first").drop(columns=["_huella"])
    df = df.reset_index(drop=True)
    duplicados = antes - len(df)

    from phishing_pipeline.features.vectorizer import compute_modality_availability

    disponibilidad = compute_modality_availability(df)
    tri = disponibilidad["has_structure_modality"] & disponibilidad["has_network_modality"]
    bi = disponibilidad["has_structure_modality"] | disponibilidad["has_network_modality"]

    informe = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "n_filas": int(len(df)),
        "duplicados_globales_eliminados": int(duplicados),
        "prevalencia": round(float(df["label"].mean()), 6),
        "por_fuente": {
            str(fuente): {
                "n": int(len(g)),
                "prevalencia": round(float(g["label"].mean()), 6),
                "clases": int(g["label"].nunique()),
            }
            for fuente, g in df.groupby("source_dataset")
        },
        "cobertura": {
            "trimodal": {
                "n": int(tri.sum()),
                "porcentaje": round(100 * float(tri.mean()), 4),
                "prevalencia": round(float(df.loc[tri, "label"].mean()), 6),
            },
            "bimodal": {
                "n": int(bi.sum()),
                "porcentaje": round(100 * float(bi.mean()), 4),
                "prevalencia": round(float(df.loc[bi, "label"].mean()), 6),
            },
        },
    }
    return df, informe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    parser.add_argument(
        "--permitir-incompleto",
        action="store_true",
        help="Escribe el corpus aunque falte alguna fuente. Sin esta bandera se aborta.",
    )
    parser.add_argument(
        "--sin-descargar",
        action="store_true",
        help="Usa lo que ya haya en disco, sin contactar con las fuentes.",
    )
    args = parser.parse_args()

    df, informe = construir(args.raw_dir, descargar=not args.sin_descargar)

    # Guardarraíl contra la degradación silenciosa. Un corpus al que le falte una
    # fuente sigue siendo un parquet bien formado y no se distingue del completo
    # salvo mirando las cifras, de modo que sobrescribir el artefacto vigente con
    # él destruye trabajo sin avisar. Escribir exige entonces o bien las seis
    # fuentes, o bien decirlo explícitamente.
    faltan = set(FUENTES_ESPERADAS) - set(informe["por_fuente"])
    if faltan and not args.permitir_incompleto:
        lineas = [
            f"ABORTADO: faltan {len(faltan)} fuentes: {sorted(faltan)}.",
            f"  Se obtuvieron {informe['n_filas']} filas con prevalencia "
            f"{informe['prevalencia']:.3f}, que NO es el corpus completo.",
            "  Ejecute `python -m phishing_pipeline.downloaders.correo_real` para "
            "obtener las fuentes crudas,",
            "  o repita con --permitir-incompleto si la omisión es intencionada.",
        ]
        print("\n".join(lineas), file=sys.stderr)
        return 2

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.salida, index=False)
    INFORME.parent.mkdir(parents=True, exist_ok=True)
    INFORME.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")

    c = informe["cobertura"]
    print(f"Corpus real: {informe['n_filas']} filas, prevalencia {informe['prevalencia']:.3f}")
    print(f"  duplicados globales eliminados: {informe['duplicados_globales_eliminados']}")
    print(f"  tri-modal: {c['trimodal']['n']:6d} ({c['trimodal']['porcentaje']:.1f}%) "
          f"prevalencia {c['trimodal']['prevalencia']:.3f}")
    print(f"  bi-modal : {c['bimodal']['n']:6d} ({c['bimodal']['porcentaje']:.1f}%) "
          f"prevalencia {c['bimodal']['prevalencia']:.3f}")
    print(f"\nEscrito en {args.salida}\nInforme en {INFORME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
