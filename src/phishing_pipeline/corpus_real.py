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

import numpy as np
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
    "Fedora",
    "kernel_lists",
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
    # Archivos de listas de discusión. Se incorporan porque son la única fuente
    # pública de correo LEGÍTIMO que conserva lo que al corpus le faltaba, y cada
    # una aporta una mitad distinta: Fedora el marcado del cuerpo (17.1% de
    # estructura frente al 1.2% del resto de legítimos) y las listas del kernel
    # las cabeceras de autenticación (69.2%, frente al 0.0% de todo lo demás).
    # Sin ellas, el subconjunto con las tres modalidades era 97.9% phishing y el
    # experimento de ablación por modalidad no podía plantearse.
    ("Fedora", "listas_html", 0),
    ("kernel_lists", "listas_correo", 0),
)


def fechas_de_obtencion(raw_dir: Path | None = None) -> dict:
    """Cuándo se descargó cada colección, leído de los ficheros crudos.

    El corpus se ensambla a partir de colecciones públicas que siguen creciendo:
    el repositorio de phishing_pot recibe muestras nuevas y los archivos de las
    listas de discusión se amplían cada mes. Una reconstrucción posterior no
    produce por fuerza el mismo corpus, de modo que la fecha en que se obtuvo
    cada colección forma parte de lo que hace interpretable el resultado y debe
    quedar registrada junto a las cifras.

    Se toma la fecha de modificación más reciente de los ficheros de cada
    colección, que es cuando la descarga terminó de escribirlos.
    """
    from datetime import datetime, timezone

    base = raw_dir or RAW_DIR
    fechas: dict[str, dict] = {}
    for nombre, subruta, _ in FUENTES_CRUDAS:
        carpeta = base / subruta
        if not carpeta.exists():
            continue
        marcas = [f.stat().st_mtime for f in carpeta.rglob("*") if f.is_file()]
        if not marcas:
            continue
        fechas[nombre] = {
            "obtenida_el": datetime.fromtimestamp(
                max(marcas), tz=timezone.utc).date().isoformat(),
            "ficheros": len(marcas),
        }
    # Las colecciones tabulares no viven en `Datasets_Originales` con la misma
    # estructura, de modo que se registran por su fichero de origen.
    for nombre, ruta in (("Kaggle", KAGGLE_PHISHING_DIR / KAGGLE_PHISHING_FILE),
                         ("CEAS_08", PHISH_MMF_EXTRACTED),
                         ("datacon2023", PHISH_MMF_EXTRACTED)):
        if ruta.exists():
            marcas = ([f.stat().st_mtime for f in ruta.rglob("*") if f.is_file()]
                      if ruta.is_dir() else [ruta.stat().st_mtime])
            if marcas:
                fechas[nombre] = {
                    "obtenida_el": datetime.fromtimestamp(
                        max(marcas), tz=timezone.utc).date().isoformat(),
                    "ficheros": len(marcas),
                }
    return fechas


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


def equilibrar(df: pd.DataFrame, prevalencia: float = 0.40,
               semilla: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    Submuestrea la clase legítima hasta la prevalencia objetivo.

    Hace falta porque los archivos de listas aportan 137,516 mensajes legítimos y
    volcarlos enteros dejaría el corpus en prevalencia 0.095: el desequilibrio que
    se venía corrigiendo, invertido. Se submuestrea la clase NEGATIVA porque el
    phishing es lo escaso y descartarlo perdería material irrepetible.

    La prevalencia de un corpus es un parámetro de diseño y no una estimación del
    mundo. La tasa real de phishing en un flujo de correo, tras el filtro de spam,
    está en el orden del 0.1 al 1%; ninguna cifra cercana al equilibrio la imita.
    Se equilibra porque el corpus existe para COMPARAR arquitecturas sobre los
    mismos datos, y un desbalance introduciría un segundo factor --la capacidad de
    explotar la probabilidad a priori-- que difiere entre arquitecturas y confunde
    la comparación. La validez externa se atiende aparte, con métricas
    independientes del umbral y una curva de sensibilidad a la tasa base.

    El reparto de la clase legítima respeta cuatro compromisos, en este orden:

    1. **Kaggle íntegra, sin tocar ninguna de sus dos clases.** Es la única
       colección que aporta las dos, y por tanto el único pliegue donde la clase
       no coincide con la procedencia. Recortar solo su mitad legítima le cambia
       la prevalencia interna --se midió pasar de 0.375 a 0.455-- y altera la
       única evaluación limpia del protocolo.
    2. **Tantos legítimos trimodales como phishing trimodales.** Es la razón de
       ser de la incorporación: deja equilibrado el subconjunto sobre el que se
       responde si la fusión aporta.
    3. **Cuota mínima para cada colección legítima restante.** Sin ella, el
       reparto proporcional dejaba SpamAssassin en 102 mensajes y kernel_lists en
       41, y esta última es la única fuente de autenticación legítima que existe:
       reducirla a 41 equivale a no haberla incorporado.
    4. **El resto, proporcional y con tope**, para que ninguna colección domine
       la clase negativa como Fedora haría por volumen.

    Si el presupuesto no alcanza para (1) y (2), se informa de la prevalencia
    máxima alcanzable en lugar de sacrificar en silencio una de las dos.
    """
    from phishing_pipeline.features.vectorizer import compute_modality_availability

    disp = compute_modality_availability(df)
    tri = (disp["has_structure_modality"] & disp["has_network_modality"]).to_numpy()
    es_phish = df["label"].to_numpy() == 1
    es_kaggle = df["source_dataset"].to_numpy() == "Kaggle"

    # Kaggle se conserva entera y queda fuera del presupuesto de submuestreo.
    intactas = df[es_kaggle]
    resto = df[~es_kaggle]
    tri_resto = tri[~es_kaggle]
    phish_resto = resto[resto["label"] == 1]
    leg_resto = resto[resto["label"] == 0]
    tri_leg = tri_resto[(resto["label"] == 0).to_numpy()]

    n_phish_total = int(len(phish_resto) + (intactas["label"] == 1).sum())
    presupuesto = int(round(n_phish_total * (1 - prevalencia) / prevalencia))
    presupuesto -= int((intactas["label"] == 0).sum())  # Kaggle ya gasta parte

    n_tri_phish = int((tri & es_phish).sum())
    rng = np.random.default_rng(semilla)
    avisos: list[str] = []

    if presupuesto < n_tri_phish:
        avisos.append(
            f"el presupuesto de legítimos ({presupuesto}) no cubre los "
            f"{n_tri_phish} trimodales que exige el equilibrio del subconjunto; "
            f"bajar la prevalencia objetivo por debajo de {prevalencia}"
        )

    elegidos: list[np.ndarray] = []
    def tomar(idx, n):
        n = int(min(max(n, 0), len(idx)))
        if n:
            elegidos.append(rng.choice(np.asarray(idx), size=n, replace=False))
        return n

    # (2) legítimos con las tres modalidades
    gastado = tomar(leg_resto.index[tri_leg], min(n_tri_phish, presupuesto))

    # (3) cuota mínima por colección, para que ninguna desaparezca
    ya = np.concatenate(elegidos) if elegidos else np.array([], dtype=leg_resto.index.dtype)
    pendientes = leg_resto.drop(index=ya, errors="ignore")
    fuentes = sorted(pendientes["source_dataset"].unique())
    if fuentes:
        minimo = max(0, (presupuesto - gastado)) // (2 * len(fuentes))
        for f in fuentes:
            idx = pendientes.index[pendientes["source_dataset"].to_numpy() == f]
            gastado += tomar(idx, min(minimo, presupuesto - gastado))

    # (4) el resto, proporcional y con tope del 40% de la clase negativa
    ya = np.concatenate(elegidos) if elegidos else np.array([], dtype=leg_resto.index.dtype)
    pendientes = leg_resto.drop(index=ya, errors="ignore")
    queda = max(0, presupuesto - gastado)
    if queda and len(pendientes):
        por_fuente = pendientes.groupby("source_dataset").size()
        tope = int(0.40 * presupuesto)
        for f, n in por_fuente.items():
            if queda <= 0:
                break
            idx = pendientes.index[pendientes["source_dataset"].to_numpy() == f]
            cupo = min(int(queda * n / por_fuente.sum()) or 1, tope, n, queda)
            queda -= tomar(idx, cupo)

    seleccion = np.concatenate(elegidos) if elegidos else np.array([], dtype=leg_resto.index.dtype)
    salida = pd.concat([intactas, phish_resto, leg_resto.loc[seleccion]]).sort_index()
    salida = salida.reset_index(drop=True)

    d2 = compute_modality_availability(salida)
    tri2 = d2["has_structure_modality"] & d2["has_network_modality"]
    informe = {
        "prevalencia_objetivo": prevalencia,
        "prevalencia_obtenida": round(float(salida["label"].mean()), 4),
        "n": int(len(salida)),
        "legitimos_descartados": int(len(leg_resto) - len(seleccion)),
        "trimodal": {
            "n": int(tri2.sum()),
            "prevalencia": round(float(salida.loc[tri2, "label"].mean()), 4),
        },
        "por_fuente": {k: int(v) for k, v in salida.groupby("source_dataset").size().items()},
        "avisos": avisos,
    }
    return salida, informe


def construir(raw_dir: Path | None = None, descargar: bool = True,
              prevalencia: float = 0.40, semilla: int = 42) -> tuple[pd.DataFrame, dict]:
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

    # Equilibrado de la clase legítima. Va DESPUÉS de deduplicar, para que el
    # presupuesto se reparta sobre filas ya únicas, y ANTES del informe, que debe
    # describir el corpus que de verdad se escribe.
    df, informe_equilibrio = equilibrar(df, prevalencia=prevalencia, semilla=semilla)
    logger.info(
        "Equilibrado a prevalencia %.4f (objetivo %.2f); %d legítimos descartados",
        informe_equilibrio["prevalencia_obtenida"], prevalencia,
        informe_equilibrio["legitimos_descartados"],
    )

    # Se agrupa DESPUÉS de equilibrar, no antes. Antes obligaba a recorrer las
    # 229,385 filas leídas para conservar 44,100 --cinco veces el trabajo-- y
    # producía identificadores que remiten a filas que el submuestreo descarta.
    # El conglomerado debe describir el corpus que se escribe.
    # Agrupación de casi-duplicados por MinHash. Va antes del equilibrado para
    # que el submuestreo reparta filas que ya tienen conglomerado asignado.
    #
    # No estaba: `find_near_duplicate_clusters` existía y solo se aplicaba en el
    # pipeline del corpus anterior, de modo que las fuentes que pasan por el
    # limpiador crudo llegaban aquí con `template_cluster_id` a nulo. Se midió el
    # alcance: 24,853 filas de 44,100 --el 56% del corpus-- sin conglomerado,
    # incluidas phishing_pot, Nazario y Fedora enteras.
    #
    # La consecuencia no era cosmética. La partición agrupada sirve para impedir
    # que una misma campaña caiga en entrenamiento y en prueba; sin identificador,
    # esa protección solo alcanzaba al 44% de los correos y el resto podía
    # repartirse a ambos lados, inflando la métrica por memorización de plantilla.
    from phishing_pipeline.dedup.near_duplicates import find_near_duplicate_clusters

    faltan = df["template_cluster_id"].isna().sum() if "template_cluster_id" in df else len(df)
    if faltan:
        logger.info("Agrupando casi-duplicados por MinHash (%d filas sin conglomerado)", faltan)
        df["template_cluster_id"] = find_near_duplicate_clusters(df)
        logger.info(
            "  %d conglomerados para %d correos; sin asignar: %d",
            df["template_cluster_id"].nunique(), len(df),
            int(df["template_cluster_id"].isna().sum()),
        )

    from phishing_pipeline.features.vectorizer import compute_modality_availability

    disponibilidad = compute_modality_availability(df)
    tri = disponibilidad["has_structure_modality"] & disponibilidad["has_network_modality"]
    bi = disponibilidad["has_structure_modality"] | disponibilidad["has_network_modality"]

    informe = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "fechas_de_obtencion": fechas_de_obtencion(),
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
    informe["equilibrado"] = informe_equilibrio
    return df, informe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--salida", type=Path, default=SALIDA)
    # La prevalencia es un parámetro de diseño y se expone como tal: la
    # justificación de 0.40 está en doc/DECISIONES_CORPUS_Y_PROTOCOLO.md.
    parser.add_argument("--prevalencia", type=float, default=0.40)
    parser.add_argument("--semilla", type=int, default=42)
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

    df, informe = construir(args.raw_dir, descargar=not args.sin_descargar,
                             prevalencia=args.prevalencia, semilla=args.semilla)

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
