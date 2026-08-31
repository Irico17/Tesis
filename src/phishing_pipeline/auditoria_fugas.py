"""
Auditoría de fugas de etiqueta por disponibilidad de campo.

Un campo que solo se extrae para una parte del corpus deja de ser una
característica del correo y pasa a ser una huella del pipeline de ingesta. Si esa
parte coincide con una clase, la simple PRESENCIA del campo revela la etiqueta,
con independencia de su valor.

Ocurrió en este trabajo y no se detectó durante meses. El limpiador de PhishMMF
solo extraía las cabeceras de autenticación de tres de sus cinco ficheros, y esos
tres eran exactamente los tres de phishing, de modo que dentro de esa fuente
`p(phishing | spf presente)` valía 1.0000 sobre 4,478 correos. El efecto se
propagaba además a la máscara de disponibilidad de red que el modelo recibe como
entrada explícita.

Este módulo convierte esa corrección en un invariante comprobable: mide, para
cada campo y DENTRO de cada fuente, cuánta información aporta su mera presencia
sobre la etiqueta, y falla cuando supera un umbral. Corregir la fuga una vez
resuelve el caso; comprobarla en cada ejecución impide que vuelva.

Se ejecuta como parte de `prueba_integral.py` y puede invocarse por separado:

    python -m phishing_pipeline.auditoria_fugas
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phishing_pipeline.config import PROCESSED_DIR, REPORTS_DIR, get_source_family
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

INFORME_PATH = REPORTS_DIR / "auditoria_fugas.json"

# Campos cuya ausencia es informativa y que por tanto deben auditarse. Son los que
# el pipeline rellena de forma condicional, no los que existen siempre.
CAMPOS_AUDITADOS = (
    "spf_result",
    "dkim_result",
    "dmarc_result",
    "received_origin_ip",
    "num_urls_metadata",
    "language",
    "subject",
    "sender",
    "body_html",
)

# Umbral de información mutua, en nats, entre la PRESENCIA de un campo y la
# etiqueta, medida dentro de una misma fuente. La entropía de una etiqueta binaria
# equilibrada es de 0.693 nats, de modo que 0.05 corresponde a algo más del 7% de
# la información total de la etiqueta: suficientemente bajo para atrapar una fuga
# real y suficientemente alto para no disparar por una asociación menor y legítima.
UMBRAL_MI_NATS = 0.05


def _mi_binaria(presente: np.ndarray, etiqueta: np.ndarray) -> float:
    """
    Información mutua en nats entre dos variables binarias.

    Se calcula de forma directa sobre la tabla de contingencia 2x2 en lugar de
    delegar en scikit-learn, porque aquí importa que el resultado sea exactamente
    reproducible y no dependa de la estimación por vecinos más próximos que esa
    biblioteca emplea para variables continuas.
    """
    n = len(etiqueta)
    if n == 0:
        return 0.0
    total = 0.0
    for a in (0, 1):
        for b in (0, 1):
            n_ab = float(np.sum((presente == a) & (etiqueta == b)))
            if n_ab == 0:
                continue
            p_ab = n_ab / n
            p_a = float(np.sum(presente == a)) / n
            p_b = float(np.sum(etiqueta == b)) / n
            total += p_ab * np.log(p_ab / (p_a * p_b))
    return float(max(total, 0.0))


def _esta_presente(serie: pd.Series) -> np.ndarray:
    """
    Presencia de un valor, tratando la cadena vacía como ausencia.

    Varias columnas del esquema canónico usan `""` como valor por defecto en lugar
    de nulo (ver `schema.empty_canonical_row`), de modo que comprobar solo el nulo
    dejaría pasar precisamente los campos que el limpiador no rellenó.
    """
    if serie.dtype == object:
        return (serie.notna() & (serie.astype(str).str.strip() != "")).to_numpy()
    return serie.notna().to_numpy()


def auditar(df: pd.DataFrame, umbral: float = UMBRAL_MI_NATS) -> dict[str, Any]:
    """
    Audita la disponibilidad de cada campo dentro de cada fuente.

    Devuelve un informe con una entrada por par (fuente, campo) y el veredicto
    global. Las fuentes en que un campo está siempre presente o siempre ausente no
    aportan información y quedan registradas con información mutua nula.
    """
    trabajo = df.copy()
    trabajo["_fuente"] = trabajo["source_dataset"].map(get_source_family)

    hallazgos: list[dict[str, Any]] = []
    for fuente, grupo in trabajo.groupby("_fuente"):
        etiqueta = grupo["label"].astype(int).to_numpy()
        if len(np.unique(etiqueta)) < 2:
            # Una fuente de clase única no puede exhibir esta fuga: no hay etiqueta
            # que revelar. Se registra para que el informe sea completo.
            hallazgos.append(
                {
                    "fuente": str(fuente),
                    "campo": None,
                    "nota": "fuente de clase única; la auditoría no aplica",
                }
            )
            continue

        for campo in CAMPOS_AUDITADOS:
            if campo not in grupo.columns:
                continue
            presente = _esta_presente(grupo[campo]).astype(int)
            cobertura = float(presente.mean())
            mi = _mi_binaria(presente, etiqueta)
            registro: dict[str, Any] = {
                "fuente": str(fuente),
                "campo": campo,
                "cobertura": round(cobertura, 6),
                "mi_presencia_etiqueta_nats": round(mi, 6),
                "supera_umbral": bool(mi > umbral),
            }
            if 0.0 < cobertura < 1.0:
                registro["p_positivo_si_presente"] = round(
                    float(etiqueta[presente == 1].mean()), 6
                )
                registro["p_positivo_si_ausente"] = round(
                    float(etiqueta[presente == 0].mean()), 6
                )
            hallazgos.append(registro)

    fugas = [h for h in hallazgos if h.get("supera_umbral")]
    informe = {
        "umbral_mi_nats": umbral,
        "n_filas": int(len(trabajo)),
        "n_fuentes": int(trabajo["_fuente"].nunique()),
        "campos_auditados": list(CAMPOS_AUDITADOS),
        "hallazgos": hallazgos,
        "fugas_detectadas": fugas,
        "veredicto": "SIN_FUGAS" if not fugas else "FUGA_DETECTADA",
    }
    return informe


def formatear(informe: dict[str, Any]) -> str:
    """Resumen legible del informe, para consola y para el registro de la corrida."""
    lineas = [
        f"Auditoría de fugas por disponibilidad — {informe['n_filas']} filas, "
        f"{informe['n_fuentes']} fuentes, umbral {informe['umbral_mi_nats']} nats",
        "",
    ]
    if informe["veredicto"] == "SIN_FUGAS":
        lineas.append("  Sin fugas: ninguna presencia de campo informa sobre la etiqueta")
        lineas.append("  por encima del umbral dentro de su propia fuente.")
        return "\n".join(lineas)

    lineas.append(f"  {len(informe['fugas_detectadas'])} FUGA(S) DETECTADA(S):")
    lineas.append("")
    for f in informe["fugas_detectadas"]:
        lineas.append(
            f"    {f['fuente']} · {f['campo']}: {f['mi_presencia_etiqueta_nats']:.4f} nats, "
            f"cobertura {100 * f['cobertura']:.1f}%"
        )
        if "p_positivo_si_presente" in f:
            lineas.append(
                f"        p(positivo | presente) = {f['p_positivo_si_presente']:.4f}   "
                f"p(positivo | ausente) = {f['p_positivo_si_ausente']:.4f}"
            )
    lineas.append("")
    lineas.append("  La presencia de esos campos identifica la clase dentro de su fuente.")
    lineas.append("  Corregir en el limpiador: o se extrae el campo de todas las sub-fuentes,")
    lineas.append("  o se anula en todas. Conservar la asimetría invalida el resultado.")
    return "\n".join(lineas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=PROCESSED_DIR / "Dataset_Unificado.parquet",
        help="Corpus consolidado a auditar.",
    )
    parser.add_argument("--umbral", type=float, default=UMBRAL_MI_NATS)
    parser.add_argument(
        "--salida", type=Path, default=INFORME_PATH, help="Ruta del informe JSON."
    )
    parser.add_argument(
        "--estricto",
        action="store_true",
        help="Devuelve código de salida 1 si se detecta alguna fuga (uso en integración continua).",
    )
    args = parser.parse_args()

    if not args.parquet.exists():
        logger.error("No existe el corpus: %s", args.parquet)
        return 2

    df = pd.read_parquet(args.parquet)
    informe = auditar(df, umbral=args.umbral)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(
        json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(formatear(informe))
    print(f"\nInforme escrito en {args.salida}")

    if args.estricto and informe["veredicto"] != "SIN_FUGAS":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
