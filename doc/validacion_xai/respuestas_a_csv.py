"""
Convierte la exportación de Google Forms al CSV que espera `script_scoring.py`.

Google Forms exporta una fila por respuesta y una columna por pregunta, con el
título de la pregunta como cabecera; en las rejillas, la cabecera es el título
seguido de la fila entre corchetes. Este guion localiza cada ítem por su código
(A1 a A10, B1 a B6, C1 a C10, P1 a P3), califica la sección C con
`formulario/clave_de_respuestas.json` y asigna a cada respuesta un identificador
anónimo (EVAL_01, EVAL_02...). Las respuestas de quien no aceptó participar se
descartan.

    python doc/validacion_xai/respuestas_a_csv.py --entrada respuestas.csv --salida panel.csv
    python doc/validacion_xai/script_scoring.py --input panel.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
CLAVE = AQUI / "formulario" / "clave_de_respuestas.json"


def _columna(cabeceras: list[str], patron: str) -> str:
    halladas = [c for c in cabeceras if re.search(patron, c)]
    if len(halladas) != 1:
        raise SystemExit(f"Se esperaba una columna para «{patron}» y hay {len(halladas)}.")
    return halladas[0]


def _escala(valor) -> int:
    m = re.match(r"\s*([1-5])", str(valor))
    if not m:
        raise SystemExit(f"Respuesta fuera de la escala 1 a 5: {valor!r}")
    return int(m.group(1))


def convertir(df: pd.DataFrame, clave: dict) -> pd.DataFrame:
    cab = list(df.columns)
    consentimiento = _columna(cab, r"^Consentimiento")
    df = df[df[consentimiento].astype(str).str.startswith("Acepto")].reset_index(drop=True)

    salida = pd.DataFrame({"evaluador_id": [f"EVAL_{i:02d}" for i in range(1, len(df) + 1)]})
    for i in range(1, 11):
        salida[f"sus_q{i}"] = df[_columna(cab, rf"\[A{i}\. ")].map(_escala)
    # B1 a B6 son los ítems 7 a 12 del PSSUQ (subescala de Calidad de la
    # Información); se guardan con su numeración original.
    for i in range(1, 7):
        salida[f"pssuq_q{i + 6}"] = df[_columna(cab, rf"\[B{i}\. ")].map(_escala)
    for i in range(1, len(clave["preguntas"]) + 1):
        pregunta = clave["preguntas"][f"C{i}"]
        respuesta = df[_columna(cab, rf"^C{i}\. ")].astype(str).str.strip()
        salida[f"comprension_q{i}"] = respuesta
        salida[f"comprension_correcta_q{i}"] = (respuesta == pregunta["correcta"]).astype(int)
    for codigo in ("P1", "P2", "P3"):
        salida[f"perfil_{codigo.lower()}"] = df[_columna(cab, rf"^{codigo}\. ")]
    return salida


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entrada", required=True, help="CSV exportado de Google Forms")
    p.add_argument("--salida", required=True, help="CSV para script_scoring.py")
    args = p.parse_args()
    clave = json.loads(CLAVE.read_text(encoding="utf-8"))
    panel = convertir(pd.read_csv(args.entrada), clave)
    panel.to_csv(args.salida, index=False)
    print(f"{len(panel)} evaluadores escritos en {args.salida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
