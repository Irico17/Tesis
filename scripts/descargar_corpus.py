"""
Descarga el corpus de la tesis desde la release `corpus-v1` del repositorio y
comprueba que sea exactamente el que usan los experimentos.

El corpus (589 MB) no se versiona en git: cada clon lo arrastraría para siempre.
Se publica como adjunto de una release, y este guion lo deja donde lo lee
`scripts/experimentos/comun.py`. La comprobación es doble: el SHA-256 del
fichero y la huella del corpus que citan todos los informes.

    python scripts/descargar_corpus.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DESTINO = BASE / "data" / "Datasets_Procesados" / "Dataset_Real.parquet"
URL = "https://github.com/Irico17/Tesis/releases/download/corpus-v1/Dataset_Real.parquet"
SHA256 = "1f92623e178ce79354a933f8f9d5bc75130052f5f793de698592f395df71b4e3"
HUELLA = "f078576f098f129d"


def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar() -> None:
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    parcial = DESTINO.with_suffix(".parquet.part")

    def avance(n, tam, total):
        if total > 0:
            print(f"\r  {min(n * tam, total) / 1e6:6.0f} de {total / 1e6:.0f} MB", end="")

    print(f"Descargando {URL}")
    urllib.request.urlretrieve(URL, parcial, reporthook=avance)
    print()
    parcial.replace(DESTINO)


def main() -> int:
    if DESTINO.exists() and sha256(DESTINO) == SHA256:
        print(f"El corpus ya está en {DESTINO.relative_to(BASE)} y su SHA-256 coincide.")
    else:
        descargar()
        if sha256(DESTINO) != SHA256:
            print("El SHA-256 no coincide: la descarga está incompleta o el fichero cambió.")
            return 1
        print("SHA-256 correcto.")

    sys.path.insert(0, str(BASE / "scripts" / "experimentos"))
    from comun import cargar_corpus, procedencia
    huella = procedencia(cargar_corpus())["corpus"]["huella"]
    if huella != HUELLA:
        print(f"La huella del corpus es {huella}, no {HUELLA}.")
        return 1
    print(f"Huella del corpus {huella}: es el corpus de los informes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
