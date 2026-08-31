"""
Descarga de los corpus de correo real en formato crudo.

Tres fuentes, tres protocolos de publicación distintos:

- **phishing_pot**: repositorio de GitHub con un `.eml` por muestra, recolectado
  por honeypot y actualizado de forma continua. Se clona con profundidad 1: no
  hace falta el historial y evita descargar decenas de megabytes de versiones
  anteriores.
- **Nazario**: directorio HTTP con ficheros `mbox`, uno por año más varios
  archivos históricos. Es el corpus de referencia del área y sigue publicándose.
- **SpamAssassin**: archivos `.tar.bz2` del proyecto Apache. Solo se descargan los
  de *ham* (correo legítimo); los de spam quedan fuera por decisión metodológica,
  ya que spam y phishing son fenómenos distintos y mezclarlos fue precisamente el
  problema de validez de constructo que este corpus viene a corregir.

**Por qué crudo y no preprocesado.** La versión de estas mismas fuentes que
llegaba a través del JSONL de PhishMMF había perdido el cuerpo HTML: SpamAssassin
figuraba con un 2.9% de cobertura estructural cuando su valor real, medido sobre
los `.eml` originales, es del 5.4% —y con el patrón de detección defectuoso llegó
a informarse un 33.8%—. Una conversión intermedia que descarta partes MIME destruye
justamente la modalidad que este trabajo pretende estudiar.

Uso:

    python -m phishing_pipeline.downloaders.correo_real
    python -m phishing_pipeline.downloaders.correo_real --solo nazario
"""

from __future__ import annotations

import argparse
import json
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.request

import certifi
from datetime import datetime, timezone
from pathlib import Path

from phishing_pipeline.config import RAW_DIR
from phishing_pipeline.logging_utils import get_logger

logger = get_logger(__name__)

PHISHING_POT_URL = "https://github.com/rf-peixoto/phishing_pot.git"
NAZARIO_BASE = "https://monkey.org/~jose/phishing/"
SPAMASSASSIN_BASE = "https://spamassassin.apache.org/old/publiccorpus/"

# Ficheros de Nazario. Los `phishing-AAAA` son las cosechas anuales; los
# `phishingN.mbox` y `20051114.mbox`, los archivos históricos originales.
NAZARIO_FICHEROS = (
    [f"phishing-{anio}" for anio in range(2015, 2026)]
    + ["phishing0.mbox", "phishing1.mbox", "phishing2.mbox", "phishing3.mbox"]
    + ["20051114.mbox"]
)

# Solo los archivos de correo legítimo. Los de spam existen en el mismo directorio
# y se omiten deliberadamente: ver la nota sobre validez de constructo arriba.
SPAMASSASSIN_HAM = (
    "20030228_easy_ham.tar.bz2",
    "20030228_easy_ham_2.tar.bz2",
    "20030228_hard_ham.tar.bz2",
)

TIEMPO_ESPERA = 180


def _contexto_tls() -> ssl.SSLContext:
    """
    Contexto TLS con verificación completa, anclado al almacén de `certifi`.

    Por qué no vale el contexto por defecto. En Windows, `ssl.create_default_context()`
    carga los certificados del almacén del sistema, y ese almacén conserva cadenas
    de certificación caducadas —el cruce histórico de Let's Encrypt con DST Root CA
    X3, expirado en 2021— que OpenSSL puede elegir al construir la ruta de
    confianza. El resultado es un error de «certificado caducado» sobre un
    certificado que está perfectamente vigente: se comprobó contra monkey.org, cuyo
    certificado vence en septiembre de 2026 y aun así era rechazado.

    Anclar la verificación al conjunto de `certifi`, que se mantiene al día como
    dependencia del proyecto, resuelve el caso **sin relajar la verificación**.
    Desactivarla habría sido la solución fácil y la equivocada: estos ficheros
    proceden de terceros y se procesan después con analizadores de correo y de HTML.
    """
    return ssl.create_default_context(cafile=certifi.where())


def _descargar(url: str, destino: Path) -> bool:
    """Descarga un fichero si no existe ya. Devuelve True si el fichero está listo."""
    if destino.exists() and destino.stat().st_size > 0:
        logger.info("Ya existe, se omite: %s", destino.name)
        return True
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=TIEMPO_ESPERA, context=_contexto_tls()) as respuesta:
            datos = respuesta.read()
    except Exception as exc:
        logger.warning("No se pudo descargar %s: %s", url, exc)
        return False
    if not datos:
        logger.warning("Descarga vacía: %s", url)
        return False
    destino.write_bytes(datos)
    logger.info("Descargado %s (%.1f MB)", destino.name, len(datos) / 1e6)
    return True


def descargar_phishing_pot(raw_dir: Path | None = None) -> Path:
    """
    Clona phishing_pot con profundidad 1.

    Si el directorio ya existe se intenta actualizar en lugar de volver a clonar,
    porque el repositorio crece de forma continua: la instantánea que contenía el
    corpus anterior tenía 2,896 muestras y la vigente supera las 8,600.
    """
    destino = (raw_dir or RAW_DIR) / "phishing_pot"
    if (destino / ".git").exists():
        logger.info("phishing_pot ya clonado; actualizando")
        subprocess.run(
            ["git", "-C", str(destino), "pull", "--depth", "1", "--ff-only"],
            check=False,
            capture_output=True,
        )
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Clonando phishing_pot (profundidad 1)")
    resultado = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", PHISHING_POT_URL, str(destino)],
        check=False,
        capture_output=True,
        text=True,
    )
    if resultado.returncode != 0:
        raise RuntimeError(f"git clone falló: {resultado.stderr.strip()}")
    return destino


def descargar_nazario(raw_dir: Path | None = None) -> Path:
    """
    Descarga los ficheros mbox de Nazario.

    Los ficheros que no estén disponibles se omiten con una advertencia en lugar
    de abortar: el índice del sitio cambia con el tiempo —cada año se añade una
    cosecha— y una URL caída no debe impedir construir el corpus con el resto.
    """
    destino = (raw_dir or RAW_DIR) / "Nazario"
    destino.mkdir(parents=True, exist_ok=True)
    obtenidos = 0
    for nombre in NAZARIO_FICHEROS:
        # Se normaliza la extensión para que el limpiador reconozca todos como mbox.
        local = destino / (nombre if nombre.endswith(".mbox") else f"{nombre}.mbox")
        if _descargar(NAZARIO_BASE + nombre, local):
            obtenidos += 1
    if obtenidos == 0:
        raise RuntimeError("No se pudo descargar ningún fichero de Nazario")
    logger.info("Nazario: %d de %d ficheros disponibles", obtenidos, len(NAZARIO_FICHEROS))
    return destino


def descargar_spamassassin(raw_dir: Path | None = None) -> Path:
    """Descarga y extrae los archivos de correo legítimo de SpamAssassin."""
    destino = (raw_dir or RAW_DIR) / "SpamAssassin"
    destino.mkdir(parents=True, exist_ok=True)
    for nombre in SPAMASSASSIN_HAM:
        archivo = destino / nombre
        if not _descargar(SPAMASSASSIN_BASE + nombre, archivo):
            continue
        marca = destino / f".{nombre}.extraido"
        if marca.exists():
            continue
        try:
            with tarfile.open(archivo, "r:bz2") as tar:
                # `filter="data"` bloquea rutas absolutas y escapes con ".." dentro
                # del archivo comprimido. Es obligatorio: estos archivos proceden de
                # un tercero y extraer sin filtro permite escribir fuera del destino.
                try:
                    tar.extractall(destino, filter="data")
                except TypeError:  # Python anterior a 3.12
                    tar.extractall(destino)
            marca.touch()
            logger.info("Extraído %s", nombre)
        except Exception as exc:
            logger.warning("No se pudo extraer %s: %s", nombre, exc)
    return destino


def _contar(ruta: Path) -> int:
    return sum(
        1
        for p in ruta.rglob("*")
        if p.is_file() and p.suffix not in {".bz2", ".gz"} and p.name != "cmds"
    )


def descargar_todo(raw_dir: Path | None = None, solo: str | None = None) -> dict:
    """Descarga las tres fuentes y devuelve un manifiesto con lo obtenido."""
    raw_dir = raw_dir or RAW_DIR
    tareas = {
        "phishing_pot": descargar_phishing_pot,
        "nazario": descargar_nazario,
        "spamassassin": descargar_spamassassin,
    }
    if solo:
        if solo not in tareas:
            raise ValueError(f"Fuente desconocida: {solo!r}. Válidas: {sorted(tareas)}")
        tareas = {solo: tareas[solo]}

    manifiesto: dict = {"generado_en": datetime.now(timezone.utc).isoformat(), "fuentes": {}}
    for nombre, funcion in tareas.items():
        try:
            ruta = funcion(raw_dir)
            manifiesto["fuentes"][nombre] = {
                "ruta": str(ruta),
                "ficheros": _contar(ruta),
                "estado": "ok",
            }
        except Exception as exc:
            logger.error("Fallo al descargar %s: %s", nombre, exc)
            manifiesto["fuentes"][nombre] = {"estado": "error", "error": str(exc)}

    ruta_manifiesto = raw_dir / "correo_real_manifiesto.json"
    ruta_manifiesto.parent.mkdir(parents=True, exist_ok=True)
    ruta_manifiesto.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifiesto


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument(
        "--solo",
        type=str,
        default=None,
        choices=["phishing_pot", "nazario", "spamassassin"],
        help="Descargar una sola fuente en lugar de las tres.",
    )
    args = parser.parse_args()

    if shutil.which("git") is None and args.solo in (None, "phishing_pot"):
        logger.error("git no está disponible y hace falta para clonar phishing_pot")
        return 2

    manifiesto = descargar_todo(args.raw_dir, args.solo)
    print(json.dumps(manifiesto, indent=2, ensure_ascii=False))
    return 0 if all(f["estado"] == "ok" for f in manifiesto["fuentes"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
