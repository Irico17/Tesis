"""
Descarga correo legítimo desde archivos públicos de listas de discusión.

Por qué hace falta. El corpus actual tiene un desequilibrio que no es de volumen
sino estructural: de los 10,125 correos con las tres modalidades, 9,916 son
phishing y 209 legítimos. La causa está en una sola columna --ninguna colección
de correo legítimo publicada conserva el marcado ni las cabeceras-- y mientras no
se corrija no puede entrenarse un clasificador binario sobre el subconjunto
completo, que es justamente el experimento que separa si la fusión aporta.

Los archivos de listas de discusión sí conservan el mensaje íntegro: cadena
`Received` con direcciones IP, `Authentication-Results` con SPF, DKIM y DMARC, y
el cuerpo tal como se envió. Son además correo legítimo verificable, no una
colección etiquetada por un tercero.

Se usa la interfaz de public-inbox de lore.kernel.org, que exporta el resultado
de una búsqueda como mbox comprimido. La exportación exige POST: con GET, el
servidor responde la página de resultados e ignora el parámetro, de modo que una
implementación con GET parece funcionar y devuelve HTML.

    python -m phishing_pipeline.downloaders.listas_correo --listar
    python -m phishing_pipeline.downloaders.listas_correo --muestra
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import certifi

logger = logging.getLogger(__name__)

BASE = "https://lore.kernel.org"
AGENTE = "tesis-pucp/1.0 (investigacion academica; deteccion de phishing)"
# Cortesía con un servicio público y gratuito: una pausa entre descargas.
PAUSA_S = 2.0

DESTINO = Path("data/Datasets_Originales/listas_correo")

# Listas y ventanas temporales de la muestra. Se eligen de forma deliberada:
#
#  - Se mezclan listas de DESARROLLO (lkml, git) con listas de USUARIOS
#    (linux-perf-users, linux-rt-users). Las primeras tienen una cultura de texto
#    plano muy marcada; las segundas reciben correo de gente que escribe desde
#    clientes gráficos, que es donde puede aparecer marcado.
#  - Se abren varias ventanas separadas en el tiempo para no medir el estilo de
#    un único hilo ni de una única campaña de parches.
MUESTRA: tuple[tuple[str, str], ...] = (
    ("linux-perf-users", "d:20250101..20250301"),
    ("linux-rt-users", "d:20240101..20241231"),
    ("git", "d:20250101..20250201"),
    ("linux-kernel-announce", "d:20200101..20251231"),
    ("linux-doc", "d:20250101..20250201"),
    ("selinux", "d:20240101..20241231"),
    ("fuse-devel", "d:20230101..20241231"),
    ("criu", "d:20220101..20241231"),
)


# --------------------------------------------------------------- HyperKitty
#
# Segunda fuente, con un compromiso distinto al de public-inbox. HyperKitty
# conserva el cuerpo tal como se envió --y por tanto el marcado-- pero descarta
# las cabeceras de encaminamiento antes de publicar, porque revelan la dirección
# IP del remitente. public-inbox hace lo contrario. Medido: 18.67% de estructura
# en Fedora frente al 1.09% de lore.kernel.org, y 0% de autenticación frente al
# 77.3%. Se descargan ambas porque ninguna basta por separado.
HK_BASE = "https://lists.fedoraproject.org/archives"
DESTINO_HTML = Path("data/Datasets_Originales/listas_html")

# Selección deliberada, no las más activas. Se mezclan tres perfiles porque el
# marcado no se reparte por igual: las listas de desarrollo tienen cultura de
# texto plano, las de usuarios reciben correo de gente que escribe desde clientes
# gráficos, y las de comunidad y regionales son las que más se apartan del
# formato técnico. Sin esa mezcla, el correo legítimo del corpus procedería de un
# único perfil de remitente.
LISTAS_HTML: tuple[str, ...] = (
    # usuarios y soporte
    "users", "epel-devel", "server", "desktop", "kde", "test", "arm",
    "389-users", "freeipa-users", "sssd-users", "virt", "cloud",
    # desarrollo
    "devel", "python-devel", "perl-devel", "packaging", "infrastructure",
    "package-review", "haskell-devel", "games", "coreos", "triage",
    # comunidad, documentación y regionales
    "ambassadors", "marketing", "design-team", "websites", "docs",
    "council-discuss", "diversity", "commops",
    "argentina", "apac", "anz", "india", "br-users", "bangladesh-users",
)

VENTANAS: tuple[tuple[str, str], ...] = tuple(
    (f"{a}-01-01", f"{a + 1}-01-01") for a in range(2015, 2026)
)

# Topes de seguridad. La descarga no debe crecer sin control sobre un servicio
# público y gratuito, y el global no basta por sí solo: `users` aporta 76,553
# mensajes en once años, de modo que un tope únicamente global se agotaría con
# una o dos listas y el correo legítimo del corpus procedería de una sola
# comunidad. Es el mismo defecto que se está corrigiendo, trasladado a la clase
# negativa. El tope por lista fuerza el reparto.
TOPE_MENSAJES = 200_000
TOPE_POR_LISTA = 9_000


def descargar_hyperkitty(lista: str, ini: str, fin: str, destino: Path) -> tuple[Path | None, int]:
    """Exporta en mbox el tramo [ini, fin) de una lista. Devuelve ruta y mensajes."""
    salida = destino / f"{lista}__{ini[:4]}.mbox"
    if salida.exists():
        # Un fichero de cero bytes es un tramo sin mensajes ya comprobado.
        if salida.stat().st_size == 0:
            return None, 0
        crudo = salida.read_bytes()
        return salida, crudo.count(b"\nFrom ") + (1 if crudo.startswith(b"From ") else 0)

    correo = f"{lista}@lists.fedoraproject.org"
    url = f"{HK_BASE}/list/{correo}/export/{lista}.mbox.gz?start={ini}&end={fin}"
    req = urllib.request.Request(url, headers={"User-Agent": AGENTE})
    try:
        with urllib.request.urlopen(req, context=_contexto(), timeout=300) as r:
            datos = r.read()
    except Exception as exc:  # noqa: BLE001 — una lista inexistente no frena el resto
        logger.debug("%s %s: %s", lista, ini[:4], exc)
        return None, 0

    crudo = gzip.decompress(datos) if datos[:2] == b"\x1f\x8b" else datos
    if not crudo.strip():
        salida.parent.mkdir(parents=True, exist_ok=True)
        salida.touch()  # marca el tramo como comprobado y vacío
        return None, 0

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(crudo)
    n = crudo.count(b"\nFrom ") + (1 if crudo.startswith(b"From ") else 0)
    return salida, n


def descargar_fedora(destino: Path = DESTINO_HTML,
                     tope: int = TOPE_MENSAJES,
                     tope_lista: int = TOPE_POR_LISTA) -> tuple[int, int]:
    """Recorre listas y años, repartiendo el volumen entre listas."""
    destino.mkdir(parents=True, exist_ok=True)
    total = ficheros = 0
    for lista in LISTAS_HTML:
        de_la_lista = 0
        # Los años se recorren en orden alterno --el más reciente, el más
        # antiguo, y así hacia el centro-- para que una lista que alcance su
        # tope no quede representada por una sola época.
        orden = [v for par in zip(reversed(VENTANAS), VENTANAS) for v in par]
        vistos = set()
        for ini, fin in orden:
            if ini in vistos or de_la_lista >= tope_lista:
                continue
            vistos.add(ini)
            ruta, n = descargar_hyperkitty(lista, ini, fin, destino)
            if ruta and n:
                ficheros += 1
                total += n
                de_la_lista += n
            time.sleep(PAUSA_S)
            if total >= tope:
                logger.info("tope global de %d mensajes alcanzado", tope)
                logger.info("TOTAL %d mensajes en %d ficheros", total, ficheros)
                return total, ficheros
        if de_la_lista:
            logger.info("%-22s %7d mensajes   (acumulado %d)", lista, de_la_lista, total)
    logger.info("TOTAL %d mensajes en %d ficheros", total, ficheros)
    return total, ficheros


def _contexto() -> ssl.SSLContext:
    """Verificación anclada al paquete de certificados de la biblioteca.

    El almacén del sistema de la máquina de desarrollo contiene un intermedio
    caducado que hace fallar descargas válidas. Se ancla, no se desactiva.
    """
    return ssl.create_default_context(cafile=certifi.where())


def listas_disponibles() -> list[tuple[str, str]]:
    """Nombre y descripción de cada lista publicada en el archivo."""
    req = urllib.request.Request(BASE + "/", headers={"User-Agent": AGENTE})
    with urllib.request.urlopen(req, context=_contexto(), timeout=30) as r:
        html = r.read().decode("utf-8", "replace")
    pares = re.findall(r'href="([^"/]+)/?">([^<]+)</a>\s*\n\s*([^\n*]*)', html)
    return [(h, d.strip()) for h, _, d in pares if h != "all"]


def descargar_lista(lista: str, consulta: str, destino: Path) -> Path | None:
    """Exporta como mbox el resultado de una búsqueda. Devuelve la ruta escrita."""
    salida = destino / f"{lista}__{consulta.replace(':', '_').replace('..', '-')}.mbox"
    if salida.exists() and salida.stat().st_size > 0:
        logger.info("ya estaba: %s (%.1f MiB)", salida.name, salida.stat().st_size / 2**20)
        return salida

    url = f"{BASE}/{lista}/?q={urllib.parse.quote(consulta)}&x=m"
    cuerpo = urllib.parse.urlencode({"q": consulta, "x": "m"}).encode()
    req = urllib.request.Request(url, data=cuerpo, headers={"User-Agent": AGENTE})
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, context=_contexto(), timeout=180) as r:
            datos = r.read()
            tipo = r.headers.get("Content-Type", "")
    except Exception as exc:  # noqa: BLE001 — una lista caída no debe frenar el resto
        logger.warning("%s: no se pudo descargar (%s: %s)", lista, type(exc).__name__, exc)
        return None

    # Si el servidor devuelve HTML es que ignoró la exportación; escribirlo
    # produciría un mbox que el limpiador leería como cero mensajes, en silencio.
    if "gzip" not in tipo and "mbox" not in tipo:
        logger.warning("%s: el servidor devolvió %r en lugar de un mbox", lista, tipo)
        return None

    crudo = gzip.decompress(datos) if datos[:2] == b"\x1f\x8b" else datos
    if not crudo.strip():
        logger.warning("%s: la búsqueda %r no devolvió mensajes", lista, consulta)
        return None

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_bytes(crudo)
    n = crudo.count(b"\nFrom ") + (1 if crudo.startswith(b"From ") else 0)
    logger.info("%s: %d mensajes, %.1f MiB, %.1fs",
                lista, n, len(crudo) / 2**20, time.time() - t0)
    return salida


def descargar_muestra(destino: Path = DESTINO,
                      muestra: tuple[tuple[str, str], ...] = MUESTRA) -> list[Path]:
    escritos = []
    for i, (lista, consulta) in enumerate(muestra):
        ruta = descargar_lista(lista, consulta, destino)
        if ruta:
            escritos.append(ruta)
        if i < len(muestra) - 1:
            time.sleep(PAUSA_S)
    return escritos


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listar", action="store_true", help="enumera las listas del archivo")
    p.add_argument("--muestra", action="store_true", help="descarga la muestra definida")
    p.add_argument("--fedora", action="store_true",
                   help="descarga los archivos de Fedora (HyperKitty), que sí conservan marcado")
    p.add_argument("--tope", type=int, default=TOPE_MENSAJES)
    p.add_argument("--destino", type=Path, default=None)
    args = p.parse_args()

    if args.listar:
        for h, d in listas_disponibles():
            print(f"{h:34s} {d}")
        return 0
    if args.muestra:
        destino = args.destino or DESTINO
        rutas = descargar_muestra(destino)
        print(f"\n{len(rutas)} fichero(s) en {destino}")
        return 0 if rutas else 1
    if args.fedora:
        destino = args.destino or DESTINO_HTML
        total, ficheros = descargar_fedora(destino, args.tope)
        print(f"\n{total:,} mensajes en {ficheros} ficheros, en {destino}")
        return 0 if total else 1
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
