"""
Limpiador único para correo en formato crudo: `.eml` individual y `mbox`.

Por qué uno solo y no uno por fuente. Las tres fuentes de correo real del corpus
—phishing_pot, Nazario y SpamAssassin— publican mensajes RFC 5322 completos, de
modo que un limpiador por fuente solo podría diferir en detalles accidentales. Y
diferir es exactamente el peligro: la revisión del pipeline encontró que
`clean_text` se construía de tres maneras distintas según la fuente (cuerpo crudo
en dos casos, HTML despojado en el tercero), lo que convierte el preprocesamiento
mismo en una huella del origen. Con un único camino de código esa clase de
asimetría no puede aparecer.

**Sobre las cabeceras de autenticación.** Se extraen de todas las fuentes con el
mismo código, que es la corrección simétrica que exige A3. Conviene advertir, sin
embargo, que la simetría del CÓDIGO no garantiza la simetría del DATO: medido
sobre los corpus reales, `Authentication-Results` aparece en el 99.6% de
phishing_pot, el 34.2% de Nazario y el **0.0%** de SpamAssassin. La causa es
temporal y no tiene arreglo en el pipeline: SpamAssassin se recolectó en 2002-2003
y SPF, DKIM y DMARC son posteriores.

La consecuencia práctica es que esos tres campos quedan confundidos con la clase
por la vía de la época, y `phishing_pipeline.auditoria_fugas` lo detectará. La
decisión de retirarlos corresponde a la etapa de selección de características
(experimento E5), no a la de extracción: el limpiador extrae
lo que hay y deja constancia; quien decide qué entra al modelo es la auditoría.
Ocultar el campo aquí impediría además medir el problema.
"""

from __future__ import annotations

import email
import hashlib
import mailbox
import re
import uuid
from datetime import datetime, timezone
from email import policy
from email.message import Message
from pathlib import Path
from typing import Iterator

import pandas as pd

from phishing_pipeline.features.dom_parser import parse_email_multimodal
from phishing_pipeline.features.dom_stats import compute_dom_stats
from phishing_pipeline.features.network import (
    URL_PATTERN,
    extract_lexical_url_features,
    extract_technical_features,
)
from phishing_pipeline.logging_utils import get_logger
from phishing_pipeline.schema import empty_canonical_row

logger = get_logger(__name__)

_SPF_RE = re.compile(r"spf\s*=\s*(\w+)", re.IGNORECASE)
_DKIM_RE = re.compile(r"dkim\s*=\s*(\w+)", re.IGNORECASE)
_DMARC_RE = re.compile(r"dmarc\s*=\s*(\w+)", re.IGNORECASE)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Índice de salto de una cabecera ARC: `i=1` es el receptor original.
_ARC_I_RE = re.compile(r"\bi\s*=\s*(\d+)")

# Longitud mínima del cuerpo, en caracteres. Por debajo de este umbral el mensaje
# no aporta texto que clasificar y suele ser un resto de la conversión.
MIN_CUERPO = 6


def _decodificar(parte: Message) -> str:
    """Contenido de una parte MIME como texto, tolerando codificaciones rotas."""
    try:
        bruto = parte.get_payload(decode=True)
    except Exception:  # pragma: no cover — payloads malformados
        return ""
    if not bruto:
        return ""
    juego = parte.get_content_charset() or "utf-8"
    try:
        return bruto.decode(juego, errors="ignore")
    except (LookupError, UnicodeDecodeError):
        return bruto.decode("utf-8", errors="ignore")


def extraer_cuerpos(msg: Message) -> tuple[str, str]:
    """
    Devuelve `(html, texto_plano)` recorriendo el árbol MIME.

    Se conserva la PRIMERA parte de cada tipo. Los mensajes multiparte alternativos
    presentan la misma información en ambos formatos y quedarse con la primera de
    cada uno evita concatenar repeticiones del mismo contenido, que inflarían el
    número de palabras y con él una de las características estructurales.
    """
    html = plano = ""
    if msg.is_multipart():
        for parte in msg.walk():
            if parte.get_content_maintype() == "multipart":
                continue
            tipo = parte.get_content_type()
            if tipo == "text/html" and not html:
                html = _decodificar(parte)
            elif tipo == "text/plain" and not plano:
                plano = _decodificar(parte)
    else:
        contenido = _decodificar(msg)
        if msg.get_content_type() == "text/html":
            html = contenido
        else:
            plano = contenido
    return html, plano


def extraer_autenticacion(msg: Message) -> dict[str, str | None]:
    """
    Resultados de SPF, DKIM y DMARC a partir de las cabeceras del mensaje.

    Se consultan tres orígenes, en este orden:

    1. `Authentication-Results` (RFC 8601), que escribe el servidor receptor.
    2. `ARC-Authentication-Results` (RFC 8617), que preserva el dictamen del
       receptor ORIGINAL cuando el mensaje pasó después por un reenviador.
    3. `Received-SPF` como respaldo para SPF, único de los tres con cabecera propia.

    El segundo origen no es un caso de borde. Se midió sobre archivos de listas de
    discusión que el relé intermedio escribe en `Authentication-Results` únicamente
    `arc=none smtp.client-ip=…`, sin resultado alguno, y deja los verdaderos en la
    cadena ARC: sobre 1,500 mensajes, `Authentication-Results` contenía `spf=` en el
    0% y `ARC-Authentication-Results` en el 77.3%. Leer solo la primera hacía que un
    correo con autenticación completa se registrara como si careciera de ella, y esa
    es precisamente la modalidad que el corpus solo tiene en una de las dos clases.

    Cuando hay varias cabeceras ARC —una por salto— se recorren de la más antigua a
    la más reciente, porque la primera corresponde al receptor original, que es quien
    pudo verificar contra la IP y el dominio del emisor de verdad.

    Los valores se devuelven en minúscula y tal como aparecen, sin forzarlos a un
    conjunto cerrado: un valor inesperado debe poder observarse en el corpus en lugar
    de quedar silenciosamente reclasificado.
    """
    resultado: dict[str, str | None] = {"spf": None, "dkim": None, "dmarc": None}
    patrones = (("spf", _SPF_RE), ("dkim", _DKIM_RE), ("dmarc", _DMARC_RE))

    def leer(cabecera: str) -> None:
        for clave, patron in patrones:
            if resultado[clave] is not None:
                continue
            m = patron.search(cabecera)
            if m:
                resultado[clave] = m.group(1).lower()

    for cabecera in msg.get_all("Authentication-Results") or []:
        leer(str(cabecera))

    # `i=` numera los saltos ARC; el 1 es el receptor original. Si falta el índice
    # se conserva el orden de aparición, que ya es el de llegada.
    arc = msg.get_all("ARC-Authentication-Results") or []
    for cabecera in sorted(arc, key=lambda c: _indice_arc(str(c))):
        leer(str(cabecera))

    if resultado["spf"] is None:
        recibido = msg.get("Received-SPF")
        if recibido and str(recibido).strip():
            resultado["spf"] = str(recibido).strip().split()[0].lower()

    return resultado


def _indice_arc(cabecera: str) -> int:
    """Número de salto `i=` de una cabecera ARC; 99 si no lo declara."""
    m = _ARC_I_RE.search(cabecera)
    return int(m.group(1)) if m else 99


def extraer_fecha(msg: Message) -> str | None:
    """
    Fecha del mensaje en ISO 8601, preferiendo la que estampó el servidor.

    La cabecera `Date` la escribe el remitente y en correo de phishing se
    falsifica: en phishing_pot se observaron fechas hasta 2031. Usarla sin más
    para una partición temporal colocaría en el futuro correo que se recogió en
    el pasado, y la partición dejaría de significar lo que dice.

    Se prefiere por eso la marca de tiempo de la última cabecera `Received`, que
    escribe el primer servidor que recibió el mensaje y el remitente no controla.
    `Date` queda como respaldo para las colecciones que no conservan el
    encaminamiento, y se descartan los valores fuera de un rango plausible.
    """
    from email.utils import parsedate_to_datetime

    def plausible(dt) -> str | None:
        if dt is None:
            return None
        # El correo electrónico es de 1971 en adelante; una fecha posterior al
        # momento de la ejecución es necesariamente falsa o un error de zona.
        ahora = datetime.now(timezone.utc)
        if not (1971 <= dt.year <= ahora.year):
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()

    recibidas = msg.get_all("Received") or []
    if recibidas:
        # El sello va tras el último punto y coma de la cabecera.
        cola = str(recibidas[-1]).rsplit(";", 1)
        if len(cola) == 2:
            try:
                if (v := plausible(parsedate_to_datetime(cola[1].strip()))):
                    return v
            except (TypeError, ValueError):
                pass

    try:
        return plausible(parsedate_to_datetime(str(msg.get("Date") or "")))
    except (TypeError, ValueError):
        return None


def _ip_de_origen(msg: Message) -> str | None:
    """
    Dirección del primer salto SMTP, tomada de la última cabecera `Received`.

    Las cabeceras `Received` se apilan en orden inverso —la más reciente arriba—,
    de modo que la última corresponde al servidor que originó el mensaje, que es la
    informativa. Tomar la primera daría la dirección del servidor de destino, que es
    la misma para todo el corpus y por tanto inútil.
    """
    recibidas = msg.get_all("Received") or []
    if not recibidas:
        return None
    ips = _IP_RE.findall(str(recibidas[-1]))
    return ips[0] if ips else None


def _mensaje_a_fila(msg: Message, fuente: str, etiqueta: int) -> dict | None:
    """Convierte un mensaje al esquema canónico, o None si no es utilizable."""
    html, plano = extraer_cuerpos(msg)
    cuerpo = html or plano
    if not cuerpo or len(cuerpo.strip()) < MIN_CUERPO:
        return None

    dom = parse_email_multimodal(cuerpo)
    texto = str(dom.get("clean_text") or plano or cuerpo)
    if len(texto.strip()) < MIN_CUERPO:
        return None

    _, indicadores = extract_technical_features(cuerpo)
    auth = extraer_autenticacion(msg)
    # Las características léxicas de URL y los estadísticos de complejidad del DOM
    # se calculan AQUÍ y no en una etapa posterior de enriquecimiento. El pipeline
    # histórico las añadía después, sobre el corpus ya unificado, y una fuente
    # incorporada más tarde se quedaba con los valores por defecto —cero— sin que
    # nada lo advirtiera: se comprobó que las tres fuentes de correo real tenían
    # `url_max_length`, `url_domain_max_entropy` y `total_nodos_dom` en cero para
    # el 100% de sus filas. Un cero indistinguible de la ausencia de cálculo es la
    # misma confusión que ya se identificó entre «no se sabe» y «vale cero».
    lexicas = extract_lexical_url_features(cuerpo)
    dom_stats = compute_dom_stats(cuerpo)

    fila = empty_canonical_row()
    fila.update(
        {
            "email_id": str(uuid.uuid4()),
            "source_dataset": fuente,
            "label": int(etiqueta),
            "label_text": "phishing" if etiqueta == 1 else "safe",
            "subject": str(msg.get("Subject") or ""),
            "sender": str(msg.get("From") or ""),
            "sent_date": extraer_fecha(msg),
            "body_raw": cuerpo,
            "body_plain": plano,
            "body_html": html,
            "clean_text": texto,
            "has_html": int(dom.get("has_html", 0)),
            "network_indicators": indicadores,
            "num_links": int(dom.get("num_links", 0)),
            "num_images": int(dom.get("num_images", 0)),
            "has_form": int(dom.get("has_form", 0)),
            "has_iframe": int(dom.get("has_iframe", 0)),
            "has_javascript": int(dom.get("has_javascript", 0)),
            "has_ip_link": int(dom.get("has_ip_link", 0)),
            "word_count": len(texto.split()),
            "received_origin_ip": _ip_de_origen(msg),
            "spf_result": auth["spf"],
            "dkim_result": auth["dkim"],
            "dmarc_result": auth["dmarc"],
            "num_urls_metadata": len(URL_PATTERN.findall(cuerpo)),
            "processing_status": dom.get("processing_status", "ok"),
            "processing_error": dom.get("processing_error"),
        }
    )
    fila.update(lexicas)
    fila.update(dom_stats)
    return fila


def _mensajes_de(ruta: Path) -> Iterator[Message]:
    """Mensajes de un `.eml` suelto, de un `mbox`, o de un directorio de cualquiera."""
    if ruta.is_dir():
        for hijo in sorted(ruta.rglob("*")):
            if hijo.is_file():
                yield from _mensajes_de(hijo)
        return

    if ruta.suffix.lower() in {".bz2", ".gz", ".zip", ".tar"} or ruta.name == "cmds":
        return

    if ruta.suffix.lower() == ".mbox":
        # `create=False` es obligatorio: por defecto `mailbox.mbox` CREA el fichero
        # si no existe, de modo que un limpiador —que solo debería leer— acababa
        # escribiendo un mbox vacío en el directorio de datos crudos. Se observó
        # exactamente eso: un fichero de 0 bytes aparecido con la marca de tiempo de
        # la limpieza, no la de la descarga, que además enmascaraba una descarga
        # fallida haciéndola parecer un fichero presente pero vacío.
        if ruta.stat().st_size == 0:
            logger.warning("mbox vacío, se omite: %s", ruta)
            return
        try:
            for msg in mailbox.mbox(str(ruta), create=False):
                yield msg
        except Exception as exc:  # pragma: no cover — mbox corrupto
            logger.warning("No se pudo leer el mbox %s: %s", ruta, exc)
        return

    try:
        yield email.message_from_bytes(ruta.read_bytes(), policy=policy.default)
    except Exception as exc:  # pragma: no cover — mensaje malformado
        logger.debug("No se pudo leer %s: %s", ruta, exc)


def limpiar_correo_crudo(ruta: Path, fuente: str, etiqueta: int) -> pd.DataFrame:
    """
    Convierte una colección de correo crudo al esquema canónico.

    Args:
        ruta: fichero `.eml`, fichero `mbox`, o directorio que contenga cualquiera
            de los dos de forma recursiva.
        fuente: valor de `source_dataset`. Debe identificar la PROCEDENCIA, no el
            fichero, porque es la unidad sobre la que se construyen los pliegues.
        etiqueta: 1 para phishing, 0 para legítimo.

    La deduplicación exacta se hace aquí, sobre el texto normalizado, porque las
    colecciones de phishing se solapan entre sí: se midió que Nazario y
    phishing_pot comparten 1,169 mensajes. Dejar que ese solape llegue al corpus
    haría que el mismo correo apareciera en dos fuentes distintas y, bajo el
    protocolo por fuente no observada, a ambos lados de la partición.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la ruta de correo crudo: {ruta}")

    filas: list[dict] = []
    vistos: set[str] = set()
    descartados = duplicados = 0

    for msg in _mensajes_de(ruta):
        if msg is None:
            continue
        fila = _mensaje_a_fila(msg, fuente, etiqueta)
        if fila is None:
            descartados += 1
            continue
        huella = hashlib.blake2b(
            " ".join(fila["clean_text"].split()).lower().encode("utf-8"), digest_size=16
        ).hexdigest()
        if huella in vistos:
            duplicados += 1
            continue
        vistos.add(huella)
        filas.append(fila)

    logger.info(
        "%s: %d mensajes utilizables, %d sin cuerpo, %d duplicados exactos",
        fuente,
        len(filas),
        descartados,
        duplicados,
    )
    df = pd.DataFrame(filas)
    df.attrs["estadisticas"] = {
        "fuente": fuente,
        "utilizables": len(filas),
        "sin_cuerpo": descartados,
        "duplicados_exactos": duplicados,
    }
    return df
