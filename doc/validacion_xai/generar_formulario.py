"""
Genera el formulario del panel de analistas de R3.3 a partir del instrumento y de
los ejemplos reales que deja E9.

Nada del formulario se escribe a mano. Las afirmaciones de las secciones A (SUS) y
B (PSSUQ) se leen de `instrumento_evaluacion.md`; los correos y sus explicaciones,
de `medios_de_verificacion/experimentos/e9/ejemplos_del_panel.json`; y las
preguntas de comprensión, con su respuesta correcta, se derivan del propio reporte
que el evaluador lee. Así el formulario no puede describir una explicación
distinta de la que el sistema generó, ni la clave de respuestas discrepar de ella.

Produce tres ficheros en `doc/validacion_xai/formulario/`:

- `crear_formulario.gs`: Apps Script que crea el formulario en Google Forms.
  Se pega en script.google.com y se ejecuta una vez.
- `clave_de_respuestas.json`: respuesta correcta de cada pregunta de la
  sección C, que usa `respuestas_a_csv.py` para calificar.
- `vista_previa.html`: el formulario completo en una página, para revisarlo o
  imprimirlo.
- `resaltado_*.png`: cada correo con sus palabras sombreadas según su peso en
  la decisión (`phishing_model.xai.resaltado`). El Apps Script las inserta
  desde el repositorio público, de modo que deben estar subidas antes de
  ejecutarlo.

El formulario no se configura como cuestionario con nota: Google Forms mostraría
las respuestas correctas al terminar y un evaluador podría pasárselas a otro. La
calificación se hace después, con la clave.

    python doc/validacion_xai/generar_formulario.py
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
BASE = AQUI.parents[1]
INSTRUMENTO = AQUI / "instrumento_evaluacion.md"
EJEMPLOS = BASE / "medios_de_verificacion" / "experimentos" / "e9" / "ejemplos_del_panel.json"
SALIDA = AQUI / "formulario"

TITULO = "Evaluación de explicaciones de un detector de phishing"
AUTOR = "Cristhofer Alegre"
CONTACTO = "a20210577@pucp.edu.pe"
# Desde aquí el Apps Script descarga las imágenes de resaltado.
URL_IMAGENES = ("https://raw.githubusercontent.com/Irico17/Tesis/master/"
                "doc/validacion_xai/formulario/")
# Las posiciones de los cuatro correos en el informe de selección.
CLAVES = ("phishing_1", "legitimo_1", "phishing_2", "legitimo_2")

# Dos versiones del panel. La principal equilibra las clases e incluye un correo
# con peso no textual; la de claridad toma, sin atender a la clase, los correos
# cuya explicación se entiende mejor e incluye entre ellos el del ejemplo guiado
# (ver scripts/experimentos/e9_ejemplos_por_claridad.py).
VERSIONES = {
    "principal": {"ejemplos": "ejemplos_del_panel.json", "salida": "formulario",
                  "claves": CLAVES},
    "claridad": {"ejemplos": "ejemplos_del_panel_claridad.json", "recortar_guia": True,
                 "salida": "formulario_claridad",
                 "claves": ("correo_1", "correo_2", "correo_3", "correo_4")},
}


def _usar_version(nombre: str) -> None:
    """Fija los rutas y las posiciones de la versión que se genera."""
    global EJEMPLOS, SALIDA, URL_IMAGENES, CLAVES, RECORTAR_GUIA
    v = VERSIONES[nombre]
    EJEMPLOS = BASE / "medios_de_verificacion" / "experimentos" / "e9" / v["ejemplos"]
    SALIDA = AQUI / v["salida"]
    URL_IMAGENES = ("https://raw.githubusercontent.com/Irico17/Tesis/master/"
                    f"doc/validacion_xai/{v['salida']}/")
    CLAVES = v["claves"]
    RECORTAR_GUIA = v.get("recortar_guia", False)
ESCALA = ["1 (totalmente en desacuerdo)", "2", "3", "4", "5 (totalmente de acuerdo)"]

VACIAS = set("""
a an and are as at be been but by can do for from had has have he her his i if in
into is it its me my no not of on or our she so than that the their them then there
these they this to us was we were what when which who will with you your yours
would could should please thank thanks dear also just more only some such very
""".split())


# --------------------------------------------------------------------------
# Lectura del instrumento y de los ejemplos
# --------------------------------------------------------------------------

def _items(seccion_inicio: str, seccion_fin: str) -> list[str]:
    texto = INSTRUMENTO.read_text(encoding="utf-8")
    tramo = texto[texto.index(seccion_inicio):texto.index(seccion_fin)]
    filas = re.findall(r"^\| (\d+) \| (.+?) \| ☐", tramo, flags=re.M)
    return [t.strip() for _, t in sorted(filas, key=lambda f: int(f[0]))]


def _cuerpo_limpio(texto: str) -> str:
    lineas = [l.strip() for l in texto.splitlines()]
    salida, vacia = [], False
    for l in lineas:
        if not l:
            if not vacia and salida:
                salida.append("")
            vacia = True
            continue
        salida.append(l)
        vacia = False
    return "\n".join(salida).strip()


def _bloques(ejemplo: dict) -> dict[str, dict]:
    return {b["modality"]: b for b in ejemplo["explicacion"]["estructurado"]["modality_blocks"]}


def _palabras(ejemplo: dict) -> list[str]:
    return [f["name"] for f in _bloques(ejemplo)["texto"]["factors"]]


def _distractores(ejemplo: dict, n: int = 2) -> list[str]:
    """Palabras frecuentes del correo que el reporte NO cita.

    Son plausibles, porque están en el mensaje, y son incorrectas, porque el
    reporte no las menciona: la pregunta mide si se leyó el reporte y no si se
    adivina qué palabra «suena» a phishing.
    """
    citadas = {w.lower() for w in _palabras(ejemplo)}
    cuerpo = ejemplo["asunto"] + " " + ejemplo["cuerpo"]
    cuerpo = re.sub(r"\[[^\]]*\]|hxxps?://\S+", " ", cuerpo)
    conteo = Counter(w.lower() for w in re.findall(r"[A-Za-z]{5,}", cuerpo))
    candidatas = [w for w, _ in sorted(conteo.items(), key=lambda kv: (-kv[1], kv[0]))
                  if w not in citadas and w not in VACIAS]
    return candidatas[:n]


def _par_para_comparar(ejemplo: dict, diferencia_minima: float = 2.0) -> tuple[str, str]:
    """Dos palabras citadas, (la de más peso, otra de peso claramente menor).

    Solo palabras legibles y con contenido: comparar con un signo («,») confunde
    sin medir mejor la lectura del orden. Y la diferencia en el bloque ha de ser
    de al menos `diferencia_minima` puntos, porque dos palabras con 20.5% y 20.3%
    no permiten una respuesta inequívoca.
    """
    sys.path.insert(0, str(BASE / "src"))
    from phishing_model.xai.narrative import es_palabra_legible
    fs = [f for f in _bloques(ejemplo)["texto"]["factors"]
          if es_palabra_legible(f["name"]) and f["name"].isalpha()
          and f["name"].lower() not in VACIAS]
    for i, mayor in enumerate(fs):
        menores = [f for f in fs[i + 1:] if mayor["pct"] - f["pct"] >= diferencia_minima]
        if menores:
            return mayor["name"], menores[-1]["name"]
    p = _palabras(ejemplo)
    return p[0], p[2]


def _clase(ejemplo: dict) -> str:
    return "phishing" if ejemplo["explicacion"]["clase_predicha"] == 1 else "legítimo"


def _no_textuales(ejemplo: dict) -> list[str]:
    """Características de estructura y red que el reporte cita con peso, en orden."""
    b = _bloques(ejemplo)
    factores = [f for m in ("estructura", "red") for f in b[m]["factors"] if f["pct"] > 0]
    return [f["human_name"] for f in sorted(factores, key=lambda f: -f["pct"])]


_RASGOS: dict[str, dict] | None = None


def _partes(ejemplo: dict) -> dict:
    """Qué tiene el correo además del texto: formato HTML, imágenes y enlaces.

    Las preguntas sobre el diseño o los enlaces solo tienen sentido si el correo
    los tiene. Los rasgos se leen del corpus, que es lo que recibió el modelo; el
    correo mostrado es el texto que leyó, sin el HTML. En un correo sin HTML los
    enlaces se cuentan en ese texto, porque es lo que el evaluador ve.
    """
    global _RASGOS
    if _RASGOS is None:
        import pandas as pd
        ruta = BASE / "data" / "Datasets_Procesados" / "Dataset_Real.parquet"
        df = pd.read_parquet(ruta, columns=["email_id", "has_html", "num_images", "num_links"])
        _RASGOS = df.set_index("email_id").to_dict("index")
    r = _RASGOS[ejemplo["email_id"]]
    html_ = bool(r["has_html"])
    enlaces = (int(r["num_links"]) if html_
               else len(re.findall(r"hxxps?://\S+", ejemplo["cuerpo"])))
    return {"html": html_, "imagenes": int(r["num_images"]) if html_ else 0,
            "enlaces": enlaces}


def _nota_html(ejemplo: dict) -> str:
    """Aviso de que el original es HTML y la imagen muestra solo su texto."""
    p = _partes(ejemplo)
    if not p["html"]:
        return ""
    cosas = [f"{n} {nombre if n == 1 else plural}" for n, nombre, plural in
             ((p["imagenes"], "imagen", "imágenes"), (p["enlaces"], "enlace", "enlaces")) if n]
    detalle = f", con {' y '.join(cosas)}" if cosas else ""
    return (f"El correo original tiene formato HTML (diseño){detalle}; aquí se muestra "
            "solo su texto.")


def _peso(ejemplo: dict, modalidad: str) -> float:
    return _bloques(ejemplo)[modalidad]["total_pct"]


def _pregunta_de_modalidad(ide: str, n: int, ejemplo: dict, respuesta: str) -> dict:
    """Verdadero o falso sobre el diseño o los enlaces, según lo que tenga el correo.

    Si el correo no tiene HTML ni enlaces, preguntar si influyeron no mide la
    lectura del reporte; se pregunta entonces por el orden de dos palabras. El
    enunciado se redacta de modo que la respuesta correcta sea `respuesta`, para
    repartir las verdaderas y las falsas.
    """
    p = _partes(ejemplo)
    queremos_v = respuesta == "Verdadero"
    if p["html"] or p["enlaces"]:
        if p["html"]:
            influye = _peso(ejemplo, "estructura") > 0
            sujeto = ("El correo original tiene formato HTML. Según la explicación, su "
                      "diseño {}influyó en esta decisión.")
        else:
            influye = _peso(ejemplo, "red") > 0
            sujeto = ("Este correo contiene enlaces. Según la explicación, los enlaces y "
                      "la verificación del remitente {}influyeron en esta decisión.")
        titulo = sujeto.format("" if influye == queremos_v else "no ")
    else:
        mayor, menor = _par_para_comparar(ejemplo)
        a, b = (mayor, menor) if queremos_v else (menor, mayor)
        titulo = f"Según la explicación, la palabra «{a}» pesó más que «{b}»."
    return {"id": ide, "ejemplo": n, "tipo": "vf", "titulo": f"{ide}. {titulo}",
            "opciones": ["Verdadero", "Falso"], "correcta": respuesta}


def _pregunta_de_parte_dominante(ide: str, n: int, ejemplo: dict) -> dict:
    """Qué parte del correo pesó más, con las tres partes del reporte como opciones.

    Lee el reporte por modalidad, que es lo que pide el indicador. Solo se hace
    si el correo tiene diseño o enlaces; si no, la pregunta se responde sin leer.
    """
    p = _partes(ejemplo)
    if not (p["html"] or p["enlaces"]):
        return _pregunta_de_modalidad(ide, n, ejemplo, "Verdadero")
    sys.path.insert(0, str(BASE / "src"))
    from phishing_model.xai.narrative import NOMBRES_SENCILLOS
    dominante = max(("texto", "estructura", "red"), key=lambda m: _peso(ejemplo, m))
    opciones = [NOMBRES_SENCILLOS[m] for m in ("texto", "estructura", "red")]
    opciones.append("Las tres pesaron lo mismo")
    correcta = NOMBRES_SENCILLOS[dominante]
    return {"id": ide, "ejemplo": n, "tipo": "opcion",
            "titulo": f"{ide}. Según la explicación, ¿qué parte del correo pesó más en esta decisión?",
            "opciones": _barajar(opciones, ide, correcta), "correcta": correcta}


def _caracteristicas_no_citadas(ejemplo: dict, n: int) -> list[str]:
    """Características continuas que el reporte de este correo no cita.

    Las de autenticación (SPF, DKIM, DMARC) se excluyen: el reporte nunca puede
    citarlas, y ofrecerlas como distractor premiaría saberlo y no leer el reporte.
    """
    sys.path.insert(0, str(BASE / "src"))
    from phishing_model.xai.narrative import FEATURE_DISPLAY_NAMES
    citadas = {f["name"] for m in ("estructura", "red") for f in _bloques(ejemplo)[m]["factors"]
               if f["pct"] > 0}
    # Pares que un lector toma por la misma cosa: si el reporte cita uno, el otro
    # no puede ser una opción incorrecta.
    for grupo in ({"num_links", "num_urls_metadata"},):
        if citadas & grupo:
            citadas |= grupo
    candidatas = [v for k, v in FEATURE_DISPLAY_NAMES.items()
                  if k not in citadas and not k.endswith("_result")]
    return sorted(candidatas, key=lambda v: hashlib.sha256(f"C6{v}".encode()).hexdigest())[:n]


def _barajar(opciones: list[str], semilla: str, correcta: str) -> list[str]:
    """Orden determinista en el que la respuesta correcta nunca va primera.

    Quien duda tiende a marcar la primera opción; dejar ahí la correcta inflaría
    la comprensión medida.
    """
    orden = sorted(opciones, key=lambda o: hashlib.sha256(f"{semilla}{o}".encode()).hexdigest())
    if orden[0] == correcta:
        orden[0], orden[2] = orden[2], orden[0]
    return orden


def _banda_de_confianza(pct: float) -> str:
    if pct > 99:
        return "Más del 99%"
    if pct >= 90:
        return "Entre 90% y 99%"
    if pct >= 60:
        return "Entre 60% y 90%"
    return "Menos del 60%"


# --------------------------------------------------------------------------
# Preguntas de comprensión
# --------------------------------------------------------------------------

def preguntas_de_comprension(e1: dict, e2: dict, e3: dict, e4: dict) -> list[dict]:
    """Diez preguntas sobre cuatro correos: dos de phishing y dos legítimos.

    Cubren lo que un analista necesita leer en el reporte: qué palabras pesaron y
    en qué orden, qué decidió el sistema y con qué confianza, y qué tipo de
    información sustentó la decisión. Las de verdadero o falso no tienen todas la
    misma respuesta, porque quien marcara siempre la misma opción acertaría sin
    leer. Las que preguntan por el diseño o los enlaces dependen de que el correo
    los tenga (ver `_pregunta_de_modalidad`).

    No hay un correo de confianza intermedia: ninguno bien clasificado tiene una
    explicación legible (ver el informe de E9). La pregunta que el instrumento
    preveía para él, qué factor empuja hacia cada clase, tampoco podría
    responderse con este reporte, porque la atención no tiene signo.
    """
    p1, p2, p3, p4 = (_palabras(e) for e in (e1, e2, e3, e4))
    conf1 = e1["explicacion"]["estructurado"]["confidence_pct"]
    conf4 = e4["explicacion"]["estructurado"]["confidence_pct"]
    clase4 = "Legítimo" if e4["explicacion"]["clase_predicha"] == 0 else "Phishing"
    vf = ["Verdadero", "Falso"]

    no_citada = _distractores(e3, 1)[0]
    no_textuales = _no_textuales(e3)
    c8_correcta = f"{clase4}, con {_banda_de_confianza(conf4).lower()} de confianza"
    c8_ops = [f"{c}, con {b} de confianza" for c in ("Legítimo", "Phishing")
              for b in ("más del 99%", "entre 60% y 90%")]

    return [
        {"id": "C1", "ejemplo": 1, "tipo": "opcion",
         "titulo": (f"C1. Según la explicación, ¿qué palabra pesó más para decidir que "
                    f"este correo es {_clase(e1)}?"),
         "opciones": _barajar([p1[0], p1[1]] + _distractores(e1), "C1", p1[0]),
         "correcta": p1[0]},
        _pregunta_de_parte_dominante("C2", 1, e1),
        {"id": "C3", "ejemplo": 1, "tipo": "opcion",
         "titulo": "C3. Según la explicación, ¿qué tan seguro está el sistema de su decisión?",
         "opciones": ["Menos del 60%", "Entre 60% y 90%", "Entre 90% y 99%", "Más del 99%"],
         "correcta": _banda_de_confianza(conf1)},
        {"id": "C4", "ejemplo": 2, "tipo": "opcion",
         "titulo": (f"C4. Según la explicación, ¿qué palabra pesó más para decidir que "
                    f"este correo es {_clase(e2)}?"),
         "opciones": _barajar([p2[0], p2[1]] + _distractores(e2), "C4", p2[0]),
         "correcta": p2[0]},
        _pregunta_de_modalidad("C5", 2, e2, "Falso"),
        ({"id": "C6", "ejemplo": 3, "tipo": "opcion",
          "titulo": ("C6. Además del texto, ¿qué otra cosa influyó en esta decisión, "
                     "según la explicación?"),
          "opciones": _barajar([no_textuales[0]] + _caracteristicas_no_citadas(e3, 3),
                               "C6", no_textuales[0]),
          "correcta": no_textuales[0]}
         if no_textuales else
         {"id": "C6", "ejemplo": 3, "tipo": "opcion",
          "titulo": "C6. ¿Cuál de estas palabras NO aparece en la lista de la explicación?",
          "opciones": _barajar(p3[:3] + [no_citada], "C6", no_citada),
          "correcta": no_citada}),
        {"id": "C7", "ejemplo": 3, "tipo": "vf",
         "titulo": ("C7. Según la explicación, la palabra «{1}» pesó más que «{0}»."
                    .format(*_par_para_comparar(e3))),
         "opciones": vf, "correcta": "Falso"},
        {"id": "C8", "ejemplo": 4, "tipo": "opcion",
         "titulo": "C8. Según la explicación, ¿qué decidió el sistema sobre este correo?",
         "opciones": _barajar(c8_ops, "C8", c8_correcta), "correcta": c8_correcta},
        {"id": "C9", "ejemplo": 4, "tipo": "opcion",
         "titulo": ("C9. ¿Cuál de estas palabras está entre las tres que más pesaron, "
                    "según la explicación?"),
         "opciones": _barajar([p4[1]] + _distractores(e4, 3), "C9", p4[1]),
         "correcta": p4[1]},
        _pregunta_de_modalidad("C10", 4, e4, "Verdadero"),
    ]


# --------------------------------------------------------------------------
# Textos fijos
# --------------------------------------------------------------------------

PRESENTACION = (
    "Este formulario es parte de una tesis de Ingeniería Informática ({autor}, PUCP). "
    "Construimos un sistema que detecta correos de phishing (correos que intentan "
    "engañarte) y que explica por qué tomó cada decisión. Queremos saber si esas "
    "explicaciones se entienden.\n\n"
    "Verás un ejemplo resuelto y luego 4 correos, cada uno con su explicación. Por "
    "cada correo responderás 2 o 3 preguntas cortas; al final, dos cuestionarios de "
    "opinión. Toma unos 15 minutos.\n\n"
    "No necesitas saber de seguridad ni entender todo el inglés de los correos: las "
    "preguntas se responden mirando la explicación. No te evaluamos a ti, sino a la "
    "explicación.\n\n"
    "Es voluntario y anónimo. Los enlaces de los correos están desactivados; no "
    "intentes abrirlos.\n\n"
    "Consultas: {contacto}"
)

CONSENTIMIENTO = ("Acepto participar de forma voluntaria y anónima, y sé que puedo "
                  "dejar el formulario cuando quiera.")

COMO_LEER = (
    "Cada explicación tiene tres partes:\n"
    "1. La decisión: si el correo es phishing o legítimo, y qué tan seguro está el "
    "sistema (en %).\n"
    "2. En qué se basó: qué partes del correo pesaron más, en orden. Casi siempre es "
    "el texto; a veces también el diseño del correo o sus enlaces.\n"
    "3. La imagen del correo: las palabras más sombreadas son las que más pesaron.\n\n"
    "Abajo tienes un ejemplo ya explicado. No se evalúa."
)

INSTRUCCION_A = ("¿Qué te parecieron las explicaciones? Marca de 1 (totalmente en "
                 "desacuerdo) a 5 (totalmente de acuerdo). No hay respuestas correctas.")
INSTRUCCION_B = "Unas preguntas más de opinión, con la misma escala de 1 a 5."
INSTRUCCION_C = ("Ahora, 4 correos reales. Responde según lo que dice la explicación, "
                 "no según lo que tú opinas del correo.")


def _ejemplo_resuelto(ej: dict) -> str:
    """El ejemplo guiado contado en palabras, a partir de su propio reporte."""
    est = ej["explicacion"]["estructurado"]
    clase = "phishing" if est["label"] == 1 else "legítimo"
    texto = next(b for b in est["modality_blocks"] if b["modality"] == "texto")
    palabras = [f"«{f['name']}»" for f in texto["factors"][:3]]
    otros = [b for b in est["modality_blocks"] if b["modality"] != "texto" and b["total_pct"] > 0]
    sys.path.insert(0, str(BASE / "src"))
    from phishing_model.xai.narrative import NOMBRES_SENCILLOS
    cierre = ("El diseño del correo y sus enlaces no influyeron." if not otros else
              "También influyó, aunque poco: " + ", ".join(
                  f"{NOMBRES_SENCILLOS[b['modality']].lower()} ({b['total_pct']}%)"
                  for b in otros) + ".")
    return (f"En este ejemplo, el sistema decidió que el correo es {clase}, con "
            f"{est['confidence_pct']}% de confianza. Se basó casi solo en el texto: las "
            f"palabras que más pesaron fueron {', '.join(palabras[:-1])} y {palabras[-1]}, "
            f"que en la imagen aparecen con recuadro. {cierre}")


def _ejemplo_texto(ej: dict) -> str:
    return (f"Asunto: {ej['asunto']}\nRemitente: {ej['remitente']}\n\n"
            f"{_cuerpo_limpio(ej['cuerpo'])}")


DIRECCION = r"([\w.+-]+)(?:@|\(a\)|\[at\])([\w-]+(?:\.[\w-]+)+)"


def _desactivar_enlace(url: str, phishing: bool = True) -> str:
    """Enlace desactivado; la dirección que lleve dentro sigue la regla de su clase.

    Los puntos ya envueltos en corchetes no se vuelven a envolver.
    """
    url = re.sub(DIRECCION, (lambda m: "[usuario]@" + m.group(2)) if phishing
                 else (lambda m: "[correo oculto]"), url)
    url = re.sub(r"^http", "hxxp", url, flags=re.I)
    return re.sub(r"(?<!\[)\.(?!\])", "[.]", url)


def ocultar_para(phishing: bool) -> list:
    """Lo mismo que aplica la selección de correos, aplicado al texto de la imagen.

    En el phishing el dominio de cada dirección es evidencia y se conserva
    desactivado; en los legítimos la dirección es de una persona y se oculta.
    """
    if phishing:
        direccion = (DIRECCION,
                     lambda txt: "[usuario]@" + re.match(DIRECCION, txt).group(2)
                     .replace(".", "[.]"))
    else:
        direccion = (DIRECCION, lambda m: "[correo oculto]")
    return [
        (r"\bphishing@pot\b,?", lambda m: ""),
        # El enlace va antes que la dirección: una dirección dentro de un enlace
        # se resuelve en `_desactivar_enlace`, y el enlace no queda sin desactivar.
        (r"(?:https?://|www\.)\S+", lambda txt: _desactivar_enlace(txt, phishing)),
        direccion,
        (r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", lambda m: "[teléfono oculto]"),
    ]


RECORTAR_GUIA = False


def _tramo_repetido(texto: str) -> tuple[int, int, str] | None:
    """El bloque que un correo repite tal cual, para no mostrarlo dos veces.

    Se busca la segunda aparición del comienzo del correo; el tramo omitido va
    de ahí hasta el último enlace, que se conserva porque puede contener
    palabras que cita la explicación.
    """
    inicio = texto[:20]
    j = texto.find(inicio, 20)
    if j < 0:
        return None
    k = texto.rfind("http")
    fin = k if k > j else len(texto)
    return j, fin, "[… aquí el correo repite el mismo aviso …]"


def generar_imagenes(datos: dict) -> dict[str, str]:
    """Dibuja cada correo resaltado y devuelve el nombre de fichero de cada uno."""
    sys.path.insert(0, str(BASE / "src"))
    from phishing_model.xai.resaltado import dibujar_resaltado

    nombres = {}
    for clave, ej in datos["ejemplos"].items():
        citadas = [f["name"] for f in _bloques(ej)["texto"]["factors"]]
        nombre = f"resaltado_{clave}.png"
        # En la guía de la versión de claridad el correo repite su aviso; mostrarlo
        # entero alargaba la primera página sin enseñar nada nuevo.
        omitir = (_tramo_repetido(ej["texto_leido"][:ej["resaltado"]["leido_hasta"]])
                  if RECORTAR_GUIA and clave == "induccion" else None)
        dibujar_resaltado(ej["texto_leido"], ej["resaltado"], SALIDA / nombre,
                          citadas=citadas, omitir=omitir,
                          ocultar=ocultar_para(ej["etiqueta_real"].lower().startswith("phish")))
        nombres[clave] = nombre
    return nombres


def _reporte_texto(ej: dict) -> str:
    """La explicación que ve el evaluador: el reporte en lenguaje sencillo del módulo.

    El reporte técnico y su narrativa expresan los pesos en dos escalas distintas
    y usan términos que las primeras pruebas mostraron confusos; esta versión dice
    lo mismo con una sola escala (ver `narrative.reporte_sencillo`).
    """
    sys.path.insert(0, str(BASE / "src"))
    from phishing_model.xai.narrative import reporte_sencillo
    return reporte_sencillo(ej["explicacion"]["estructurado"])


# --------------------------------------------------------------------------
# Estructura común del formulario
# --------------------------------------------------------------------------

def estructura(datos: dict, imagenes: dict[str, str] | None = None) -> list[dict]:
    """El formulario como lista de elementos, que luego se traduce a cada salida.

    La comprensión va antes que las escalas de opinión: el SUS y el PSSUQ se
    diseñaron para responderse después de usar el sistema, y aquí usarlo es leer
    los cuatro reportes. Responderlos tras el ejemplo guiado mediría la opinión
    sobre un único reporte.
    """
    ej = datos["ejemplos"]
    correos = [ej[k] for k in CLAVES]
    ind = ej["induccion"]
    sus = _items("## 2. Sección A", "## 3. Sección B")
    # Del PSSUQ solo se aplica la subescala de Calidad de la Información (sus
    # ítems 7 a 12): tras la prueba piloto se acortó el formulario y se conservó
    # la subescala más pertinente para evaluar explicaciones.
    pssuq = _items("#### B.2", "#### B.3")
    if len(sus) != 10 or len(pssuq) != 6:
        raise SystemExit(f"Se esperaban 10 ítems SUS y 6 PSSUQ; hay {len(sus)} y {len(pssuq)}.")
    preguntas = preguntas_de_comprension(*correos)

    el = [
        {"tipo": "consentimiento", "titulo": "Consentimiento", "texto": CONSENTIMIENTO,
         "opciones": ["Acepto participar", "No acepto"]},
        {"tipo": "pagina", "titulo": "Sobre ti",
         "texto": "Estas preguntas solo sirven para describir al grupo de participantes."},
        {"tipo": "opcion", "id": "P1", "titulo": "P1. ¿Cuál es tu situación actual?",
         "opciones": ["Estudiante de pregrado", "Egresado o egresada",
                      "Profesional en ejercicio", "Otra"]},
        {"tipo": "opcion", "id": "P2", "titulo": "P2. ¿Cuál es tu carrera o área de formación?",
         "opciones": ["Ingeniería Informática", "Ingeniería de Sistemas",
                      "Ingeniería de Telecomunicaciones o Electrónica",
                      "Ciencias de la Computación", "Otra"]},
        {"tipo": "opcion", "id": "P3",
         "titulo": ("P3. ¿Tienes formación o experiencia en ciberseguridad (cursos, trabajo "
                    "en un centro de operaciones de seguridad, competencias tipo CTF)?"),
         "opciones": ["Sí", "No"]},
        {"tipo": "pagina", "titulo": "Cómo leer la explicación", "texto": COMO_LEER},
    ]
    imagenes = imagenes or {}

    def correo_y_reporte(ej: dict, clave: str, titulo: str) -> list[dict]:
        """El correo se muestra una sola vez: su imagen resaltada, con el asunto.

        Mostrarlo como texto y además en la imagen duplicaba la lectura (unas mil
        palabras en todo el formulario) sin añadir información: la imagen ya
        contiene el cuerpo que el modelo leyó.
        """
        if clave in imagenes:
            nota = _nota_html(ej)
            correo = {"tipo": "imagen", "titulo": titulo, "archivo": imagenes[clave],
                      "texto": f"Asunto: {ej['asunto']}" + (f"\n{nota}" if nota else "")}
        else:
            correo = {"tipo": "texto", "titulo": titulo, "texto": _ejemplo_texto(ej)}
        return [correo, {"tipo": "texto", "titulo": f"Explicación del sistema: {titulo.lower()}",
                         "texto": _reporte_texto(ej)}]

    el += correo_y_reporte(ind, "induccion", "Ejemplo resuelto")
    el.append({"tipo": "texto", "titulo": "Cómo se lee este ejemplo",
               "texto": _ejemplo_resuelto(ind)})
    for n, correo in enumerate(correos, 1):
        el.append({"tipo": "pagina", "titulo": f"Correo {n} de 4",
                   "texto": INSTRUCCION_C if n == 1 else ""})
        clave = CLAVES[n - 1]
        el += correo_y_reporte(correo, clave, f"Correo {n}")
        el += [dict(q, tipo="pregunta") for q in preguntas if q["ejemplo"] == n]
    el += [
        {"tipo": "pagina", "titulo": "Tu opinión (1 de 2)",
         "texto": INSTRUCCION_A},
        {"tipo": "rejilla", "id": "A", "titulo": "Grado de acuerdo (1 a 5)",
         "filas": [f"A{i}. {t}" for i, t in enumerate(sus, 1)]},
        {"tipo": "pagina", "titulo": "Tu opinión (2 de 2)",
         "texto": INSTRUCCION_B},
        {"tipo": "rejilla", "id": "B", "titulo": "Grado de acuerdo (1 a 5)",
         "filas": [f"B{i}. {t}" for i, t in enumerate(pssuq, 1)]},
        {"tipo": "pagina", "titulo": "Cierre",
         "texto": "Gracias por tu tiempo. Si quieres, deja un comentario sobre los reportes."},
        {"tipo": "parrafo", "id": "COMENTARIO", "titulo": "Comentarios (opcional)"},
    ]
    return el


# --------------------------------------------------------------------------
# Salidas
# --------------------------------------------------------------------------

def _js(texto: str) -> str:
    return json.dumps(texto, ensure_ascii=False)


def apps_script(el: list[dict]) -> str:
    l = [
        "/**",
        " * Crea el formulario del panel de R3.3 en Google Forms.",
        " * Uso: script.google.com > Nuevo proyecto > pegar este código > Ejecutar",
        " * la función crearFormulario y autorizar. El enlace para compartir y el de",
        " * edición aparecen en el registro de ejecución.",
        " * Generado por doc/validacion_xai/generar_formulario.py; no editar a mano.",
        " */",
        "function crearFormulario() {",
        f"  var form = FormApp.create({_js(TITULO)});",
        f"  form.setDescription({_js(PRESENTACION.format(autor=AUTOR, contacto=CONTACTO))});",
        "  form.setCollectEmail(false);",
        # El panel incluye evaluadores de fuera de la universidad. En algunas
        # cuentas institucionales la llamada falla; entonces se abre a mano desde
        # «Publicado», «Vista de encuestado», «Cualquier persona con el enlace».
        "  try { form.setRequireLogin(false); } catch (e) { Logger.log('Abrir a mano: ' + e); }",
        "  form.setProgressBar(true);",
        "  form.setShowLinkToRespondAgain(false);",
        "  form.setPublishingSummary(false);",
        "  form.setConfirmationMessage('Gracias por participar. Tus respuestas se registraron.');",
        "  var item;",
    ]
    for e in el:
        t = e["tipo"]
        if t == "consentimiento":
            l += [
                "  item = form.addMultipleChoiceItem();",
                f"  item.setTitle({_js(e['titulo'])}).setHelpText({_js(e['texto'])}).setRequired(true);",
                f"  item.setChoices([item.createChoice({_js(e['opciones'][0])}, FormApp.PageNavigationType.CONTINUE),",
                f"                   item.createChoice({_js(e['opciones'][1])}, FormApp.PageNavigationType.SUBMIT)]);",
            ]
        elif t == "pagina":
            l.append(f"  form.addPageBreakItem().setTitle({_js(e['titulo'])})"
                     + (f".setHelpText({_js(e['texto'])});" if e["texto"] else ";"))
        elif t == "texto":
            l.append(f"  form.addSectionHeaderItem().setTitle({_js(e['titulo'])})"
                     f".setHelpText({_js(e['texto'])});")
        elif t in ("opcion", "pregunta"):
            ops = ", ".join(_js(o) for o in e["opciones"])
            l += ["  item = form.addMultipleChoiceItem();",
                  f"  item.setTitle({_js(e['titulo'])}).setChoiceValues([{ops}]).setRequired(true);"]
        elif t == "rejilla":
            filas = ", ".join(_js(f) for f in e["filas"])
            l += ["  item = form.addGridItem();",
                  f"  item.setTitle({_js(e['titulo'])}).setRows([{filas}])"
                  f".setColumns({json.dumps(ESCALA)}).setRequired(true);"]
        elif t == "parrafo":
            l.append(f"  form.addParagraphTextItem().setTitle({_js(e['titulo'])});")
        elif t == "imagen":
            l.append(f"  form.addImageItem().setTitle({_js(e['titulo'])})"
                     + (f".setHelpText({_js(e['texto'])})" if e.get("texto") else "")
                     + f".setImage(UrlFetchApp.fetch({_js(URL_IMAGENES + e['archivo'])}));")
    l += [
        "  Logger.log('Enlace para responder: ' + form.getPublishedUrl());",
        "  Logger.log('Enlace para editar: ' + form.getEditUrl());",
        "}",
        "",
    ]
    return "\n".join(l)


def vista_previa(el: list[dict]) -> str:
    h = lambda s: html.escape(s).replace("\n", "<br>")  # noqa: E731
    partes = [f"<h1>{h(TITULO)}</h1>",
              f"<p class='desc'>{h(PRESENTACION.format(autor=AUTOR, contacto=CONTACTO))}</p>"]
    for e in el:
        t = e["tipo"]
        if t == "pagina":
            partes.append(f"<h2>{h(e['titulo'])}</h2>")
            if e["texto"]:
                partes.append(f"<p class='desc'>{h(e['texto'])}</p>")
        elif t == "texto":
            partes.append(f"<h3>{h(e['titulo'])}</h3><pre>{html.escape(e['texto'])}</pre>")
        elif t in ("opcion", "pregunta", "consentimiento"):
            titulo = e["titulo"] if t != "consentimiento" else e["texto"]
            ops = "".join(f"<li>&#9711; {h(o)}</li>" for o in e["opciones"])
            partes.append(f"<div class='q'><p><b>{h(titulo)}</b></p><ul>{ops}</ul></div>")
        elif t == "rejilla":
            cab = "".join(f"<th>{c.split()[0]}</th>" for c in ESCALA)
            filas = "".join(f"<tr><td>{h(f)}</td>" + "<td>&#9711;</td>" * 5 + "</tr>"
                            for f in e["filas"])
            partes.append("<p class='desc'>1 = totalmente en desacuerdo; 5 = totalmente de "
                          f"acuerdo.</p><table><tr><th></th>{cab}</tr>{filas}</table>")
        elif t == "parrafo":
            partes.append(f"<div class='q'><p><b>{h(e['titulo'])}</b></p>"
                          "<div class='caja'></div></div>")
        elif t == "imagen":
            partes.append(f"<h3>{h(e['titulo'])}</h3>"
                          + (f"<p class='desc'>{h(e['texto'])}</p>" if e.get("texto") else "")
                          + f"<img src='{e['archivo']}' style='width:100%'>")
    estilo = """
    body{font-family:Georgia,serif;max-width:820px;margin:32px auto;padding:0 16px;color:#111}
    h1{font-size:24px} h2{font-size:19px;margin-top:36px;border-top:1px solid #999;padding-top:16px}
    h3{font-size:15px;margin-bottom:4px} .desc{line-height:1.5}
    pre{white-space:pre-wrap;background:#f4f4f4;border:1px solid #ddd;padding:12px;font-size:13px}
    table{border-collapse:collapse;width:100%;font-size:13px} td,th{border:1px solid #ccc;padding:6px}
    td:not(:first-child),th{text-align:center;width:36px} ul{list-style:none;padding-left:8px}
    .q{margin:14px 0} .caja{border:1px solid #999;height:80px}
    @media print{h2{page-break-before:always}}"""
    return ("<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>"
            f"<title>Formulario del panel</title><style>{estilo}</style></head><body>"
            + "\n".join(partes) + "</body></html>")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Genera el formulario del panel de R3.3.")
    ap.add_argument("--version", choices=sorted(VERSIONES), default="principal")
    _usar_version(ap.parse_args().version)
    datos = json.loads(EJEMPLOS.read_text(encoding="utf-8"))
    SALIDA.mkdir(exist_ok=True)
    el = estructura(datos, generar_imagenes(datos))
    (SALIDA / "crear_formulario.gs").write_text(apps_script(el), encoding="utf-8")
    (SALIDA / "vista_previa.html").write_text(vista_previa(el), encoding="utf-8")
    clave = {
        "origen": str(EJEMPLOS.relative_to(BASE)).replace("\\", "/"),
        "ejemplos": {str(n): datos["ejemplos"][k]["email_id"]
                     for n, k in enumerate(CLAVES, 1)},
        "preguntas": {e["id"]: {"titulo": e["titulo"], "correcta": e["correcta"],
                                "opciones": e["opciones"]}
                      for e in el if e["tipo"] == "pregunta"},
    }
    (SALIDA / "clave_de_respuestas.json").write_text(
        json.dumps(clave, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in clave["preguntas"].items():
        print(f"  {k}: correcta «{v['correcta']}» entre {v['opciones']}")
    print(f"  escritos en {SALIDA.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
