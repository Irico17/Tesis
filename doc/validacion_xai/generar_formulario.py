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
    información sustentó la decisión. Las de verdadero o falso se reparten dos y
    dos, porque quien marcara siempre la misma opción acertaría sin leer.

    No hay un correo de confianza intermedia: ninguno bien clasificado tiene una
    explicación legible (ver el informe de E9). La pregunta que el instrumento
    preveía para él, qué factor empuja hacia cada clase, tampoco podría
    responderse con este reporte, porque la atención no tiene signo.
    """
    p1, p2, p3, p4 = (_palabras(e) for e in (e1, e2, e3, e4))
    b1, b2, b4 = _bloques(e1), _bloques(e2), _bloques(e4)
    conf1 = e1["explicacion"]["estructurado"]["confidence_pct"]
    conf4 = e4["explicacion"]["estructurado"]["confidence_pct"]
    clase4 = "Legítimo" if e4["explicacion"]["clase_predicha"] == 0 else "Phishing"
    vf = ["Verdadero", "Falso"]

    estructura_pesa_mas = b1["estructura"]["total_pct"] > b1["texto"]["total_pct"]
    texto_domina = b2["texto"]["total_pct"] >= max(b2["estructura"]["total_pct"],
                                                   b2["red"]["total_pct"])
    sin_estructura = b4["estructura"]["total_pct"] < 1
    no_citada = _distractores(e3, 1)[0]
    c8_correcta = f"{clase4}, con {_banda_de_confianza(conf4).lower()} de confianza"
    c8_ops = [f"{c}, con {b} de confianza" for c in ("Legítimo", "Phishing")
              for b in ("más del 99%", "entre 60% y 90%")]

    return [
        {"id": "C1", "ejemplo": 1, "tipo": "opcion",
         "titulo": ("C1. Según el reporte, ¿qué palabra tuvo mayor peso en la "
                    "clasificación de este correo como phishing?"),
         "opciones": _barajar([p1[0], p1[1]] + _distractores(e1), "C1", p1[0]),
         "correcta": p1[0]},
        {"id": "C2", "ejemplo": 1, "tipo": "vf",
         "titulo": ("C2. Según el reporte, la estructura HTML del correo pesó más que "
                    "su contenido textual en esta decisión."),
         "opciones": vf, "correcta": "Verdadero" if estructura_pesa_mas else "Falso"},
        {"id": "C3", "ejemplo": 1, "tipo": "opcion",
         "titulo": ("C3. Según el reporte, ¿con qué nivel de confianza clasificó el "
                    "sistema este correo?"),
         "opciones": ["Menos del 60%", "Entre 60% y 90%", "Entre 90% y 99%", "Más del 99%"],
         "correcta": _banda_de_confianza(conf1)},
        {"id": "C4", "ejemplo": 2, "tipo": "opcion",
         "titulo": ("C4. Según el reporte, ¿qué palabra contribuyó más a que el sistema "
                    "considerara legítimo este correo?"),
         "opciones": _barajar([p2[0], p2[1]] + _distractores(e2), "C4", p2[0]),
         "correcta": p2[0]},
        {"id": "C5", "ejemplo": 2, "tipo": "vf",
         "titulo": ("C5. Según el reporte, la decisión sobre este correo se apoyó "
                    "principalmente en su contenido textual."),
         "opciones": vf, "correcta": "Verdadero" if texto_domina else "Falso"},
        {"id": "C6", "ejemplo": 3, "tipo": "opcion",
         "titulo": ("C6. ¿Cuál de estas palabras NO figura entre las que el reporte "
                    "señala como influyentes?"),
         "opciones": _barajar(p3[:3] + [no_citada], "C6", no_citada),
         "correcta": no_citada},
        {"id": "C7", "ejemplo": 3, "tipo": "vf",
         "titulo": (f"C7. Según el reporte, la palabra «{p3[2]}» tuvo más peso que la "
                    f"palabra «{p3[0]}»."),
         "opciones": vf, "correcta": "Falso"},
        {"id": "C8", "ejemplo": 4, "tipo": "opcion",
         "titulo": "C8. Según el reporte, ¿qué decidió el sistema sobre este correo?",
         "opciones": _barajar(c8_ops, "C8", c8_correcta), "correcta": c8_correcta},
        {"id": "C9", "ejemplo": 4, "tipo": "opcion",
         "titulo": ("C9. ¿Cuál de estas palabras está entre las tres que, según el "
                    "reporte, más influyeron en la decisión?"),
         "opciones": _barajar([p4[1]] + _distractores(e4, 3), "C9", p4[1]),
         "correcta": p4[1]},
        {"id": "C10", "ejemplo": 4, "tipo": "vf",
         "titulo": ("C10. Según el reporte, la estructura HTML del correo no aportó "
                    "peso a esta decisión."),
         "opciones": vf, "correcta": "Verdadero" if sin_estructura else "Falso"},
    ]


# --------------------------------------------------------------------------
# Textos fijos
# --------------------------------------------------------------------------

PRESENTACION = (
    "Este formulario forma parte de la tesis de Ingeniería Informática de {autor} "
    "(PUCP), sobre detección de phishing en correo electrónico con inteligencia "
    "artificial explicable. El sistema clasifica cada correo como phishing o legítimo "
    "y genera un reporte que explica qué factores pesaron en esa decisión. Queremos "
    "saber si ese reporte se entiende.\n\n"
    "Tomará unos 25 minutos. Verás un ejemplo guiado, leerás cuatro correos reales con "
    "su reporte y contestarás diez preguntas sobre lo que el reporte dice y, al final, "
    "responderás dos cuestionarios breves de opinión. Las preguntas de comprensión "
    "tienen respuestas correctas, pero no te evalúan a ti: miden si el reporte "
    "comunica bien. Las de opinión no tienen respuestas correctas.\n\n"
    "La participación es voluntaria y anónima. No se registra tu nombre ni tu correo, "
    "y las respuestas se analizan de forma agregada. Puedes retirarte en cualquier "
    "momento cerrando el formulario.\n\n"
    "Seguridad: los correos son ejemplos reales de phishing. Sus enlaces están "
    "desactivados (hxxp, [.]) y se han ocultado direcciones y teléfonos. No intentes "
    "abrir ningún enlace.\n\n"
    "Consultas: {contacto}"
)

CONSENTIMIENTO = ("Confirmo que participo de forma voluntaria, que puedo retirarme en "
                  "cualquier momento y que mis respuestas se usarán de forma anónima y "
                  "agregada con fines de investigación.")

COMO_LEER = (
    "Antes de empezar, este es un ejemplo de cómo leer el reporte. No forma parte de "
    "las preguntas.\n\n"
    "1. La primera línea indica la decisión del sistema (phishing o legítimo) y su "
    "confianza, entre 0% y 100%.\n"
    "2. Después, el reporte se divide en tres bloques, uno por cada tipo de "
    "información del correo: su contenido textual, su estructura HTML y sus "
    "metadatos de red y autenticación. Cada bloque indica qué parte del peso total "
    "de la decisión le corresponde.\n"
    "3. Dentro de cada bloque, los factores aparecen ordenados de mayor a menor peso. "
    "En el bloque de texto, los factores son palabras del correo; un porcentaje mayor "
    "significa que esa palabra influyó más. A veces el sistema cita como palabra un "
    "fragmento de un enlace o un signo de puntuación: se lee igual, como algo que el "
    "sistema tuvo en cuenta.\n"
    "4. La síntesis final resume qué tipo de información sustentó la decisión.\n\n"
    "El párrafo inicial (narrativa) dice lo mismo en una frase: la decisión, la "
    "confianza y las tres palabras más influyentes."
)

INSTRUCCION_A = ("Piensa en los reportes de explicación que leíste. Para cada "
                 "afirmación, indica tu grado de acuerdo: 1 = totalmente en desacuerdo, "
                 "5 = totalmente de acuerdo. No hay respuestas correctas o incorrectas.")
INSTRUCCION_B = ("Las siguientes afirmaciones tratan sobre la utilidad del reporte, la "
                 "claridad de su información y su presentación. Usa la misma escala de "
                 "1 a 5.")
INSTRUCCION_C = ("Ahora verás cuatro correos reales ya clasificados por el sistema, cada uno "
                 "con su reporte. Léelos con calma y responde basándote únicamente en lo "
                 "que dice el reporte.")


def _ejemplo_texto(ej: dict) -> str:
    return (f"Asunto: {ej['asunto']}\nRemitente: {ej['remitente']}\n\n"
            f"{_cuerpo_limpio(ej['cuerpo'])}")


def _reporte_texto(ej: dict) -> str:
    exp = ej["explicacion"]
    return f"{exp['narrativa']}\n\n{exp['reporte_por_modalidad'].strip()}"


# --------------------------------------------------------------------------
# Estructura común del formulario
# --------------------------------------------------------------------------

def estructura(datos: dict) -> list[dict]:
    """El formulario como lista de elementos, que luego se traduce a cada salida.

    La comprensión va antes que las escalas de opinión: el SUS y el PSSUQ se
    diseñaron para responderse después de usar el sistema, y aquí usarlo es leer
    los cuatro reportes. Responderlos tras el ejemplo guiado mediría la opinión
    sobre un único reporte.
    """
    ej = datos["ejemplos"]
    correos = [ej["phishing_1"], ej["legitimo_1"], ej["phishing_2"], ej["legitimo_2"]]
    ind = ej["induccion"]
    sus = _items("## 2. Sección A", "## 3. Sección B")
    pssuq = _items("## 3. Sección B", "## 4. Sección C")
    if len(sus) != 10 or len(pssuq) != 16:
        raise SystemExit(f"Se esperaban 10 ítems SUS y 16 PSSUQ; hay {len(sus)} y {len(pssuq)}.")
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
        {"tipo": "pagina", "titulo": "Cómo leer el reporte", "texto": COMO_LEER},
        {"tipo": "texto", "titulo": "Ejemplo guiado: el correo", "texto": _ejemplo_texto(ind)},
        {"tipo": "texto", "titulo": "Ejemplo guiado: el reporte del sistema",
         "texto": _reporte_texto(ind)},
    ]
    for n, correo in enumerate(correos, 1):
        el += [
            {"tipo": "pagina", "titulo": f"Sección C. Comprensión: correo {n} de 4",
             "texto": INSTRUCCION_C if n == 1 else ""},
            {"tipo": "texto", "titulo": f"Correo {n}", "texto": _ejemplo_texto(correo)},
            {"tipo": "texto", "titulo": f"Reporte del sistema para el correo {n}",
             "texto": _reporte_texto(correo)},
        ]
        el += [dict(q, tipo="pregunta") for q in preguntas if q["ejemplo"] == n]
    el += [
        {"tipo": "pagina", "titulo": "Sección A. Opinión sobre los reportes (1 de 2)",
         "texto": INSTRUCCION_A},
        {"tipo": "rejilla", "id": "A", "titulo": "Grado de acuerdo (1 a 5)",
         "filas": [f"A{i}. {t}" for i, t in enumerate(sus, 1)]},
        {"tipo": "pagina", "titulo": "Sección B. Opinión sobre los reportes (2 de 2)",
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
    datos = json.loads(EJEMPLOS.read_text(encoding="utf-8"))
    el = estructura(datos)
    SALIDA.mkdir(exist_ok=True)
    (SALIDA / "crear_formulario.gs").write_text(apps_script(el), encoding="utf-8")
    (SALIDA / "vista_previa.html").write_text(vista_previa(el), encoding="utf-8")
    clave = {
        "origen": "medios_de_verificacion/experimentos/e9/ejemplos_del_panel.json",
        "ejemplos": {str(n): datos["ejemplos"][k]["email_id"] for n, k in enumerate(
            ("phishing_1", "legitimo_1", "phishing_2", "legitimo_2"), 1)},
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
