"""
E9 · Ejemplos reales para el panel de analistas (R3.3)

El criterio primario de R3.3 es un panel humano que lee explicaciones reales y
responde si las entiende. Este guion prepara los correos que ese panel revisa, con
la explicación que el sistema genera para cada uno, y nada más: no simula
evaluadores ni produce respuestas.

**Qué se elige y por qué.** El instrumento (`doc/validacion_xai/`) usa cuatro
correos en la sección de comprensión, dos de phishing y dos legítimos, todos
correctamente clasificados para que la pregunta sea si se entiende la explicación
y no si el modelo acertó, y un quinto, de phishing, que solo se usa en la
inducción para enseñar a leer el reporte sin adelantar las respuestas. Con cuatro
correos y diez preguntas el porcentaje de comprensión avanza de diez en diez y no
depende de un solo mensaje. Dentro de cada clase, el segundo correo se toma de una
colección distinta del primero cuando la hay, para no evaluar un solo estilo de
mensaje. Se busca además un correo de confianza intermedia; el modelo casi nunca
duda y el informe registra si lo hubo.

Los candidatos salen del conjunto de prueba de la corrida reportada, con las
predicciones ya guardadas, y se filtran por criterios fijos: idioma inglés o
español, un cuerpo legible en pantalla, y que las palabras que la explicación cita
sean palabras con contenido. El último filtro no es elegir lo favorable: el
instrumento mide si el evaluador entiende el reporte, y uno que cite «the» o una
frontera MIME mediría la calidad del corpus. Cuántos candidatos pasa cada filtro
queda en el informe, porque esa proporción es parte del resultado.

**Seguridad del panel.** Los correos de phishing contienen enlaces reales. Antes
de mostrarlos se desactivan (``hxxp``, ``[.]``) y se ocultan direcciones de correo
y teléfonos, de modo que ningún evaluador pueda abrir un enlace malicioso ni ver
datos de terceros.

    python scripts/experimentos/e9_ejemplos_del_panel.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import BASE, SALIDA, cargar_corpus  # noqa: E402

sys.path.insert(0, str(BASE / "src"))

from phishing_model.explicabilidad import carga  # noqa: E402

import e9_narrativa_y_validacion as e9  # noqa: E402

LARGO_MIN, LARGO_MAX = 250, 1600       # caracteres del cuerpo mostrado
CANDIDATOS_POR_GRUPO = 300             # cuántos se explican antes de elegir
VACIAS = set("""
a an and are as at be been but by can do for from had has have he her his i if in
into is it its me my no not of on or our she so than that the their them then there
these they this to us was we were what when which who will with you your yours
de del el la las los un una unos unas y o u e en con por para que se su sus al lo
le les es son fue ser como mas más pero sin sobre este esta estos estas ese esa
""".split())


def _huella(email_id: str) -> str:
    """Orden determinista y no ligado a la posición en el fichero."""
    return hashlib.sha256(f"panel-{email_id}".encode()).hexdigest()


def _idioma(texto: str) -> str | None:
    """Idioma del cuerpo, solo si todo el texto está en un mismo idioma.

    Un correo que mezcla idiomas (un aviso en inglés con el pie en neerlandés)
    se detecta como inglés, pero sus palabras decisivas pueden no serlo, y el
    evaluador no podría leerlas.
    """
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0
    # Se detecta por tramos y no sobre el conjunto: un pie de página en otro
    # idioma queda en minoría en el cuerpo entero y la detección global lo ignora.
    tramos = [t for t in re.split(r"(?<=[.!?])\s+|\n{2,}", texto[:3000])
              if len(re.findall(r"[^\W\d_]", t)) >= 60]
    idiomas = set()
    for tramo in tramos or [texto[:2000]]:
        try:
            mejor = detect_langs(tramo)[0]
        except Exception:
            return None
        if mejor.prob < 0.9:
            return None
        idiomas.add(mejor.lang)
    return idiomas.pop() if len(idiomas) == 1 else None


def _desactivar(texto: str) -> str:
    """Enlaces sin activar, y sin direcciones de correo ni teléfonos."""
    texto = re.sub(r"[\w.+-]+(?:@|\(a\)|\[at\])[\w-]+(?:\.[\w-]+)+", "[correo oculto]", texto)
    # La colección phishing_pot reescribe el destinatario como «phishing@pot»,
    # que delataría la clase del correo al evaluador.
    texto = re.sub(r"\bphishing@pot\b,?\s*", "", texto, flags=re.I)
    texto = re.sub(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)", "[teléfono oculto]", texto)

    def enlace(m: re.Match) -> str:
        url = m.group(0)
        url = re.sub(r"^http", "hxxp", url, flags=re.I)
        return url.replace(".", "[.]")
    texto = re.sub(r"(?:https?://|www\.)\S+", enlace, texto, flags=re.I)
    return texto


def _cuerpo_visible(fila: pd.Series) -> str:
    texto = fila.get("body_plain") or fila.get("clean_text") or ""
    texto = re.sub(r"\r\n?", "\n", str(texto))
    texto = re.sub(r"\n{3,}", "\n\n", texto).strip()
    return texto


def _recortar(texto: str) -> str:
    if len(texto) <= LARGO_MAX:
        return texto
    corte = texto.rfind(" ", 0, LARGO_MAX)
    return texto[:corte if corte > 0 else LARGO_MAX].rstrip() + " [...]"


def _con_contenido(palabra: str) -> bool:
    return (e9._parece_palabra(palabra) and len(palabra) >= 3
            and palabra.lower() not in VACIAS and palabra.isalpha())


def _apta(explicacion: dict) -> bool:
    """La explicación se puede leer con el mismo criterio de E9.

    E9 cuenta como legible la explicación cuyas tres palabras decisivas son
    legibles; de ahí salen los 438 candidatos que el capítulo informa. Aquí se
    exige eso mismo, que no se repitan, y que la principal sea una palabra con
    contenido, porque la sección de comprensión pregunta por ella. Exigirlo a las
    cinco palabras del bloque dejaba uno de cada ciento cincuenta candidatos: la
    atención cita a menudo signos de puntuación o cifras, que es la limitación de
    legibilidad que R3.2 ya declara.
    """
    bloque = next((b for b in explicacion["estructurado"]["modality_blocks"]
                   if b["modality"] == "texto"), None)
    if not bloque or len(bloque["factors"]) < 3:
        return False
    nombres = [f["name"].lower() for f in bloque["factors"]]
    if len(set(nombres)) < len(nombres):
        return False
    decisivas = nombres[:3]
    return all(e9._parece_palabra(n) for n in decisivas) and _con_contenido(decisivas[0])

def _es_nombre_propio(palabra: str, cuerpo: str) -> bool:
    """La palabra solo aparece en el cuerpo con mayúscula inicial.

    En la correspondencia legítima de las listas técnicas, esas palabras son casi
    siempre nombres de quienes escriben: no informan al lector y son datos de
    terceros. En el phishing, en cambio, la marca suplantada es evidencia, de
    modo que el filtro solo se aplica al grupo legítimo.
    """
    apariciones = re.findall(r"\b" + re.escape(palabra) + r"\b", cuerpo, flags=re.I)
    return bool(apariciones) and all(a[:1].isupper() and not a.isupper() for a in apariciones)


def main() -> int:
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"E9 · ejemplos del panel (R3.3) en {dispositivo}")

    corpus = cargar_corpus()
    corrida = carga.CORRIDA_DE_REFERENCIA
    ruta = sorted((BASE / "data" / "predictions").glob(f"{corrida}*_preds.parquet"))[0]
    pred = pd.read_parquet(ruta)[["email_id", "y_true", "y_pred", "y_proba"]]
    prueba = corpus.merge(pred, on="email_id")
    prueba = prueba[prueba["y_true"] == prueba["y_pred"]].copy()
    prueba["cuerpo"] = prueba.apply(_cuerpo_visible, axis=1)
    prueba["largo"] = prueba["cuerpo"].str.len()
    prueba["confianza"] = prueba["y_proba"].where(prueba["y_pred"] == 1, 1 - prueba["y_proba"])

    conteo = {"prueba_bien_clasificados": int(len(prueba))}
    legible = prueba[prueba["largo"].between(LARGO_MIN, LARGO_MAX)
                     & prueba["subject"].fillna("").str.strip().astype(bool)].copy()
    conteo["con_cuerpo_legible"] = int(len(legible))
    legible["idioma"] = legible["cuerpo"].map(_idioma)
    legible = legible[legible["idioma"].isin(["en", "es"])].copy()
    conteo["en_ingles_o_espanol"] = int(len(legible))
    legible["orden"] = legible["email_id"].map(_huella)

    # El modelo apenas duda: de los bien clasificados, solo una decena tiene una
    # confianza entre 0.5 y 0.95. Para ese grupo no se exige un cuerpo corto, que
    # los dejaba en tres; el cuerpo se recorta al mostrarlo.
    dudosos = prueba[prueba["confianza"].between(0.5, 0.95)].copy()
    dudosos["idioma"] = dudosos["cuerpo"].map(_idioma)
    dudosos = dudosos[dudosos["idioma"].isin(["en", "es"])
                      & dudosos["subject"].fillna("").str.strip().astype(bool)].copy()
    dudosos["orden"] = dudosos["email_id"].map(_huella)
    conteo["dudosos_bien_clasificados_en_ingles_o_espanol"] = int(len(dudosos))

    grupos = {
        "phishing_alta_confianza": legible[(legible["y_pred"] == 1) & (legible["confianza"] >= 0.99)],
        "legitimo": legible[(legible["y_pred"] == 0) & (legible["confianza"] >= 0.99)],
        "confianza_intermedia": dudosos,
    }
    cargado = carga.cargar(BASE, corrida, dispositivo=dispositivo)
    tecnica = e9._tecnica_mas_fiel()

    elegidos, informe_grupos = {}, {}
    for nombre, g in grupos.items():
        g = g.sort_values("orden").head(CANDIDATOS_POR_GRUPO)
        lotes = carga.filas_como_lotes(cargado, g, dispositivo=dispositivo)
        aptos = []
        for (_, fila), lote in zip(g.iterrows(), lotes):
            exp = e9._explicar(cargado, lote, tecnica)
            if exp["clase_predicha"] != int(fila["y_pred"]) or not _apta(exp):
                continue
            # Las palabras decisivas tienen que verse en el correo tal como se
            # muestra. Si proceden de una dirección que se oculta, el evaluador no
            # podría encontrarlas y la pregunta mediría otra cosa.
            visible = (str(fila["subject"]) + " " + _desactivar(_recortar(fila["cuerpo"]))).lower()
            tres = [f["name"] for b in exp["estructurado"]["modality_blocks"]
                    if b["modality"] == "texto" for f in b["factors"][:3]]
            if not all(w.lower() in visible for w in tres):
                continue
            if nombre == "legitimo":
                decisivas = [f["name"] for b in exp["estructurado"]["modality_blocks"]
                             if b["modality"] == "texto" for f in b["factors"][:3]]
                if any(_es_nombre_propio(w, _desactivar(fila["cuerpo"])) for w in decisivas):
                    continue
            aptos.append((fila, exp))
        informe_grupos[nombre] = {"candidatos": int(len(grupos[nombre])),
                                  "explicados": int(len(g)), "aptos": len(aptos)}
        print(f"  {nombre}: {len(grupos[nombre])} candidatos, {len(g)} explicados, "
              f"{len(aptos)} con explicación legible")
        claves = {"phishing_alta_confianza": ["phishing_1", "induccion", "phishing_2"],
                  "legitimo": ["legitimo_1", "legitimo_2"],
                  "confianza_intermedia": ["confianza_intermedia"]}[nombre]
        if len(aptos) < len(claves):
            # El de confianza intermedia es opcional en el instrumento; el resto, no.
            if nombre == "confianza_intermedia":
                print("  sin ejemplo de confianza intermedia legible")
                continue
            raise SystemExit(f"No hay ejemplos aptos suficientes para {nombre}.")
        # El último de cada clase, de una colección que aún no esté representada.
        orden = aptos[:len(claves) - 1]
        vistas = {f["source_dataset"] for f, _ in orden}
        resto = aptos[len(claves) - 1:]
        orden.append(next(((f, e) for f, e in resto if f["source_dataset"] not in vistas),
                          resto[0]))
        for clave, (fila, exp) in zip(claves, orden):
            elegidos[clave] = {
                "email_id": fila["email_id"],
                "coleccion": fila["source_dataset"],
                "etiqueta_real": fila["label_text"],
                "idioma": fila["idioma"],
                "asunto": _desactivar(str(fila["subject"]).strip()),
                "remitente": "[remitente oculto]",
                "cuerpo": _desactivar(_recortar(fila["cuerpo"])),
                "explicacion": exp,
            }

    informe = {
        "experimento": "e9_ejemplos_del_panel",
        "resultado": "R3.3",
        "corrida_explicada": corrida,
        "tecnica_de_texto": tecnica,
        "criterios": {
            "bien_clasificados": True,
            "largo_del_cuerpo": [LARGO_MIN, LARGO_MAX],
            "idiomas": ["en", "es"],
            "alta_confianza": ">= 0.99",
            "confianza_intermedia": ("entre 0.5 y 0.95, sin límite de largo; el cuerpo "
                                     "se recorta al mostrarlo"),
            "palabras_citadas": ("las tres decisivas, legibles (criterio de E9); la "
                                 "principal, alfabética, de 3 o más letras y no vacía; "
                                 "ninguna repetida en el bloque; las tres visibles en "
                                 "el correo mostrado; en el grupo legítimo, ninguna "
                                 "decisiva que sea un nombre propio"),
            "idioma": "un solo idioma, detectado por tramos con probabilidad >= 0.9",
            "orden": "huella SHA-256 del identificador, determinista",
            "seguridad": "enlaces desactivados; correos y teléfonos ocultos",
        },
        "conteo": conteo,
        "grupos": informe_grupos,
        "ejemplos": elegidos,
    }
    destino = SALIDA / "e9" / "ejemplos_del_panel.json"
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
    print(f"  escrito {destino.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
