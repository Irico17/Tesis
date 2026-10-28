"""
Resaltado del correo según el peso de cada palabra en la decisión (R3.2).

El reporte del módulo de narrativa cita las cinco palabras con más peso, pero el
analista lee el correo entero. Este módulo devuelve el mismo correo con cada
palabra sombreada en proporción a su peso, para que la evidencia se vea en su
contexto: dónde está la palabra decisiva y qué la rodea.

El peso es el mismo que usa el reporte: la propagación de la atención a través de
las capas del codificador de texto (Abnar y Zuidema, 2020), leída desde el token
de clasificación, y agrupada por palabra sumando sus fragmentos. Se calcula sobre
el texto que el modelo leyó (`clean_text`, truncado a la longitud máxima de
entrada), y las posiciones de cada token en ese texto se recuperan del propio
tokenizador, de modo que el sombreado cae exactamente sobre lo que el modelo vio.
Lo que quedó fuera por el truncamiento se indica como no leído.

No cambia ninguna predicción ni ninguna atribución: es otra forma de presentar el
mismo vector que la narrativa traduce a texto.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable


from phishing_model.xai.attention_explainer import extract_text_saliency_rollout


def pesos_por_palabra(cargado, texto: str) -> dict:
    """Peso de atención de cada palabra del texto, con su posición en caracteres.

    `cargado` es el modelo que devuelve `explicabilidad.carga.cargar`. Devuelve
    las palabras en el orden del texto como `[inicio, fin, peso]`, el carácter
    hasta el que llegó la lectura del modelo y si hubo truncamiento.
    """
    tok = cargado.tokenizador
    enc = tok(texto, truncation=True, max_length=cargado.config.max_token_length,
              return_offsets_mapping=True, return_tensors="pt")
    dispositivo = next(cargado.modelo.parameters()).device
    ids = enc["input_ids"].to(dispositivo)
    mascara = enc["attention_mask"].to(dispositivo)
    rollout = extract_text_saliency_rollout(cargado.modelo.text_encoder, ids, mascara)[0]

    palabras: dict[int, list] = {}
    for pos, (palabra, (ini, fin)) in enumerate(zip(enc.word_ids(0),
                                                    enc["offset_mapping"][0].tolist())):
        if palabra is None or fin <= ini:      # [CLS], [SEP]
            continue
        if palabra not in palabras:
            palabras[palabra] = [ini, fin, 0.0]
        palabras[palabra][1] = fin
        palabras[palabra][2] += float(rollout[pos])
    spans = [palabras[k] for k in sorted(palabras)]
    leido_hasta = spans[-1][1] if spans else 0
    return {"spans": [[i, f, round(p, 8)] for i, f, p in spans],
            "leido_hasta": leido_hasta,
            "truncado": leido_hasta < len(texto.rstrip())}


def palabras_con_mas_peso(texto: str, spans: list, n: int = 5) -> list[str]:
    """Las `n` palabras de mayor peso, en minúsculas, como las cita el reporte."""
    orden = sorted(spans, key=lambda s: -s[2])[:n]
    return [texto[i:f].lower() for i, f, _ in orden]


# ---------------------------------------------------------------------------
# Dibujo
# ---------------------------------------------------------------------------

def _segmentos(texto: str, spans: list, ocultar: list[tuple[str, Callable]]):
    """Trozos separados por espacios, con su peso y su forma visible.

    `ocultar` es una lista de (patrón, función) que sustituye direcciones,
    enlaces o teléfonos por su forma segura; un trozo así conserva el peso
    máximo de las palabras que cubría.
    """
    rangos = []
    for patron, reemplazo in ocultar:
        for m in re.finditer(patron, texto, flags=re.I):
            if not any(a < m.end() and m.start() < b for a, b, _ in rangos):
                rangos.append((m.start(), m.end(), reemplazo(m.group(0))))
    rangos.sort()

    def peso(a, b):
        return max([p for i, f, p in spans if i < b and a < f] or [0.0])

    salida, pos = [], 0
    for m in re.finditer(r"\S+", texto):
        a, b = m.span()
        if a < pos:
            continue
        rango = next((r for r in rangos if r[0] < b and a < r[1]), None)
        if rango:
            ini, fin, visible = rango
            fin_real = max(fin, b)
            salida.append((visible, peso(min(ini, a), fin_real), a, fin_real))
            pos = fin_real
        else:
            salida.append((m.group(0), peso(a, b), a, b))
            pos = b
    return salida


def dibujar_resaltado(texto: str, resultado: dict, destino: Path,
                      citadas: list[str] | None = None,
                      ocultar: list[tuple[str, Callable]] | None = None,
                      ancho_px: int = 1100, dpi: int = 150,
                      omitir: tuple[int, int, str] | None = None) -> Path:
    """Dibuja el texto con cada palabra sombreada según su peso y lo guarda en PNG.

    Se sombrea el quinto de palabras con más peso, con intensidad lineal entre el
    percentil 80 (sin sombra) y el peso máximo (sombra plena): sombrear todo lo que
    supera la mediana teñía medio correo y escondía lo decisivo. Las palabras que
    cita el reporte (`citadas`) llevan además recuadro.

    `omitir` = (inicio, fin, nota) sustituye por la nota el tramo de texto entre
    esas posiciones; sirve para no mostrar dos veces un bloque que el correo
    repite. Los pesos de lo que se muestra no cambian.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import IdentityTransform

    leido = texto[:resultado["leido_hasta"]]
    segs = _segmentos(leido, resultado["spans"], ocultar or [])
    pesos = sorted(p for _, p, _, _ in segs)
    nota_k = None
    if omitir:
        ini, fin, nota = omitir
        antes = [g for g in segs if g[2] < ini]
        despues = [g for g in segs if g[2] >= fin]
        nota_k = len(antes)
        segs = antes + [(nota, 0.0, ini, fin)] + despues
    mediana = pesos[int(len(pesos) * 0.8)] if pesos else 0.0
    maximo = pesos[-1] if pesos else 1.0
    citadas = [c.lower() for c in (citadas or [])]
    # Cada palabra citada se recuadra una sola vez: en su aparición de más peso.
    recuadro = set()
    for c in citadas:
        candidatos = [k for k, (s, p, _, _) in enumerate(segs) if k != nota_k
                      and (c in re.split(r"[^\w]+", s.lower()) or s.lower() == c)]
        if candidatos:
            recuadro.add(max(candidatos, key=lambda k: segs[k][1]))

    fuente = FontProperties(family="DejaVu Sans", size=10.5)
    fuente_nota = FontProperties(family="DejaVu Sans", size=9.5, style="italic")
    escala = dpi / 72.0
    margen, espacio = 14 * escala / 2, 8 * escala / 2
    alto_linea = 27 * escala / 2

    fig = plt.figure(figsize=(ancho_px / dpi, 4), dpi=dpi)
    renderer = fig.canvas.get_renderer()

    def ancho(s, fp=fuente):
        w, _, _ = renderer.get_text_width_height_descent(s, fp, ismath=False)
        return w

    lineas, actual, x = [], [], margen
    for k, (s, p, _, _) in enumerate(segs):
        w = ancho(s, fuente_nota if k == nota_k else fuente)
        if actual and x + w > ancho_px - margen:
            lineas.append(actual)
            actual, x = [], margen
        actual.append((k, s, p, x, w))
        x += w + espacio
    if actual:
        lineas.append(actual)

    leyenda = ["Sombreado: cuanto más intenso, más pesó esa palabra en la decisión.",
               "Recuadro: las palabras que aparecen en la explicación."]
    extra = 3 + (1 if resultado.get("truncado") else 0)
    alto_px = int(margen * 2 + alto_linea * (len(lineas) + extra))
    fig.set_size_inches(ancho_px / dpi, alto_px / dpi)
    T = IdentityTransform()
    y = alto_px - margen - alto_linea * 0.8
    for renglon in leyenda:
        fig.text(margen, y, renglon, transform=T, fontproperties=fuente_nota,
                 color="#444444")
        y -= alto_linea * 0.85
    y -= alto_linea * 0.6
    for linea in lineas:
        for k, s, p, x, w in linea:
            if k == nota_k:
                fig.text(x, y, s, transform=T, fontproperties=fuente_nota, color="#666666")
                continue
            nivel = 0.0 if maximo <= mediana else max(0.0, (p - mediana) / (maximo - mediana))
            caja = dict(boxstyle="square,pad=0.18", linewidth=1.4 if k in recuadro else 0,
                        facecolor=(0.86, 0.24, 0.12, 0.12 + 0.68 * nivel) if nivel > 0
                        else (1, 1, 1, 0),
                        edgecolor="#111111" if k in recuadro else "none")
            fig.text(x, y, s, transform=T, fontproperties=fuente, color="#111111",
                     bbox=caja)
        y -= alto_linea
    if resultado.get("truncado"):
        fig.text(margen, y - alto_linea * 0.2,
                 "[…] El sistema solo lee el comienzo del correo; el resto no influyó.",
                 transform=T, fontproperties=fuente_nota, color="#666666")
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=dpi, facecolor="white")
    plt.close(fig)
    return destino
