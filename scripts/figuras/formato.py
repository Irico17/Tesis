"""
Geometría de impresión de las figuras, en un solo sitio.

La revisión de forma midió lo que ocurría antes: las láminas se dibujaban con
lienzos de once a dieciséis pulgadas y tipografías de 7.6 a 10 puntos, y el
documento las encajaba en una caja de texto de 15.9 cm. El factor de compresión
resultante iba de 0.46 a 0.68, de modo que un rótulo de 7.6 puntos se imprimía a
4.6 y la figura quedaba ilegible a tamaño de página aunque se viera bien en
pantalla.

La regla que impone este módulo es una sola: **una figura se dibuja al ancho al
que se imprime**. Con el lienzo del ancho de la caja de texto, el factor vale uno
y un rótulo de ocho puntos se imprime a ocho puntos. Deja de hacer falta razonar
sobre compresiones, y lo que se ve en el PNG es lo que sale en el papel.

De ahí que el ancho no sea un parámetro estético sino una constante del
documento: `ANCHO_VERTICAL` es la caja de la página vertical y `ANCHO_APAISADO`
la de las páginas giradas, que el capítulo reserva para la estructura de
descomposición del trabajo y el diagrama de la arquitectura.

Este módulo no importa matplotlib al cargarse: lo consume tanto el guion de
figuras como `scripts/experimentos/comun.py`, que corre en el servidor.
"""

from __future__ import annotations

# Caja de texto del documento, en pulgadas. La página es A4 con márgenes de
# 2.5 cm, de modo que quedan 15.9 cm de ancho en vertical y 24.5 cm en apaisado.
ANCHO_VERTICAL = 15.9 / 2.54
ANCHO_APAISADO = 24.5 / 2.54

# Altura máxima utilizable antes de que la figura y su pie se partan en dos
# páginas. Una lámina más alta que esto se imprime sola, sin el párrafo que la
# introduce, y el lector pierde el hilo.
ALTO_MAXIMO_VERTICAL = 19.0 / 2.54
ALTO_MAXIMO_APAISADO = 12.5 / 2.54

# Resolución de guardado. Con el lienzo ya al tamaño de impresión, lo único que
# decide la nitidez es la densidad de puntos.
PPP = 200

# Tamaños en puntos, que ahora son los tamaños impresos. El cuerpo del documento
# se compone a once puntos, de modo que ocho en la figura es el escalón habitual
# para material auxiliar y el mínimo por debajo del cual un jurado no lee.
CUERPO = 8.5
ROTULO = 8.0
TITULO = 9.5
MINIMO_LEGIBLE = 7.0


def estilo() -> None:
    """Fija los tamaños de letra por defecto de matplotlib.

    Se llama antes de dibujar. Evita tener que repetir `fontsize=` en cada
    llamada y, sobre todo, evita que una lámina se quede con el tamaño por
    defecto de la biblioteca, que son diez puntos pensados para una figura de
    seis pulgadas y media vista en pantalla.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": CUERPO,
        "axes.titlesize": TITULO,
        "axes.labelsize": ROTULO,
        "xtick.labelsize": ROTULO,
        "ytick.labelsize": ROTULO,
        "legend.fontsize": ROTULO,
        "figure.dpi": PPP,
        "savefig.dpi": PPP,
        "savefig.bbox": "tight",
        "figure.autolayout": False,
    })


def alto_por_filas(n: int, alto_de_fila: float = 0.30,
                   margen: float = 1.65, apaisada: bool = False) -> float:
    """Altura de una lámina de barras horizontales con `n` categorías.

    Las barras horizontales resuelven el problema que tenían las verticales: con
    doce arquitecturas, los nombres solo caben girados y partidos en tres líneas,
    y aun así obligaban a un lienzo de trece pulgadas. Puestos en el eje
    vertical, el nombre se lee de corrido y el ancho deja de depender de cuántas
    filas haya.
    """
    tope = ALTO_MAXIMO_APAISADO if apaisada else ALTO_MAXIMO_VERTICAL
    return min(tope, max(2.6, n * alto_de_fila + margen))


def barras(destinos, nombre: str, titulo: str, etiquetas, medias, desviaciones,
           ylabel: str = "F1", resaltar: int | None = None,
           etiqueta_resaltada: str = "arquitectura propuesta",
           envolver: int = 0) -> None:
    """Barras con dispersión entre semillas, al ancho al que se imprimen.

    Vive aquí, y no en la cola ni en el guion de figuras, porque las dos la
    necesitan y hasta ahora había dos implementaciones que se fueron separando:
    una rotulaba «arquitectura propuesta» la barra que la otra rotulaba «línea
    base», y las láminas del mismo capítulo se contradecían.

    El eje no arranca en cero de forma deliberada: cuando todas las barras están
    entre 0.85 y 0.95, un eje de 0 a 1 las vuelve indistinguibles y oculta
    justamente lo que la lámina debe mostrar. La compresión se declara en el
    rótulo del eje, no como una nota flotante sobre las barras: allí se
    superponía a los valores y quedaba ilegible.

    Las barras van en horizontal a partir de seis categorías. En vertical, doce
    nombres de arquitectura solo caben girados y partidos, y obligaban a un
    lienzo de trece pulgadas que el documento encajaba en 15.9 cm: el rótulo se
    imprimía entonces por debajo de cuatro puntos. Puestas en horizontal, el
    ancho deja de depender de cuántas filas haya y el nombre se lee de corrido.

    `resaltar` señala la barra que conviene distinguir y `etiqueta_resaltada`
    dice qué es. La etiqueta es un parámetro y no un literal porque no siempre se
    resalta la arquitectura propuesta: E2 destaca su línea base de comparación y
    E3 una condición de evaluación. Se distingue con un trazado distinto y no con
    un color de alarma: el trabajo no presupone ganadora y la lámina no debe
    sugerirla.
    """
    import textwrap

    estilo()
    import matplotlib.pyplot as plt

    if envolver:
        etiquetas = ["\n".join(textwrap.wrap(e, envolver)) for e in etiquetas]
    horizontal = len(etiquetas) >= 6
    bajo = max(0.0, min(m - d for m, d in zip(medias, desviaciones)) - 0.05)
    alto = min(1.0, max(m + d for m, d in zip(medias, desviaciones)) + 0.04)
    holgura = (alto - bajo) * 0.02
    rotulo_eje = f"{ylabel}  (eje recortado desde {bajo:.2f})"

    if horizontal:
        # El orden se invierte para que la primera categoría quede arriba: leídas
        # de abajo arriba, las láminas contradecían al texto, que las enumera en
        # el orden en que se pasan.
        y = list(range(len(etiquetas)))[::-1]
        fig, ax = plt.subplots(
            figsize=(ANCHO_VERTICAL, alto_por_filas(len(etiquetas))))
        trazos = ax.barh(y, medias, xerr=desviaciones, capsize=3, height=0.68,
                         color="#4C72B0", edgecolor="#22303F", linewidth=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(etiquetas)
        ax.set_xlim(bajo, alto + (alto - bajo) * 0.18)
        ax.set_xlabel(rotulo_eje)
        ax.grid(axis="x", alpha=0.3)
        for pos, (m, d) in zip(y, zip(medias, desviaciones)):
            ax.text(m + d + holgura, pos, f"{m:.4f}", va="center",
                    fontsize=MINIMO_LEGIBLE)
    else:
        fig, ax = plt.subplots(figsize=(ANCHO_VERTICAL, 3.9))
        trazos = ax.bar(etiquetas, medias, yerr=desviaciones, capsize=3,
                        color="#4C72B0", edgecolor="#22303F", linewidth=0.6)
        ax.set_ylim(bajo, alto + (alto - bajo) * 0.12)
        ax.set_ylabel(rotulo_eje)
        ax.grid(axis="y", alpha=0.3)
        for i, (m, d) in enumerate(zip(medias, desviaciones)):
            ax.text(i, m + d + holgura, f"{m:.4f}", ha="center",
                    fontsize=MINIMO_LEGIBLE)

    # `pad` reserva sitio: con título de dos líneas la segunda caía sobre las
    # barras más altas y tapaba su valor, que es justo el dato que se mira.
    ax.set_title(titulo, pad=10)
    # El marco superior cruzaba los valores impresos sobre las barras más altas.
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

    if resaltar is not None and 0 <= resaltar < len(trazos):
        trazos[resaltar].set_hatch("//")
        trazos[resaltar].set_edgecolor("#1A2733")
        trazos[resaltar].set_linewidth(1.4)
        # La leyenda va en coordenadas de FIGURA y bajo el trazado: dentro de los
        # ejes tapaba las barras de la esquina, que es donde caen las más altas.
        fig.legend([trazos[resaltar]], [etiqueta_resaltada], loc="lower center",
                   bbox_to_anchor=(0.5, -0.015), frameon=False)

    fig.tight_layout()
    guardar(fig, destinos, nombre)


def guardar(fig, destinos, nombre: str) -> None:
    """Escribe la figura en cada destino pedido y cierra el lienzo.

    `destinos` admite una carpeta suelta, en cuyo caso el fichero se llama
    `nombre.png`, o una secuencia de pares (carpeta, fichero) para las láminas
    que el capítulo y el medio de verificación nombran de forma distinta.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    if isinstance(destinos, (str, Path)):
        destinos = ((Path(destinos), f"{nombre}.png"),)
    for carpeta, fichero in destinos:
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        fig.savefig(carpeta / fichero)
    plt.close(fig)
    print(f"  {nombre}.png")


def escala_de_fuentes(ancho_antiguo: float, ancho_nuevo: float) -> float:
    """Factor que conserva el tamaño relativo del texto al cambiar el lienzo.

    Se usa al reencuadrar una lámina ya compuesta a mano, como el diagrama de la
    arquitectura: si el texto se deja en el mismo número de puntos sobre un
    lienzo más estrecho, ocupa proporcionalmente más y se sale de las cajas.
    """
    return ancho_nuevo / ancho_antiguo
