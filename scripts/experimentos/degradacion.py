"""
Degradación adversaria del texto, aplicada SOLO en evaluación.

Es el eje simétrico del que ya tenía E3. Allí se retira una rama porque en
despliegue hay correos que sencillamente no traen HTML --ausencia no adversaria--.
Aquí la rama está pero llega corrompida, que es lo que hace un atacante.

**Modelo de amenaza, y sus límites.** El atacante controla el cuerpo del mensaje.
No controla el resultado de autenticación, que lo estampa la infraestructura
receptora. La asimetría es PARCIAL y así hay que declararla: un atacante puede
comprar un dominio y pasar SPF sin problema; lo que no puede es pasar DMARC
alineado suplantando la marca que imita. Afirmar más que eso sería sobrevender el
supuesto.

**Por qué esto no fuerza el resultado.** Los operadores y las intensidades se
fijan en `doc/PREREGISTRO_HIPOTESIS.md` ANTES de ejecutar; se aplican idénticos a
todos los modelos, sin reentrenar a ninguno; y se informa la rejilla completa,
incluidas las celdas donde el modelo propuesto pierde. Elegir después el operador
que más separa sería exactamente la manipulación que esto evita.

**Limitación declarada.** Son proxies sintéticos de evasión, no comportamiento de
atacante observado. No se puede afirmar que estén calibrados a la distribución
real de ataques; solo que cada operador corresponde a una técnica documentada.
"""

from __future__ import annotations


import numpy as np
import pandas as pd

# Intensidades de la rejilla. El cero es el control: sin él, una caída podría ser
# del operador o del propio paso de reescritura.
INTENSIDADES = (0.0, 0.05, 0.10, 0.25, 0.50, 1.00)

# Latino -> homóglifo cirílico o griego. Se ven igual en pantalla y son fichas
# distintas para cualquier tokenizador.
_HOMOGLIFOS = {"a": "а", "c": "с", "e": "е", "i": "і",
               "j": "ј", "o": "о", "p": "р", "s": "ѕ",
               "x": "х", "y": "у", "A": "А", "B": "В",
               "C": "С", "E": "Е", "H": "Н", "K": "К",
               "M": "М", "O": "О", "P": "Р", "T": "Т",
               "X": "Х"}

_ANCHO_CERO = "​"


def _rng(semilla: int, etiqueta: str) -> np.random.Generator:
    """Generador reproducible por (semilla, operador): la misma celda de la
    rejilla produce el mismo texto en cualquier máquina y en cualquier corrida."""
    return np.random.default_rng(abs(hash((semilla, etiqueta))) % (2**32))


def homoglifos(texto: str, eps: float, rng: np.random.Generator) -> str:
    """Sustituye una fracción `eps` de los caracteres sustituibles."""
    ch = list(texto)
    idx = [i for i, c in enumerate(ch) if c in _HOMOGLIFOS]
    if not idx:
        return texto
    n = int(round(eps * len(idx)))
    for i in rng.choice(idx, size=n, replace=False) if n else ():
        ch[i] = _HOMOGLIFOS[ch[i]]
    return "".join(ch)


def ancho_cero(texto: str, eps: float, rng: np.random.Generator) -> str:
    """Inserta caracteres invisibles DENTRO de las palabras, que es lo que parte
    la ficha; entre palabras no cambiaría nada."""
    palabras = texto.split(" ")
    idx = [i for i, p in enumerate(palabras) if len(p) > 3]
    if not idx:
        return texto
    n = int(round(eps * len(idx)))
    for i in rng.choice(idx, size=n, replace=False) if n else ():
        p = palabras[i]
        corte = int(len(p) / 2)
        palabras[i] = p[:corte] + _ANCHO_CERO + p[corte:]
    return " ".join(palabras)


def entidades_html(texto: str, eps: float, rng: np.random.Generator) -> str:
    """Codifica caracteres como entidades numéricas. Se renderiza idéntico."""
    ch = list(texto)
    idx = [i for i, c in enumerate(ch) if c.isalpha() and ord(c) < 128]
    if not idx:
        return texto
    n = int(round(eps * len(idx)))
    for i in rng.choice(idx, size=n, replace=False) if n else ():
        ch[i] = f"&#{ord(ch[i])};"
    return "".join(ch)


def truncamiento(texto: str, eps: float, rng: np.random.Generator) -> str:
    """Conserva solo el (1-eps) inicial del cuerpo.

    Modela el cuerpo puesto como IMAGEN: el texto extraíble se reduce a lo poco
    que quede fuera de ella. Es el operador más realista de los cuatro --el
    phishing lo hace constantemente-- y el único que no depende de que el
    tokenizador se confunda.
    """
    if eps <= 0:
        return texto
    corte = int(round(len(texto) * (1.0 - eps)))
    return texto[:corte]


OPERADORES = {
    "homoglifos": homoglifos,
    "ancho_cero": ancho_cero,
    "entidades_html": entidades_html,
    "truncamiento": truncamiento,
}


def degradar(df: pd.DataFrame, operador: str, eps: float,
             semilla: int = 42) -> pd.DataFrame:
    """Devuelve una copia con `clean_text` degradado. No toca ninguna otra rama:
    la degradación es del texto, y mezclarla con la ausencia de modalidades
    impediría atribuir la caída a ninguna de las dos."""
    if operador not in OPERADORES:
        raise KeyError(f"operador desconocido: {operador}")
    if eps == 0.0:
        return df
    fn, rng = OPERADORES[operador], _rng(semilla, operador)
    salida = df.copy()
    salida["clean_text"] = [fn(str(t or ""), eps, rng)
                            for t in salida["clean_text"].fillna("")]
    return salida
