"""
Manifiesto de la ejecución completa de la cola de experimentos.

Cada informe declara cuándo se generó él, pero el capítulo necesita declarar
cuándo se ejecutó el CONJUNTO: una tabla de procedencia que solo recoge la fecha
de un experimento describe una parte y sugiere que esa es la del todo. Además,
cualquier experimento puede reejecutarse por separado después, y entonces su
marca deja de representar a la corrida.

El manifiesto se construye a partir del registro que la propia cola escribe, y no
a partir de los informes, porque el registro es el único artefacto que conoce el
orden, los fallos y los reintentos. Se puede reconstruir en cualquier momento y
sobre cualquier corrida pasada, que es lo que lo hace verificable.

    python scripts/experimentos/manifiesto_de_ejecucion.py
    python scripts/experimentos/manifiesto_de_ejecucion.py --registros <carpeta>
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
REGISTROS = BASE / "medios_de_verificacion" / "R1.4_entrenamiento" / "registros"
SALIDA = BASE / "medios_de_verificacion" / "EJECUCION_DE_LA_COLA.json"

FORMATO = "%Y-%m-%d %H:%M:%S"
# La cola escribe una línea al empezar cada experimento y otra al acabarlo. El
# cierre, que consolida y dibuja, deja su propio registro aparte porque puede
# ejecutarse más tarde si algún experimento hubo de repetirse.
RE_INICIO = re.compile(r"^\[(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)\] (e\d_\w+)$")
RE_FIN = re.compile(r"^\[(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)\] (e\d_\w+) (terminado|FALLÓ.*)$")


def _momento(texto: str) -> datetime:
    return datetime.strptime(texto, FORMATO)


def _legible(delta_segundos: float) -> str:
    horas, resto = divmod(int(delta_segundos), 3600)
    minutos = resto // 60
    if horas and minutos:
        return f"{horas} h {minutos} min"
    if horas:
        return f"{horas} h"
    return f"{minutos} min"


def leer_cola(registros: Path) -> dict:
    """Recorre el registro de la cola y reconstruye qué corrió, cuándo y con qué suerte."""
    cola = registros / "cola.log"
    if not cola.exists():
        raise SystemExit(f"No existe {cola}. Sin registro no hay manifiesto.")

    abiertos: dict[str, datetime] = {}
    tramos: list[dict] = []
    for linea in cola.read_text(encoding="utf-8", errors="replace").splitlines():
        linea = linea.rstrip()
        fin = RE_FIN.match(linea)
        if fin:
            momento, nombre, estado = fin.groups()
            arranque = abiertos.pop(nombre, None)
            tramos.append({
                "experimento": nombre,
                "inicio": arranque.strftime(FORMATO) if arranque else None,
                "fin": momento,
                "duracion": (_legible((_momento(momento) - arranque).total_seconds())
                             if arranque else None),
                "resultado": "terminado" if estado == "terminado" else "falló",
            })
            continue
        ini = RE_INICIO.match(linea)
        if ini:
            momento, nombre = ini.groups()
            abiertos[nombre] = _momento(momento)
    return {"tramos": tramos}


def leer_cierre(registros: Path) -> dict:
    """El cierre puede ejecutarse aparte, si algún experimento hubo de repetirse."""
    cierre = registros / "cierre.log"
    if not cierre.exists():
        return {}
    datos: dict = {}
    for linea in cierre.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\[(20\d\d-\d\d-\d\d \d\d:\d\d:\d\d)\] (.+)$", linea.rstrip())
        if not m:
            continue
        momento, texto = m.groups()
        if "esperando" in texto:
            datos["espera_iniciada"] = momento
        elif "se cierra la cola" in texto:
            datos["reintento_concluido"] = momento
    return datos


def construir(registros: Path) -> dict:
    cola = leer_cola(registros)
    tramos = cola["tramos"]
    if not tramos:
        raise SystemExit("El registro no contiene ningún experimento; no hay nada que declarar.")

    cierre = leer_cierre(registros)
    marca = registros / "COLA_COMPLETA"
    fin_declarado = marca.read_text(encoding="utf-8").strip() if marca.exists() else None

    inicios = [t["inicio"] for t in tramos if t["inicio"]]
    finales = [t["fin"] for t in tramos if t["fin"]]
    # El reintento de un experimento que falló en la cola concluye DESPUÉS del
    # último tramo, de modo que cuenta para el cierre de la ejecución.
    if cierre.get("reintento_concluido"):
        finales.append(cierre["reintento_concluido"])
    if fin_declarado:
        finales.append(fin_declarado)

    inicio, fin = min(inicios), max(finales)
    fallidos = [t["experimento"] for t in tramos if t["resultado"] == "falló"]
    repetidos = sorted({t["experimento"] for t in tramos
                        if t["experimento"] in fallidos})

    return {
        "que_es": ("Ventana temporal de la ejecución completa de la cola de "
                   "experimentos, reconstruida desde su propio registro."),
        "inicio": inicio,
        "fin": fin,
        "duracion": _legible((_momento(fin) - _momento(inicio)).total_seconds()),
        "huso": "hora local del servidor de ejecución",
        "n_experimentos": len({t["experimento"] for t in tramos}),
        "orden_de_ejecucion": [t["experimento"] for t in tramos],
        "tramos": tramos,
        "experimentos_repetidos": repetidos,
        "nota_sobre_repeticiones": (
            "Un experimento que falla no cancela los demás. Si alguno hubo de "
            "repetirse, su reejecución concluye después del último tramo de la cola "
            "y el cierre espera a que termine antes de consolidar."
            if repetidos else
            "Ningún experimento tuvo que repetirse."),
        "marca_de_cierre": fin_declarado,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registros", default=str(REGISTROS))
    p.add_argument("--salida", default=str(SALIDA))
    args = p.parse_args()

    manifiesto = construir(Path(args.registros))
    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2),
                      encoding="utf-8")

    print(f"Ejecución de la cola: {manifiesto['inicio']} a {manifiesto['fin']} "
          f"({manifiesto['duracion']})")
    for t in manifiesto["tramos"]:
        print(f"  {t['experimento']:28s} {t['inicio']} -> {t['fin']}  "
              f"{t['duracion'] or '—':>10s}  {t['resultado']}")
    if manifiesto["experimentos_repetidos"]:
        print("  repetidos: " + ", ".join(manifiesto["experimentos_repetidos"]))
    print(f"Manifiesto en {salida.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
