"""
E9 · Versión del panel con los correos de explicación más clara (R3.3)

La versión principal del panel equilibra las clases (dos de phishing y dos
legítimos) e incluye un correo con peso no textual. Esta versión alternativa
elige, sin atender a la clase, los correos cuya explicación se entiende mejor,
e incluye entre los cuatro el del ejemplo guiado, que además sigue como guía.

Criterio de claridad, aplicado a los candidatos de `candidatos_del_panel.json`:
- las tres palabras que cita la narrativa son palabras con contenido (legibles,
  alfabéticas, de tres o más letras, que no sean palabras vacías) y se ven en el
  correo mostrado;
- ninguna es un nombre propio ni forma parte de la dirección del remitente
  (las iniciales o el usuario de quien escribe no explican nada al lector);
- se ordenan por una puntuación que premia las palabras que son señales
  reconocibles de su clase (de fraude en el phishing, de conversación técnica en
  los legítimos), las palabras con contenido y la legibilidad de los cinco
  factores; los empates se resuelven por número de señales y por confianza.

    python scripts/experimentos/e9_ejemplos_por_claridad.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import BASE, SALIDA, cargar_corpus  # noqa: E402

sys.path.insert(0, str(BASE / "src"))

from phishing_model.explicabilidad import carga  # noqa: E402
from phishing_model.xai.narrative import es_palabra_legible  # noqa: E402

import e9_ejemplos_del_panel as P  # noqa: E402
import e9_narrativa_y_validacion as e9  # noqa: E402

VACIAS = P.VACIAS | set("""all any also been more most other some such only own same very just
don should now about after again against before being below between both during each few
from further here how once out over under until up why would could their there where while
http https www com html org net""".split())

SENALES = {
    "phishing": re.compile(
        r"^(account|accounts|verif\w*|confirm\w*|password\w*|login|signin|suspend\w*|restrict\w*|"
        r"limit\w*|unusual|secur\w*|update\w*|bank\w*|paypal|ebay|amazon|apple|microsoft|netflix|"
        r"urgent\w*|immediate\w*|click|expire\w*|unlock\w*|restor\w*|billing|invoice|payment\w*|"
        r"credit|card|refund\w*|won|winner|prize|lottery|claim|transfer\w*|inbox|mailbox|quota|"
        r"deactivat\w*|reactivat\w*|alert|notice|notification\w*|myaccount|identity|unauthori\w*|"
        r"fraud\w*|validat\w*|access|customer|dear|bonus|reward\w*|cuenta|verifique|contraseña|"
        r"banco|tarjeta|urgente|bloque\w*|transffered|authomated|recieve\w*|acount|pasword)$"),
    "legitimo": re.compile(
        r"^(wrote|thanks|thank|patch\w*|list|mailing|kernel|fedora|package\w*|bug\w*|version|"
        r"install\w*|config\w*|reply|question|regards|build|compile\w*|error|log|debug\w*|release|"
        r"upstream|repo\w*|commit\w*|driver\w*|module\w*|systemd|rpm|dnf|yum|gnome|kde|python|"
        r"script|server|test\w*|issue\w*|fix\w*|source|maintainer|devel|announce\w*|meeting|wiki|"
        r"docs|cheers|hi|hello|gracias|saludos|paquete\w*|versión|instal\w*)$"),
}


def _contenido(w: str) -> bool:
    return es_palabra_legible(w) and w.isalpha() and len(w) >= 3 and w.lower() not in VACIAS


def ranking(candidatos: list[dict], remitentes: dict[str, str]) -> list[dict]:
    salida = []
    for c in candidatos:
        f = c["factores"]
        nombres = [x["name"].lower() for x in f]
        if len(f) < 3 or len(set(nombres)) < len(nombres):
            continue
        top3 = f[:3]
        remitente = remitentes.get(c["email_id"], "").lower()
        if not all(x["visible"] and _contenido(x["name"]) and not x["nombre_propio"]
                   and x["name"].lower() not in remitente for x in top3):
            continue
        sen = SENALES[c["clase"]]
        senales3 = sum(bool(sen.match(x["name"].lower())) for x in top3)
        senales5 = sum(bool(sen.match(x["name"].lower())) for x in f[:5])
        puntos = (3 * senales3 + senales5 + 2 * 3 + sum(_contenido(x["name"]) for x in f[:5])
                  + sum(es_palabra_legible(x["name"]) for x in f[:5]) + 2)
        salida.append({"email_id": c["email_id"], "clase": c["clase"], "coleccion": c["coleccion"],
                       "asunto": c["asunto"], "confianza": c["confianza"], "puntos": puntos,
                       "senales3": senales3, "citadas": [x["name"] for x in top3]})
    salida.sort(key=lambda r: (-r["puntos"], -r["senales3"], -r["confianza"]))
    return salida


# El ejemplo guiado de esta versión es el de TD Canada Trust, que además figura
# entre los cuatro correos. Se fija aquí porque la versión principal cambió de
# guía tras la prueba piloto.
GUIADO = "a68f6ab2-d2d5-4570-9914-ea410c4f788e"


def main() -> int:
    candidatos = json.loads((SALIDA / "e9" / "candidatos_del_panel.json").read_text(encoding="utf-8"))
    principal = json.loads((SALIDA / "e9" / "ejemplos_del_panel.json").read_text(encoding="utf-8"))
    corpus = cargar_corpus()
    remitentes = dict(zip(corpus["email_id"], corpus["sender"].astype(str)))
    orden = ranking(candidatos, remitentes)
    guiado = GUIADO
    elegidos_ids = [r["email_id"] for r in orden if r["email_id"] != guiado][:3]
    print("  ranking de claridad (diez primeros):")
    for i, r in enumerate(orden[:10], 1):
        print(f"    {i:2d}. [{r['puntos']}] {r['email_id'][:8]} {r['clase']:9s} {r['asunto'][:50]} "
              f"| {', '.join(r['citadas'])}")

    # El orden de presentación alterna las clases cuando se puede, y el guiado va
    # al final, para que el evaluador lea primero correos que no ha visto.
    por_id = {r["email_id"]: r for r in orden}
    tres = sorted(elegidos_ids, key=lambda i: (por_id[i]["clase"] != "phishing", -por_id[i]["puntos"]))
    ph = [i for i in tres if por_id[i]["clase"] == "phishing"]
    le = [i for i in tres if por_id[i]["clase"] == "legitimo"]
    alterno = []
    while ph or le:
        if ph:
            alterno.append(ph.pop(0))
        if le:
            alterno.append(le.pop(0))
    secuencia = alterno + [guiado]

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    cargado = carga.cargar(BASE, carga.CORRIDA_DE_REFERENCIA, dispositivo=dispositivo)
    tecnica = e9._tecnica_seleccionada()
    corpus = corpus.set_index("email_id", drop=False)
    claves = ["correo_1", "correo_2", "correo_3", "correo_4"]
    ejemplos = {}
    for clave, eid in zip(["induccion"] + claves, [guiado] + secuencia):
        fila = corpus.loc[eid].copy()
        fila["cuerpo"] = P._cuerpo_visible(fila)
        fila["idioma"] = P._idioma(fila["cuerpo"])
        pred = 1 if por_id[eid]["clase"] == "phishing" else 0
        fila["y_pred"] = pred
        lote = carga.filas_como_lotes(cargado, fila.to_frame().T, dispositivo=dispositivo)[0]
        exp = e9._explicar(cargado, lote, tecnica)
        ejemplos[clave] = P._ficha(fila, exp, cargado)
        print(f"  {clave}: {eid[:8]} {por_id[eid]['clase']} {por_id[eid]['asunto'][:50]}")

    informe = {
        "experimento": "e9_ejemplos_por_claridad",
        "resultado": "R3.3 (versión alternativa del panel)",
        "criterio": __doc__.split("Criterio de claridad")[1].split("python ")[0].strip(),
        "ranking": orden,
        "secuencia": secuencia,
        "ejemplos": ejemplos,
    }
    destino = SALIDA / "e9" / "ejemplos_del_panel_claridad.json"
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  escrito {destino.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
