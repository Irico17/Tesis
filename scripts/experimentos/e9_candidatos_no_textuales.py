"""
E9 · Candidatos del panel con peso de estructura o de red en su reporte (R3.3)

Complementa `e9_candidatos_del_panel.py`. Aquel aplica los filtros previos del
panel (confianza de 0.99 o más, cuerpo de 250 a 1 600 caracteres) y, con ellos,
casi ningún reporte atribuye peso a la estructura o a la red. Este guion parte de
`modalidades_del_reporte.json` y explica, sin esos filtros, todos los correos
bien clasificados cuyo reporte da algún peso no textual, para juzgar si alguno
se entiende bien. El cuerpo se recorta al mostrarlo, como en el panel.

    python scripts/experimentos/e9_candidatos_no_textuales.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import BASE, SALIDA, cargar_corpus  # noqa: E402

sys.path.insert(0, str(BASE / "src"))

from phishing_model.explicabilidad import carga  # noqa: E402

import e9_ejemplos_del_panel as P  # noqa: E402
import e9_narrativa_y_validacion as e9  # noqa: E402


def main() -> int:
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    mod = json.loads((SALIDA / "e9" / "modalidades_del_reporte.json").read_text(encoding="utf-8"))
    ids = [r["email_id"] for r in mod["por_correo"]
           if r["clase_predicha"] == r["etiqueta_real"]
           and r["reporte_pct"]["estructura"] + r["reporte_pct"]["red"] > 0]
    corpus = cargar_corpus().set_index("email_id")
    cargado = carga.cargar(BASE, carga.CORRIDA_DE_REFERENCIA, dispositivo=dispositivo)
    tecnica = e9._tecnica_seleccionada()
    salida = []
    for eid in ids:
        fila = corpus.loc[eid].copy()
        fila["email_id"] = eid
        cuerpo_crudo = P._cuerpo_visible(fila)
        lote = carga.filas_como_lotes(cargado, fila.to_frame().T, dispositivo=dispositivo)[0]
        exp = e9._explicar(cargado, lote, tecnica)
        es_ph = exp["clase_predicha"] == 1
        asunto = P._desactivar(str(fila["subject"]).strip(), es_ph)
        cuerpo = P._desactivar(P._recortar(cuerpo_crudo), es_ph)
        visible = (asunto + " " + cuerpo).lower()
        bloques = {b["modality"]: b for b in exp["estructurado"]["modality_blocks"]}
        salida.append({
            "email_id": eid, "coleccion": fila["source_dataset"],
            "clase": "phishing" if es_ph else "legitimo",
            "idioma": P._idioma(cuerpo_crudo), "largo": len(cuerpo_crudo),
            "confianza": exp["confianza"], "asunto": asunto, "cuerpo": cuerpo,
            "factores": [{"name": f["name"], "human_name": f["human_name"], "pct": f["pct"],
                          "visible": f["name"].lower() in visible}
                         for f in bloques["texto"]["factors"]],
            "no_textuales": [{"modalidad": m, "human_name": f["human_name"], "pct": f["pct"]}
                             for m in ("estructura", "red") for f in bloques[m]["factors"]
                             if f["pct"] > 0],
            "no_textual_pct": {m: bloques[m]["total_pct"] for m in ("estructura", "red")},
            "narrativa": exp["narrativa"],
        })
    destino = SALIDA / "e9" / "candidatos_no_textuales.json"
    destino.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {len(salida)} correos; escrito {destino.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
