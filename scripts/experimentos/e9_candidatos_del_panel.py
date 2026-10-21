"""
E9 · Todos los candidatos del panel con su explicación (R3.3)

`e9_ejemplos_del_panel.py` explica los 300 primeros candidatos de cada grupo, en
un orden fijo, y se queda con los primeros que cumplen los criterios. Para juzgar
si hay correos cuya explicación se entienda mejor, este guion explica a todos los
candidatos con los mismos filtros previos (bien clasificados, confianza de 0.99
o más, cuerpo de 250 a 1 600 caracteres, inglés o español) y guarda el reporte de
cada uno, el cuerpo tal como se mostraría y las comprobaciones de visibilidad.
La puntuación y la elección se hacen después, sobre este informe.

    python scripts/experimentos/e9_candidatos_del_panel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import BASE, SALIDA, cargar_corpus  # noqa: E402

sys.path.insert(0, str(BASE / "src"))

from phishing_model.explicabilidad import carga  # noqa: E402

import e9_ejemplos_del_panel as P  # noqa: E402
import e9_narrativa_y_validacion as e9  # noqa: E402


def main() -> int:
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = cargar_corpus()
    corrida = carga.CORRIDA_DE_REFERENCIA
    ruta = sorted((BASE / "data" / "predictions").glob(f"{corrida}*_preds.parquet"))[0]
    pred = pd.read_parquet(ruta)[["email_id", "y_true", "y_pred", "y_proba"]]
    prueba = corpus.merge(pred, on="email_id")
    prueba = prueba[prueba["y_true"] == prueba["y_pred"]].copy()
    prueba["cuerpo"] = prueba.apply(P._cuerpo_visible, axis=1)
    prueba["confianza"] = prueba["y_proba"].where(prueba["y_pred"] == 1, 1 - prueba["y_proba"])
    cand = prueba[prueba["cuerpo"].str.len().between(P.LARGO_MIN, P.LARGO_MAX)
                  & prueba["subject"].fillna("").str.strip().astype(bool)
                  & (prueba["confianza"] >= 0.99)].copy()
    cand["idioma"] = cand["cuerpo"].map(P._idioma)
    cand = cand[cand["idioma"].isin(["en", "es"])].copy()
    print(f"  candidatos: {len(cand)}")

    cargado = carga.cargar(BASE, corrida, dispositivo=dispositivo)
    tecnica = e9._tecnica_seleccionada()
    salida = []
    for i, (_, fila) in enumerate(cand.iterrows(), 1):
        lote = carga.filas_como_lotes(cargado, fila.to_frame().T, dispositivo=dispositivo)[0]
        exp = e9._explicar(cargado, lote, tecnica)
        es_ph = int(fila["y_pred"]) == 1
        asunto = P._desactivar(str(fila["subject"]).strip(), es_ph)
        cuerpo = P._desactivar(P._recortar(fila["cuerpo"]), es_ph)
        visible = (asunto + " " + cuerpo).lower()
        texto = next(b for b in exp["estructurado"]["modality_blocks"] if b["modality"] == "texto")
        salida.append({
            "email_id": fila["email_id"], "coleccion": fila["source_dataset"],
            "clase": "phishing" if es_ph else "legitimo", "idioma": fila["idioma"],
            "confianza": exp["confianza"], "asunto": asunto, "cuerpo": cuerpo,
            "factores": [{"name": f["name"], "human_name": f["human_name"], "pct": f["pct"],
                          "visible": f["name"].lower() in visible,
                          "nombre_propio": (not es_ph) and P._es_nombre_propio(
                              f["name"], P._desactivar(fila["cuerpo"]))}
                         for f in texto["factors"]],
            "no_textual_pct": {b["modality"]: b["total_pct"]
                               for b in exp["estructurado"]["modality_blocks"]
                               if b["modality"] != "texto"},
            "apta_con_criterios_actuales": P._apta(exp),
            "narrativa": exp["narrativa"],
        })
        if i % 100 == 0:
            print(f"    {i}/{len(cand)}", flush=True)
    destino = SALIDA / "e9" / "candidatos_del_panel.json"
    destino.write_text(json.dumps(salida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  escrito {destino.relative_to(BASE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
