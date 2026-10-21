"""
E9 · Cuánto pesan la estructura y la red en los reportes (R3.2 y R3.3)

El reporte por modalidad atribuye a cada bloque la suma de sus factores
normalizada. Los factores del texto son pesos de atención y los de estructura y
red son caídas de probabilidad por ablación, de modo que el porcentaje de cada
bloque compara dos escalas. Este guion recorre el conjunto de prueba y registra,
para cada correo:

- el porcentaje que el reporte asigna a la estructura y a la red;
- cuánto cae la probabilidad de la clase predicha si falta cada modalidad no
  textual, medido con el mecanismo de ausencia del propio modelo, que es la
  misma escala para ambas.

Sirve para saber cuántos reportes muestran evidencia no textual y para elegir
correos del panel en que la haya.

    python scripts/experimentos/e9_modalidades_del_reporte.py --fragmento 0/3
    python scripts/experimentos/e9_modalidades_del_reporte.py --unir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import BASE, SALIDA, cargar_corpus  # noqa: E402

sys.path.insert(0, str(BASE / "src"))

from phishing_model.explicabilidad import carga  # noqa: E402
from phishing_model.explicabilidad.atribucion_modal import lote_limpio  # noqa: E402

import e9_narrativa_y_validacion as e9  # noqa: E402

NOMBRE = "modalidades_del_reporte"


def _p(modelo, entrada, clase):
    with torch.no_grad():
        return float(torch.softmax(modelo(entrada), dim=-1)[0, clase])


def _sin(entrada, modalidad):
    """La entrada con una modalidad declarada ausente, como la trata el modelo."""
    e = dict(entrada)
    if modalidad == "estructura":
        e["structural_continuous"] = torch.zeros_like(e["structural_continuous"])
        e["has_structure"] = torch.zeros_like(e["has_structure"])
    else:
        e["network_continuous"] = torch.zeros_like(e["network_continuous"])
        e["network_categorical"] = torch.zeros_like(e["network_categorical"])
        e["has_network"] = torch.zeros_like(e["has_network"])
    return e


def _prueba(corpus, corrida):
    ruta = sorted((BASE / "data" / "predictions").glob(f"{corrida}*_preds.parquet"))[0]
    pred = pd.read_parquet(ruta)[["email_id", "y_true", "y_pred", "y_proba"]]
    return corpus.merge(pred, on="email_id").sort_values("email_id").reset_index(drop=True)


def fragmento(k: int, n: int) -> None:
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = cargar_corpus()
    corrida = carga.CORRIDA_DE_REFERENCIA
    cargado = carga.cargar(BASE, corrida, dispositivo=dispositivo)
    tecnica = e9._tecnica_seleccionada()
    filas = _prueba(corpus, corrida).iloc[k::n]
    salida = []
    for i, (_, fila) in enumerate(filas.iterrows(), 1):
        lote = carga.filas_como_lotes(cargado, fila.to_frame().T, dispositivo=dispositivo)[0]
        exp = e9._explicar(cargado, lote, tecnica)
        bloques = {b["modality"]: b for b in exp["estructurado"]["modality_blocks"]}
        entrada = lote_limpio(lote)
        clase = exp["clase_predicha"]
        p = _p(cargado.modelo, entrada, clase)
        salida.append({
            "email_id": fila["email_id"], "coleccion": fila["source_dataset"],
            "etiqueta_real": int(fila["y_true"]), "clase_predicha": clase,
            "confianza": exp["confianza"],
            "tiene_estructura": float(entrada["has_structure"].flatten()[0]),
            "tiene_red": float(entrada["has_network"].flatten()[0]),
            "reporte_pct": {m: bloques[m]["total_pct"] for m in bloques},
            "factores_no_textuales": [f["name"] for m in ("estructura", "red")
                                      for f in bloques[m]["factors"] if abs(f["pct"]) > 0],
            "caida_sin": {m: round(p - _p(cargado.modelo, _sin(entrada, m), clase), 6)
                          for m in ("estructura", "red")},
            "ilegibles": len(exp["legibilidad"]["ilegibles"]),
        })
        if i % 100 == 0:
            print(f"  {i}/{len(filas)}", flush=True)
    destino = SALIDA / "e9" / f"{NOMBRE}_parte{k}.json"
    destino.write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")
    print(f"  escrito {destino.name}")


def unir() -> None:
    partes = sorted((SALIDA / "e9").glob(f"{NOMBRE}_parte*.json"))
    filas = [r for p in partes for r in json.loads(p.read_text(encoding="utf-8"))]

    def resumen(sub):
        def cuenta(f):
            return sum(1 for r in sub if f(r))
        return {
            "n": len(sub),
            "reporte_estructura_mayor_que_0": cuenta(lambda r: r["reporte_pct"]["estructura"] > 0),
            "reporte_red_mayor_que_0": cuenta(lambda r: r["reporte_pct"]["red"] > 0),
            "reporte_no_textual_al_menos_5": cuenta(
                lambda r: r["reporte_pct"]["estructura"] + r["reporte_pct"]["red"] >= 5),
            "reporte_no_textual_al_menos_20": cuenta(
                lambda r: r["reporte_pct"]["estructura"] + r["reporte_pct"]["red"] >= 20),
            "dominante_no_textual": cuenta(
                lambda r: max(r["reporte_pct"], key=r["reporte_pct"].get) != "texto"),
            "caida_sin_estructura_mayor_que_0_01": cuenta(
                lambda r: r["caida_sin"]["estructura"] > 0.01),
            "caida_sin_red_mayor_que_0_01": cuenta(lambda r: r["caida_sin"]["red"] > 0.01),
        }

    informe = {
        "experimento": "e9_modalidades_del_reporte",
        "n_correos": len(filas),
        "todos": resumen(filas),
        "predichos_phishing": resumen([r for r in filas if r["clase_predicha"] == 1]),
        "predichos_legitimos": resumen([r for r in filas if r["clase_predicha"] == 0]),
        "por_correo": filas,
    }
    destino = SALIDA / "e9" / f"{NOMBRE}.json"
    destino.write_text(json.dumps(informe, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in informe.items() if k != "por_correo"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fragmento")
    ap.add_argument("--unir", action="store_true")
    a = ap.parse_args()
    if a.unir:
        unir()
    else:
        k, n = (int(x) for x in (a.fragmento or "0/1").split("/"))
        fragmento(k, n)
