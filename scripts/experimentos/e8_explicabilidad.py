"""
E8 · Selección del esquema técnico de explicabilidad (R3.1)

El indicador de R3.1 pide un cuadro comparativo que justifique la elección de al
menos una técnica idónea, evaluando comparativamente tres familias: los
mecanismos intrínsecos de atención, los métodos post-hoc basados en valores de
Shapley mediante Captum y las aproximaciones lineales locales. La elección ha de
apoyarse en evidencia empírica, no en preferencia.

Qué mide, y por qué cada criterio:

1. **Fidelidad**, que es el criterio decisivo. Se retiran del mensaje las palabras
   que la técnica declaró decisivas y se observa cuánto cae la probabilidad
   (exhaustividad), y después se conservan solo esas y se observa cuánto se
   sostiene (suficiencia). Es la única evidencia directa de que la explicación
   describe lo que el modelo usa. El acuerdo entre técnicas no lo acredita, porque
   dos pueden coincidir y errar ambas, y la estabilidad tampoco, porque una
   técnica que siempre devuelve la misma explicación equivocada es estable.

2. **Eficiencia y aditividad local**, que la matriz de objetivos compromete de
   forma expresa. Se contrasta la suma de las atribuciones contra la diferencia de
   probabilidad entre el correo y su referencia. Una técnica cuya suma no se
   aproxima a esa diferencia no reparte la decisión, la describe por encima.

3. **Alcance modal**, que es propio de esta tesis y no figura en los manuales.
   LIME y los gradientes integrados atribuyen sobre palabras y dejan fijas las
   ramas de estructura y de red, de modo que no pueden decir cuánto pesó que SPF
   fallara. Una tesis sobre fusión multimodal que explicase solo el texto no
   explicaría su propia aportación.

4. **Coste y estabilidad**, que deciden si la técnica es aplicable en línea y si
   dos ejecuciones sobre el mismo correo dicen lo mismo.

La comparación se hace sobre el modelo que el capítulo reporta, y el guion se
niega a ejecutarse sobre cualquier otro: ver `explicabilidad.procedencia`.

    python scripts/experimentos/e8_explicabilidad.py [--muestra 30] [--sin-contraste]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import BASE, cargar_corpus, emitir, metricas, particion_agrupada  # noqa: E402

sys.path.insert(0, str(BASE / "src"))

from phishing_model.explicabilidad import carga  # noqa: E402
from phishing_model.explicabilidad.atribucion_modal import (  # noqa: E402
    atribucion_gradient_shap, atribucion_por_ablacion, lote_limpio, pesos_de_modalidad,
)
from phishing_model.explicabilidad.procedencia import (  # noqa: E402
    comprobar_contra_lo_reportado, f1_reportado,
)

TOP_K = 5


def _fusionar_subpalabras(tokens: list[str], pesos: list[float]) -> list[tuple[str, float]]:
    """Reagrupa los fragmentos de WordPiece en palabras, sumando sus pesos.

    Sin esto, la comparación entre técnicas sería injusta: LIME opera sobre
    palabras completas y los gradientes sobre fragmentos, de modo que el acuerdo
    entre ambos se mediría entre vocabularios distintos.
    """
    palabras: list[tuple[str, float]] = []
    actual, suma = "", 0.0
    for tok, peso in zip(tokens, pesos):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        if tok.startswith("##"):
            actual += tok[2:]
            suma += peso
        else:
            if actual:
                palabras.append((actual, suma))
            actual, suma = tok, peso
    if actual:
        palabras.append((actual, suma))
    return palabras


def _cima(pares: list[tuple[str, float]], k: int = TOP_K) -> set[str]:
    return {p.lower() for p, _ in sorted(pares, key=lambda kv: -abs(kv[1]))[:k]}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not (a or b):
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def _probabilidad(modelo, lote, clase: int = 1) -> float:
    with torch.no_grad():
        return float(torch.softmax(modelo(lote_limpio(lote)), dim=-1)[0, clase])


def _aditividad(modelo, lote, atribuciones: dict[str, float], clase: int) -> dict:
    """Contrasta la suma de atribuciones con la diferencia de probabilidad.

    La referencia es el correo con las dos ramas tabulares anuladas, que es el
    mismo punto de partida que usan la ablación y GradientSHAP en este guion. Si
    la técnica reparte la decisión, ambas cantidades deben aproximarse.
    """
    entrada = lote_limpio(lote)
    p_correo = _probabilidad(modelo, entrada, clase)

    referencia = dict(entrada)
    referencia["structural_continuous"] = torch.zeros_like(entrada["structural_continuous"])
    referencia["network_continuous"] = torch.zeros_like(entrada["network_continuous"])
    p_referencia = _probabilidad(modelo, referencia, clase)

    suma = sum(atribuciones.values())
    diferencia = p_correo - p_referencia
    return {
        "suma_de_atribuciones": round(suma, 6),
        "diferencia_de_probabilidad": round(diferencia, 6),
        "error_absoluto": round(abs(suma - diferencia), 6),
        "p_correo": round(p_correo, 6),
        "p_referencia": round(p_referencia, 6),
    }


def _sobre_un_correo(cargado, lote, repeticiones: int = 2) -> dict:
    """Corre las tres familias sobre un correo y mide todos los criterios."""
    from phishing_model.xai.attention_explainer import extract_text_saliency_rollout
    from phishing_model.xai.faithfulness import compute_faithfulness
    from phishing_model.xai.lime_explainer import explain_lime
    from phishing_model.xai.shap_explainer import explain_shap

    modelo, tok = cargado.modelo, cargado.tokenizador
    entrada = lote_limpio(lote)
    with torch.no_grad():
        probas = torch.softmax(modelo(entrada), dim=-1)[0]
    clase = int(probas.argmax())
    confianza = float(probas[clase])

    texto = tok.decode(entrada["input_ids"][0][entrada["attention_mask"][0].bool()],
                       skip_special_tokens=True)
    fila: dict = {"clase_predicha": clase, "confianza": round(confianza, 6), "errores": {}}
    palabras_por_tecnica: dict[str, list[tuple[str, float]]] = {}

    # --- Familia 1: atención intrínseca -------------------------------------
    try:
        t0 = time.time()
        modal = pesos_de_modalidad(modelo, entrada)["filas"][0]
        rollout = extract_text_saliency_rollout(
            modelo.text_encoder, entrada["input_ids"], entrada["attention_mask"])[0]
        toks = tok.convert_ids_to_tokens(entrada["input_ids"][0].tolist())
        valida = entrada["attention_mask"][0].bool().tolist()
        palabras = _fusionar_subpalabras(
            [t for t, m in zip(toks, valida) if m],
            [s for s, m in zip(rollout.tolist(), valida) if m])
        palabras_por_tecnica["atencion"] = palabras
        fila["atencion"] = {
            "segundos": round(time.time() - t0, 4),
            "alcance": "texto y modalidades",
            "peso_del_centinela": round(modal["peso_del_centinela"], 6),
            "entropia_normalizada": round(modal["entropia_normalizada"], 6),
            "uniforme_de_referencia": round(modal["uniforme_de_referencia"], 6),
            "por_caracteristica": {k: round(v, 6) for k, v in sorted(
                modal["por_caracteristica"].items(), key=lambda kv: -kv[1])},
        }
    except Exception as exc:
        fila["errores"]["atencion"] = f"{type(exc).__name__}: {exc}"

    # --- Familia 2: Shapley mediante Captum ---------------------------------
    # Dos superficies distintas, y la distinción importa: sobre el texto el
    # método aplicable es el de gradientes integrados por capa, porque
    # GradientSHAP perturba la entrada con ruido y los identificadores de token
    # son enteros que indexan una tabla de embeddings; sobre los vectores de
    # características, que es lo que la matriz de objetivos compromete de forma
    # literal, GradientSHAP sí es aplicable.
    corridas_shap = []
    for _ in range(repeticiones):
        try:
            corridas_shap.append(explain_shap(modelo, tok, entrada, target_class=clase, n_steps=16))
        except Exception as exc:
            fila["errores"].setdefault("shap_texto", []).append(f"{type(exc).__name__}: {exc}")
    if corridas_shap:
        palabras_por_tecnica["shap"] = _fusionar_subpalabras(
            corridas_shap[0]["tokens"], corridas_shap[0]["attributions"])
        fila["shap_texto"] = {
            "segundos": round(np.mean([c["wall_time_seconds"] for c in corridas_shap]), 4),
            "alcance": "solo texto",
            "delta_de_convergencia": corridas_shap[0].get("convergence_delta"),
        }
        if len(corridas_shap) >= 2:
            fila["shap_texto"]["estabilidad_jaccard"] = _jaccard(
                _cima(palabras_por_tecnica["shap"]),
                _cima(_fusionar_subpalabras(corridas_shap[1]["tokens"],
                                            corridas_shap[1]["attributions"])))
    try:
        t0 = time.time()
        gs = atribucion_gradient_shap(modelo, entrada, clase=clase)
        fila["shap_tabular"] = {
            "segundos": round(time.time() - t0, 4),
            "alcance": "solo modalidades",
            "atribuciones": {k: round(v, 6) for k, v in gs.items()},
            "aditividad": _aditividad(modelo, entrada, gs, clase),
        }
    except Exception as exc:
        fila["errores"]["shap_tabular"] = f"{type(exc).__name__}: {exc}"

    # --- Familia 3: aproximaciones locales ----------------------------------
    corridas_lime = []
    for _ in range(repeticiones):
        try:
            corridas_lime.append(explain_lime(modelo, tok, entrada, target_class=clase,
                                              num_samples=200))
        except Exception as exc:
            fila["errores"].setdefault("lime", []).append(f"{type(exc).__name__}: {exc}")
    if corridas_lime:
        palabras_por_tecnica["lime"] = list(zip(corridas_lime[0]["words"],
                                                corridas_lime[0]["attributions"]))
        fila["lime"] = {
            "segundos": round(np.mean([c["wall_time_seconds"] for c in corridas_lime]), 4),
            "alcance": "solo texto",
        }
        if len(corridas_lime) >= 2:
            fila["lime"]["estabilidad_jaccard"] = _jaccard(
                _cima(palabras_por_tecnica["lime"]),
                _cima(list(zip(corridas_lime[1]["words"], corridas_lime[1]["attributions"]))))

    # --- Ablación de rasgos, patrón de referencia sobre las modalidades -----
    try:
        t0 = time.time()
        ab = atribucion_por_ablacion(modelo, entrada, clase=clase)
        fila["ablacion"] = {
            "segundos": round(time.time() - t0, 4),
            "alcance": "solo modalidades",
            "atribuciones": {k: round(v, 6) for k, v in ab.items()},
            "aditividad": _aditividad(modelo, entrada, ab, clase),
        }
    except Exception as exc:
        fila["errores"]["ablacion"] = f"{type(exc).__name__}: {exc}"

    # --- Acuerdo entre las técnicas que atribuyen sobre lo mismo ------------
    acuerdo = {}
    nombres = list(palabras_por_tecnica)
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            a, b = nombres[i], nombres[j]
            acuerdo[f"{a}_vs_{b}"] = round(
                _jaccard(_cima(palabras_por_tecnica[a]), _cima(palabras_por_tecnica[b])), 4)
    fila["acuerdo_jaccard_sobre_palabras"] = acuerdo

    # --- Fidelidad, el criterio decisivo ------------------------------------
    for tecnica, pares in palabras_por_tecnica.items():
        if not pares:
            continue
        try:
            destino = {"atencion": "atencion", "shap": "shap_texto", "lime": "lime"}[tecnica]
            if destino in fila:
                fila[destino]["fidelidad"] = compute_faithfulness(
                    modelo, tok, entrada, pares, texto, target_class=clase,
                    max_token_length=cargado.config.max_token_length)
        except Exception as exc:
            fila["errores"][f"fidelidad_{tecnica}"] = f"{type(exc).__name__}: {exc}"

    return fila


def _promedio(filas: list[dict], camino: list[str]) -> float | None:
    valores = []
    for f in filas:
        nodo = f
        for clave in camino:
            if not isinstance(nodo, dict) or clave not in nodo:
                nodo = None
                break
            nodo = nodo[clave]
        if isinstance(nodo, (int, float)):
            valores.append(nodo)
    return round(float(np.mean(valores)), 6) if valores else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--muestra", type=int, default=30,
                   help="correos sobre los que se compara (LIME domina el coste)")
    p.add_argument("--corrida", default=carga.CORRIDA_DE_REFERENCIA)
    p.add_argument("--sin-contraste", action="store_true",
                   help="omite la evaluación completa que contrasta contra la cifra publicada")
    args = p.parse_args()

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"E8 · selección del esquema de explicabilidad (R3.1) en {dispositivo}")

    corpus = cargar_corpus()
    particion = particion_agrupada(corpus)
    cargado = carga.cargar(BASE, args.corrida, dispositivo=dispositivo)
    print(f"  punto de control: {args.corrida} "
          f"(paso {cargado.metadatos['global_step']}, época {cargado.metadatos['epoch']})")

    # El contraste contra la cifra publicada es lo que acredita que se explica el
    # modelo del capítulo y no otra corrida de la misma arquitectura.
    contraste = {"contrastado": False, "motivo": "omitido por --sin-contraste"}
    if not args.sin_contraste:
        prueba = corpus.loc[particion.prueba]
        lotes = carga.filas_como_lotes(cargado, prueba, dispositivo=dispositivo)
        y_true, y_pred, y_proba = [], [], []
        with torch.no_grad():
            for lote in lotes:
                etiqueta = int(lote["label"][0])
                pr = float(torch.softmax(cargado.modelo(lote_limpio(lote)), dim=-1)[0, 1])
                y_true.append(etiqueta); y_proba.append(pr); y_pred.append(int(pr >= 0.5))
        m = metricas(np.array(y_true), np.array(y_pred), np.array(y_proba))
        publicado = f1_reportado(BASE / "medios_de_verificacion" / "experimentos" / "e4" / "e4.json")
        punto, _ = carga.rutas_de(BASE, args.corrida)
        contraste = comprobar_contra_lo_reportado(m["f1"], publicado, punto)
        print(f"  contraste: F1 medido {m['f1']:.4f} frente a {publicado} publicado")

    filas_df = carga.muestra_de_positivos(BASE, corpus, args.muestra, corrida=args.corrida)
    print(f"  muestra: {len(filas_df)} correos clasificados como phishing")
    lotes = carga.filas_como_lotes(cargado, filas_df, dispositivo=dispositivo)

    resultados = []
    for i, lote in enumerate(lotes, 1):
        resultados.append(_sobre_un_correo(cargado, lote))
        if i % 5 == 0 or i == len(lotes):
            print(f"    {i}/{len(lotes)}")

    resumen = {
        "fidelidad_exhaustividad": {
            t: _promedio(resultados, [t, "fidelidad", "comprehensiveness"])
            for t in ("atencion", "shap_texto", "lime")},
        "fidelidad_suficiencia": {
            t: _promedio(resultados, [t, "fidelidad", "sufficiency"])
            for t in ("atencion", "shap_texto", "lime")},
        "coste_segundos": {
            t: _promedio(resultados, [t, "segundos"])
            for t in ("atencion", "shap_texto", "shap_tabular", "lime", "ablacion")},
        "estabilidad_jaccard": {
            t: _promedio(resultados, [t, "estabilidad_jaccard"]) for t in ("shap_texto", "lime")},
        "aditividad_error_absoluto": {
            t: _promedio(resultados, [t, "aditividad", "error_absoluto"])
            for t in ("shap_tabular", "ablacion")},
        "atencion_entropia_normalizada": _promedio(resultados, ["atencion", "entropia_normalizada"]),
        "atencion_peso_del_centinela": _promedio(resultados, ["atencion", "peso_del_centinela"]),
        "acuerdo_medio": {
            par: round(float(np.mean([r["acuerdo_jaccard_sobre_palabras"][par]
                                      for r in resultados
                                      if par in r.get("acuerdo_jaccard_sobre_palabras", {})])), 4)
            for par in {k for r in resultados for k in r.get("acuerdo_jaccard_sobre_palabras", {})}},
        "errores_por_tecnica": {
            t: sum(1 for r in resultados if t in r.get("errores", {}))
            for t in ("atencion", "shap_texto", "shap_tabular", "lime", "ablacion")},
        "alcance_modal": {
            "atencion": "texto y modalidades",
            "shap_texto": "solo texto",
            "shap_tabular": "solo modalidades",
            "lime": "solo texto",
            "ablacion": "solo modalidades",
        },
    }

    informe = {
        "experimento": "e8",
        "pregunta": ("Cual de las familias de explicabilidad describe con fidelidad lo que "
                     "el modelo multimodal usa para decidir, y con que alcance y coste"),
        "corrida_explicada": args.corrida,
        "punto_de_control": cargado.metadatos,
        "contraste_con_lo_reportado": contraste,
        "n_correos": len(resultados),
        "familias_comparadas": ["atencion intrinseca", "valores de Shapley mediante Captum",
                                "aproximaciones lineales locales"],
        "resumen": resumen,
        "por_correo": resultados,
    }
    emitir("e8", informe, corpus=corpus, particion=particion)
    print("  informe emitido")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
