"""
E9 · Módulo de interpretación semántica y validación de fidelidad (R3.2 y R3.3)

Dos resultados en un mismo guion porque el segundo se calcula sobre lo que
produce el primero, y separarlos obligaría a persistir las narrativas solo para
volver a leerlas.

**R3.2.** Su indicador exige extracción exitosa de pesos de importancia sobre la
muestra de prueba de predicciones positivas, y mapeo sintáctico sin excepciones
del cien por cien de esa muestra. De ahí que el guion recorra todas las
predicciones positivas y no una selección: el indicador se acredita por cobertura
completa, y una muestra parcial no lo acreditaría aunque saliera bien.

Cada explicación combina dos superficies, porque el reporte por modalidad que el
indicador pide no puede construirse con una sola. Las características tabulares se
atribuyen por ablación, que las mide sobre el propio modelo; las palabras del
mensaje, con la técnica de texto que E8 haya mostrado más fiel. La narrativa y el
reporte estructurado se delegan en `xai.narrative`, que ya traduce identificadores
técnicos a frases legibles y agrupa por modalidad.

**R3.3.** Su criterio primario es un panel de evaluadores humanos, y ese panel no
se simula. El guion deja preparados los tres ejemplos que el instrumento necesita
con explicaciones reales, y declara el panel como pendiente. Conviene decir por
qué de forma expresa: `doc/validacion_xai/script_scoring.py` puede generar cinco
evaluadores ficticios y guardar un fichero de resultados con la misma forma que
uno real, de modo que la única salvaguarda frente a presentar evidencia fabricada
es no producir ese fichero por ninguna vía automática.

Lo que sí se automatiza es el chequeo complementario de fidelidad narrativa, que
verifica que las cifras del texto generado corresponden a las del vector de
atribución que lo originó. Por omisión lo resuelve un verificador determinista por
reglas, no un modelo de lenguaje, y así queda declarado en el informe: no hay
credenciales de ningún proveedor en este entorno y atribuir a un modelo de
lenguaje una comprobación hecha con expresiones regulares sería falsear el medio.

    python scripts/experimentos/e9_narrativa_y_validacion.py [--muestra 200] [--tecnica lime]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import BASE, SALIDA, cargar_corpus, emitir, particion_agrupada  # noqa: E402

sys.path.insert(0, str(BASE / "src"))

from phishing_model.explicabilidad import carga  # noqa: E402
from phishing_model.explicabilidad.atribucion_modal import (  # noqa: E402
    atribucion_por_ablacion, lote_limpio,
)

TOP_PALABRAS = 5
TOP_CARACTERISTICAS = 5


def _tecnica_mas_fiel(por_defecto: str = "lime") -> str:
    """Lee de E8 qué técnica de texto resultó más fiel, si E8 ya se ejecutó.

    La exhaustividad manda: mide cuánto cae la predicción al retirar lo que la
    técnica declaró decisivo, que es la evidencia directa de que la explicación
    describe lo que el modelo usa. Si no hay informe de E8 se emplea el valor por
    omisión y el informe lo declara, en vez de aparentar una decisión que no se
    tomó con evidencia.
    """
    ruta = SALIDA / "e8" / "e8.json"
    if not ruta.exists():
        return por_defecto
    resumen = json.loads(ruta.read_text(encoding="utf-8")).get("resumen", {})
    exhaustividad = resumen.get("fidelidad_exhaustividad", {})
    candidatas = {k: v for k, v in exhaustividad.items()
                  if v is not None and k in ("lime", "shap_texto", "atencion")}
    if not candidatas:
        return por_defecto
    return max(candidatas, key=candidatas.get)


def _fusionar_subpalabras(tokens: list[str], pesos: list[float]) -> list[tuple[str, float]]:
    """Reagrupa los fragmentos de WordPiece en palabras, sumando sus pesos.

    La narrativa cita palabras al analista, y un fragmento como "##ificacion" no
    es una palabra: mostrarlo incumpliría el criterio de comprensibilidad que R3.2
    persigue.
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


def _parece_palabra(cadena: str) -> bool:
    """Si el factor citado es una palabra que un analista puede leer.

    Los correos arrastran fronteras MIME, identificadores codificados en base 64 y
    nombres de fuentes tipográficas, y el codificador los convierte en tokens como
    cualquier otro. Una cadena de cuarenta y ocho caracteres alfanuméricos sin
    vocales reconocibles no es evidencia de phishing para quien lee el reporte,
    aunque el modelo la haya atendido.

    Mide legibilidad y no pertinencia, y conviene no confundirlas: el nombre de
    una fuente tipográfica es perfectamente legible y no es evidencia de nada.

    Exigir que la cadena fuese solo alfabética rechazaba términos de seguridad
    como el del protocolo de autenticación de tarjetas, que llevan un dígito, y
    por tanto subestimaba la legibilidad precisamente donde el dominio la
    concentra. El discriminante que sí separa la basura de codificación es la
    longitud junto con la presencia de vocales: una cadena de cuarenta y ocho
    caracteres cae por larga, y una secuencia de consonantes sin vocal alguna cae
    por impronunciable, mientras un término alfanumérico corto se conserva.
    """
    if not (2 <= len(cadena) <= 20):
        return False
    limpia = cadena.replace("-", "").replace("'", "")
    letras = sum(1 for c in limpia if c.isalpha())
    if not limpia or letras / len(limpia) < 0.6:
        return False
    return any(v in cadena.lower() for v in "aeiouáéíóú")


def _palabras_de(tecnica: str, cargado, entrada, clase: int) -> list[tuple[str, float]]:
    from phishing_model.xai.attention_explainer import extract_text_saliency_rollout
    from phishing_model.xai.lime_explainer import explain_lime
    from phishing_model.xai.shap_explainer import explain_shap

    if tecnica == "atencion":
        # La propagación de la atención a través de las capas del codificador de
        # texto, que es la técnica que E8 midió como más fiel. Es además la más
        # barata de las tres por un orden de magnitud, lo que importa porque este
        # guion recorre el cien por cien de la muestra crítica y no una selección.
        rollout = extract_text_saliency_rollout(
            cargado.modelo.text_encoder, entrada["input_ids"], entrada["attention_mask"])[0]
        toks = cargado.tokenizador.convert_ids_to_tokens(entrada["input_ids"][0].tolist())
        valida = entrada["attention_mask"][0].bool().tolist()
        palabras = _fusionar_subpalabras(
            [t for t, m in zip(toks, valida) if m],
            [s for s, m in zip(rollout.tolist(), valida) if m])
        return sorted(palabras, key=lambda kv: -abs(kv[1]))[:TOP_PALABRAS]

    if tecnica == "lime":
        r = explain_lime(cargado.modelo, cargado.tokenizador, entrada,
                         target_class=clase, num_features=TOP_PALABRAS, num_samples=200)
        return list(zip(r["words"], r["attributions"]))[:TOP_PALABRAS]
    if tecnica == "shap_texto":
        r = explain_shap(cargado.modelo, cargado.tokenizador, entrada,
                         target_class=clase, n_steps=16)
        pares = [(t, a) for t, a in zip(r["tokens"], r["attributions"])
                 if t not in ("[CLS]", "[SEP]", "[PAD]")]
        return sorted(pares, key=lambda kv: -abs(kv[1]))[:TOP_PALABRAS]
    return []


def _explicar(cargado, lote, tecnica: str) -> dict:
    """Atribución, narrativa y reporte estructurado de un correo."""
    from phishing_model.xai.narrative import (
        ExplanationInput, build_modality_report, generate_modality_report, generate_narrative,
    )

    entrada = lote_limpio(lote)
    with torch.no_grad():
        probas = torch.softmax(cargado.modelo(entrada), dim=-1)[0]
    clase = int(probas.argmax())
    confianza = float(probas[clase])

    caracteristicas = atribucion_por_ablacion(cargado.modelo, entrada, clase=clase)
    mejores = sorted(caracteristicas.items(), key=lambda kv: -abs(kv[1]))[:TOP_CARACTERISTICAS]
    palabras = _palabras_de(tecnica, cargado, entrada, clase)

    entrada_narrativa = ExplanationInput(
        label=clase, confidence=confianza,
        top_factors=list(mejores) + list(palabras),
        # El nombre ha de ser el que `narrative.XAI_METHOD_DISPLAY` reconoce, o el
        # reporte citaría como técnica el identificador interno del guion.
        xai_method={"atencion": "attention", "lime": "lime"}.get(tecnica, "shap"),
    )
    estructurado = build_modality_report(entrada_narrativa)

    # El vector se entrega en la MISMA escala que el texto imprime, que es el peso
    # crudo multiplicado por cien. El reporte por modalidad usa otra: renormaliza
    # los factores para que sumen cien, de modo que un mismo factor aparece como
    # 3.6 en la narrativa y como 41.0 en el bloque. Entregar el segundo hacía que
    # el verificador marcase como incorrecta toda cifra del texto, y el chequeo
    # informaba un cuarenta por ciento de aciertos que no describía la narrativa
    # sino la discrepancia entre dos normalizaciones.
    en_porcentaje = {f["name"]: round(abs(f["raw_weight"]) * 100, 1)
                     for b in estructurado["modality_blocks"] for f in b["factors"]}

    # Cuántos de los factores que la narrativa cita son palabras legibles. El
    # indicador de R3.2 se satisface con que el mapeo no lance excepciones, pero
    # una explicación que cite un resto de codificación MIME o el nombre de una
    # fuente tipográfica cumple la letra y no el propósito. Se mide aquí para que
    # el panel humano de R3.3 no sea quien lo descubra.
    citados = [n for n, _ in sorted(entrada_narrativa.top_factors,
                                    key=lambda kv: -abs(kv[1]))[:entrada_narrativa.max_factors]]
    del_texto = [n for n in citados if n not in caracteristicas]
    legibles = [n for n in del_texto if _parece_palabra(n)]

    return {
        "clase_predicha": clase,
        "confianza": round(confianza, 6),
        "narrativa": generate_narrative(entrada_narrativa),
        "reporte_por_modalidad": generate_modality_report(entrada_narrativa),
        "estructurado": estructurado,
        "atribucion_en_porcentaje": en_porcentaje,
        "legibilidad": {
            "factores_citados": len(citados),
            "citados_del_texto": len(del_texto),
            "citados_legibles": len(legibles),
            "ilegibles": [n for n in del_texto if not _parece_palabra(n)],
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--muestra", type=int, default=None,
                   help="por omisión, el cien por cien de las predicciones positivas")
    p.add_argument("--tecnica", default=None, choices=["atencion", "lime", "shap_texto"])
    p.add_argument("--verificar", type=int, default=50,
                   help="cuántas narrativas pasan por el chequeo numérico de R3.3")
    p.add_argument("--corrida", default=carga.CORRIDA_DE_REFERENCIA)
    args = p.parse_args()

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"E9 · narrativa y validación (R3.2 y R3.3) en {dispositivo}")

    corpus = cargar_corpus()
    particion = particion_agrupada(corpus)
    cargado = carga.cargar(BASE, args.corrida, dispositivo=dispositivo)

    tecnica = args.tecnica or _tecnica_mas_fiel()
    elegida_por_evidencia = args.tecnica is None and (SALIDA / "e8" / "e8.json").exists()
    print(f"  técnica de texto: {tecnica} "
          f"({'elegida por la fidelidad medida en E8' if elegida_por_evidencia else 'por omisión'})")

    filas = carga.muestra_de_positivos(BASE, corpus, args.muestra, corrida=args.corrida)
    lotes = carga.filas_como_lotes(cargado, filas, dispositivo=dispositivo)
    print(f"  muestra crítica: {len(lotes)} correos clasificados como phishing")

    explicaciones, fallos = [], []
    for i, lote in enumerate(lotes, 1):
        try:
            explicaciones.append(_explicar(cargado, lote, tecnica))
        except Exception as exc:
            fallos.append({"indice": i, "error": f"{type(exc).__name__}: {exc}"})
        if i % 25 == 0 or i == len(lotes):
            print(f"    {i}/{len(lotes)}  fallos: {len(fallos)}")

    cobertura = len(explicaciones) / len(lotes) * 100 if lotes else 0.0

    # --- R3.3, chequeo numérico complementario ------------------------------
    from phishing_model.xai.fidelity_check import mock_rule_based_llm_call, run_fidelity_check_batch

    a_verificar = explicaciones[:args.verificar]
    verificacion = run_fidelity_check_batch(
        [(e["atribucion_en_porcentaje"], e["narrativa"]) for e in a_verificar],
        mock_rule_based_llm_call)
    verificacion.pop("results", None)

    # Los ejemplos que alimentan el instrumento del panel se eligen entre los que
    # citan solo factores legibles. No es seleccionar lo favorable: el instrumento
    # mide si el evaluador ENTIENDE la explicación, y presentarle una que cita una
    # cadena en base 64 mediría la calidad del corpus y no la del reporte. Cuántos
    # correos cumplen la condición se informa junto a los ejemplos, porque esa
    # proporción es el hallazgo y no puede quedar escondida tras la selección.
    ejemplos = {}
    positivos = [e for e in explicaciones if e["clase_predicha"] == 1]
    legibles = [e for e in positivos if not e["legibilidad"]["ilegibles"]]
    elegibles = legibles or positivos
    if elegibles:
        ejemplos["alta_confianza"] = max(elegibles, key=lambda e: e["confianza"])
        ejemplos["confianza_intermedia"] = min(elegibles, key=lambda e: abs(e["confianza"] - 0.6))
        ejemplos["seleccion"] = {
            "candidatos_totales": len(positivos),
            "candidatos_con_todos_los_factores_legibles": len(legibles),
            "se_relajo_el_criterio": not legibles,
        }

    informe = {
        "experimento": "e9",
        "pregunta": ("Puede el sistema traducir sus atribuciones a una justificacion "
                     "comprensible sobre el cien por cien de la muestra critica, y "
                     "corresponden las cifras del texto a las del vector que lo origina"),
        "corrida_explicada": args.corrida,
        "punto_de_control": cargado.metadatos,
        "tecnica_de_texto": tecnica,
        "tecnica_elegida_por_evidencia": elegida_por_evidencia,
        "r3_2": {
            "n_muestra_critica": len(lotes),
            "n_explicadas": len(explicaciones),
            "cobertura_porcentaje": round(cobertura, 4),
            "excepciones": fallos,
            "sin_excepciones": not fallos,
            "confianza_media": round(float(np.mean([e["confianza"] for e in explicaciones])), 6)
            if explicaciones else None,
            "modalidad_dominante": {
                m: sum(1 for e in explicaciones if e["estructurado"]["dominant_modality"] == m)
                for m in ("texto", "estructura", "red")},
            # La cobertura sin excepciones acredita el indicador, pero no que lo
            # explicado se entienda. Se informa por separado cuántos de los
            # factores citados son palabras legibles, porque el corpus arrastra
            # fronteras MIME y nombres de fuentes que el codificador trata como
            # cualquier otro token y que la narrativa acabaría citando al analista.
            "legibilidad": {
                "factores_del_texto_citados": sum(
                    e["legibilidad"]["citados_del_texto"] for e in explicaciones),
                "de_ellos_legibles": sum(
                    e["legibilidad"]["citados_legibles"] for e in explicaciones),
                "porcentaje_legible": round(
                    sum(e["legibilidad"]["citados_legibles"] for e in explicaciones)
                    / max(1, sum(e["legibilidad"]["citados_del_texto"] for e in explicaciones))
                    * 100, 2),
                "correos_con_algun_factor_ilegible": sum(
                    1 for e in explicaciones if e["legibilidad"]["ilegibles"]),
                "ejemplos_ilegibles": [x for e in explicaciones
                                       for x in e["legibilidad"]["ilegibles"]][:15],
            },
        },
        "r3_3": {
            "chequeo_numerico": {
                "verificador": "regla_determinista",
                "nota": ("No se empleo un modelo de lenguaje: este entorno no tiene "
                         "credenciales de ningun proveedor. Atribuir a un modelo de "
                         "lenguaje una comprobacion hecha con expresiones regulares "
                         "falsearia el medio de verificacion."),
                **verificacion,
            },
            "panel_humano": {
                "estado": "pendiente de aplicacion",
                "criterio": "meta del 90 por ciento y minimo del 80 por ciento de comprension",
                "instrumento": "doc/validacion_xai/instrumento_evaluacion.md",
                "nota": ("El panel requiere evaluadores reales. El guion de puntuacion "
                         "puede generar evaluadores ficticios y guardar un fichero con la "
                         "misma forma que uno real, de modo que no se invoca por ninguna "
                         "via automatica."),
            },
        },
        "ejemplos_para_el_instrumento": ejemplos,
        "muestra_de_explicaciones": explicaciones[:5],
    }
    emitir("e9", informe, corpus=corpus, particion=particion)
    print(f"  cobertura de R3.2: {cobertura:.2f} por ciento, excepciones: {len(fallos)}")
    print(f"  chequeo numerico: {verificacion.get('pct_correct')} por ciento correcto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
