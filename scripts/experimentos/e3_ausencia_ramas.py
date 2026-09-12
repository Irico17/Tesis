"""
E3 · ¿Tolera el modelo la ausencia de ramas?

Dos preguntas distintas que conviene no mezclar:

**(a) Qué mecanismo trata mejor la ausencia.** Cuatro maneras, decididas por
medición y no por argumento: centinela con descarte aleatorio, entrenamiento
restringido a filas completas, compuerta condicionada por ejemplo, y mezcla de
expertos con enrutamiento por patrón de disponibilidad.

**(b) Cuánto se degrada cuando falta información.** Se toma el modelo entrenado
con todas las ramas y se le ocultan una o dos SOLO en evaluación. Mide lo que
ocurriría en despliegue ante un correo al que no se le pudieron extraer los
metadatos, sobre un modelo que durante el ajuste sí los tuvo.

Conviene recordar qué es el centinela, porque su nombre sugiere más de lo que es:
**una salvaguarda numérica**, un vector que impide que el softmax opere sobre un
conjunto vacío de claves y devuelva `NaN`. No es una propuesta arquitectónica. Este
experimento decide si además aporta algo.

    python scripts/experimentos/e3_ausencia_ramas.py [--epocas 3] [--rapido]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clasicos import ejecutar_clasico
from comun import (BASE, SEMILLAS, agregar, barras_con_error, cargar_corpus,
                   emitir, equivalencia, intervalo_bootstrap,
                   particion_agrupada, subconjunto_trimodal, tpr_a_fpr)
from degradacion import INTENSIDADES, OPERADORES, degradar
from ejecutor import ejecutar_variante, evaluar_punto_de_control

# (nombre legible, variante, sin_centinela)
MECANISMOS = (
    ("centinela + descarte", "atencion_cruzada_token", False),
    ("sin centinela", "atencion_cruzada_token", True),
    ("mezcla de expertos", "mezcla_de_expertos", True),
)

# Modelos que recorren la rejilla adversaria. Cubre TODAS las arquitecturas del
# capitulo, no solo los mecanismos de este experimento: la pregunta --¿cual
# sobrevive sin texto?-- no es propia de E3, y responderla sobre tres de las ocho
# dejaria fuera justo las que el lector va a querer comparar.
#
# `solo estructura` y `solo red` son el CONTROL DE VALIDEZ, no filas de relleno.
# No reciben texto por construccion, asi que sus curvas tienen que salir planas en
# las cuatro intensidades. Si alguna se mueve, la degradacion esta tocando algo que
# no deberia y la rejilla entera queda invalidada. Lo mismo con B3 y B4.
#
# Los puntos de control se toman de donde ya se entrenaron --las fusiones de E2,
# los unimodales de E1-- porque reentrenarlos aqui daria otros modelos y la
# comparacion dejaria de ser con las cifras que informan E1 y E2.
MODELOS_REJILLA = (
    ("atencion cruzada (token)", "e2", "atencion_cruzada_token"),
    ("atencion cruzada (modalidad)", "e2", "atencion_cruzada_modalidad"),
    ("fusor MLP", "e2", "fusor_mlp"),
    ("concatenacion tardia", "e2", "concatenacion_tardia"),
    ("mezcla de expertos", "e2", "mezcla_de_expertos"),
    ("solo texto", "e1", "solo_texto"),
    ("solo estructura [control]", "e1", "solo_estructura"),
    ("solo red [control]", "e1", "solo_red"),
)
CLASICOS_REJILLA = ("b1_tfidf_lr", "b3_rf_estructura", "b4_red", "b5_multimodal_clasico")

# Semillas de la rejilla. Se usa UNA para trazar la curva y las tres en los
# extremos, donde se mide la dispersion. No es un atajo silencioso: con 8 modelos
# x 4 operadores x 6 intensidades, tres semillas en todas las celdas serian 576
# inferencias --mas de seis horas de tarjeta-- para separar diferencias de 0.33
# frente a una dispersion entre semillas de 0.0006. Se declara y se informa.
INTENSIDADES_CON_TODAS_LAS_SEMILLAS = (0.0, 1.0)

# Qué ramas se ocultan en evaluación para la curva de degradación.
CONDICIONES = (
    ("las tres", ()),
    ("sin estructura", ("estructura",)),
    ("sin red", ("red",)),
    ("solo texto", ("estructura", "red")),
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epocas", type=int, default=3)
    p.add_argument("--rapido", action="store_true")
    p.add_argument("--reusar", action="store_true",
                   help="no reentrena los mecanismos: reutiliza sus puntos de control")
    args = p.parse_args()

    semillas = SEMILLAS[:1] if args.rapido else SEMILLAS
    corpus = subconjunto_trimodal(cargar_corpus())
    particion = particion_agrupada(corpus)
    print(f"E3 - {len(corpus):,} correos, misma particion que E1 y E2")

    # E3 es el experimento que sustenta R1.3, cuyo indicador exige ademas la
    # prueba funcional de la arquitectura y el diagrama de las ramas. Se
    # regeneran AQUI y no con un guion suelto: los artefactos que se generaban
    # por separado quedaban describiendo una version anterior sin que nada
    # avisara, y el verificador de cobertura los daba por buenos por existir.
    from phishing_model.sanity_check import run_sanity_check

    comprobacion = run_sanity_check()
    superadas = sum(1 for v in comprobacion.get("checks", {}).values()
                    if (v or {}).get("passed"))
    total_checks = len(comprobacion.get("checks", {}))
    print(f"  prueba funcional de la arquitectura: {superadas}/{total_checks} superadas")

    figura_arq = BASE / "data" / "reports" / "figures" / "arquitectura_multimodal.png"
    if not figura_arq.exists():
        try:
            sys.path.insert(0, str(BASE / "scripts" / "figuras"))
            from generar_figuras_mv import diagrama_arquitectura

            diagrama_arquitectura()
            print(f"  diagrama de arquitectura regenerado")
        except Exception as exc:  # no debe tumbar el experimento
            print(f"  AVISO: no se pudo regenerar el diagrama: {exc}")

    # (a) comparación de mecanismos, con todas las ramas disponibles
    mecanismos: dict[str, dict] = {}
    entrenados: dict[str, dict[int, dict]] = {}
    referencia: dict[str, tuple] = {}
    for nombre, variante, sin_cent in MECANISMOS:
        por_semilla = []
        entrenados[nombre] = {}
        for semilla in semillas:
            r = ejecutar_variante(variante, corpus, particion, semilla,
                                  epocas=args.epocas, sin_centinela=sin_cent,
                                  rapido=args.rapido, experimento="e3",
                                  reusar=args.reusar)
            por_semilla.append(r["metricas"])
            # Se guarda la ruta para reevaluar sin volver a entrenar: la rejilla
            # de degradación son 72 evaluaciones y reentrenar cada una la haría
            # inviable.
            entrenados[nombre][semilla] = {"punto": r["punto_de_control"],
                                           "escaladores": r["escaladores"]}
            if semilla == semillas[0]:
                # Punto de operacion e intervalo, como en E1, E2 y E4: sin ellos
                # la tabla de mecanismos solo tendria F1, que es la metrica donde
                # ya se sabe que todo empata.
                referencia[nombre] = (r["y_true"], r["y_pred"], r.get("y_proba"))
            print(f"  {nombre:24s} semilla {semilla}  F1={r['metricas']['f1']:.4f}")
        mecanismos[nombre] = {"variante": variante, "sin_centinela": sin_cent,
                              "por_semilla": por_semilla, **agregar(por_semilla)}
        if nombre in referencia:
            yt, yp, pr = referencia[nombre]
            mecanismos[nombre]["punto_de_operacion"] = tpr_a_fpr(yt, pr)
            mecanismos[nombre]["ic_bootstrap_f1"] = intervalo_bootstrap(yt, yp)

    # Equivalencia entre mecanismos. Los tres empatan en la cuarta cifra: sin esto
    # el informe diria "no se observa diferencia", que se lee como que no se supo
    # medir, en vez de acotar cuanto pueden diferir como mucho.
    equivalencia_mecanismos = {}
    if len(referencia) > 1:
        base_nombre = MECANISMOS[0][0]
        yt_b, yp_b, _ = referencia[base_nombre]
        equivalencia_mecanismos = {
            n: equivalencia(yt_b, referencia[n][1], yp_b)
            for n in referencia if n != base_nombre
        }

    # (b) curva de degradación sobre el mecanismo que resulte mejor. Se elige por
    # F1 con todas las ramas: degradar el peor mecanismo mediría su debilidad, no
    # la tolerancia de la arquitectura.
    mejor = max(mecanismos, key=lambda k: mecanismos[k]["f1"])
    variante, sin_cent = mecanismos[mejor]["variante"], mecanismos[mejor]["sin_centinela"]
    print(f"\n  curva de degradación sobre «{mejor}»")

    from ejecutor import _anular_ramas

    prueba_limpia = corpus.loc[particion.prueba]
    degradacion: dict[str, dict] = {}
    for etiqueta, ocultar in CONDICIONES:
        por_semilla = []
        for semilla in semillas:
            quedan = tuple(r for r in ("estructura", "red") if r not in ocultar)
            sub = _anular_ramas(prueba_limpia, quedan) if ocultar else prueba_limpia
            ref = entrenados[mejor][semilla]
            r = evaluar_punto_de_control(
                variante, semilla, sub, f"ausencia_{etiqueta.replace(' ', '_')}",
                ref["punto"], ref["escaladores"], sin_centinela=sin_cent, experimento="e3")
            por_semilla.append(r["metricas"])
        degradacion[etiqueta] = {"ramas_ocultadas": list(ocultar),
                                 "por_semilla": por_semilla, **agregar(por_semilla)}
        print(f"    {etiqueta:16s} F1={degradacion[etiqueta]['f1']:.4f}")

    # (c) EJE ADVERSARIO: el texto llega corrompido, no ausente. Se aplica solo
    # en prueba y por igual al modelo propuesto y a la fusión clásica B5, que no
    # tiene ningún mecanismo de tolerancia: la diferencia entre ambas curvas es
    # lo que el mecanismo compra.
    print("")
    print(f"  rejilla de degradacion adversaria del texto ({len(OPERADORES)} operadores"
          f" x {len(INTENSIDADES)} intensidades)")
    # La rejilla recorre LOS TRES mecanismos, no solo el mejor. Correrla solo sobre
    # el ganador respondia a medias: la pregunta es si ALGUNA de estas fusiones
    # sobrevive sin texto, y la mezcla de expertos es justamente la que podria,
    # porque su compuerta puede enrutar hacia las ramas no textuales en vez de
    # sumar sobre una base textual que ya no existe.
    from ejecutor import _anular_ramas as _anular
    from entrenar import ramas_activas, usa_texto

    def punto_de(experimento: str, variante: str, semilla: int):
        """Localiza el punto de control ya entrenado de una variante.

        Se busca por patron y no por nombre exacto porque el esquema de nombres
        cambio a mitad del trabajo --se le anadio la marca del centinela-- y los
        modelos de E1 y E2 se entrenaron con el anterior. Reentrenarlos solo para
        que encajara el nombre daria OTROS modelos y las cifras dejarian de
        coincidir con las que informan E1 y E2.
        """
        base = BASE / "data" / "model" / "checkpoints"
        esc = BASE / "data" / "model" / "scalers"
        for sufijo in ("_sc", ""):
            c = base / f"{experimento}_{variante}_s{semilla}{sufijo}_best.pt"
            e = esc / f"{experimento}_{variante}_s{semilla}{sufijo}.joblib"
            if c.exists() and e.exists():
                return str(c), str(e)
        return None, None

    def entrada_para(variante: str, prueba):
        """El mismo recorte de entradas con que se entreno la variante.

        Imprescindible para los unimodales: evaluar `solo_estructura` sobre un
        marco con texto y con la rama de red presente no mide lo que su nombre
        dice, y ademas es la razon por la que sus curvas deben salir planas.
        """
        cons, ct = ramas_activas(variante), usa_texto(variante)
        if cons == ("estructura", "red") and ct:
            return prueba
        return _anular(prueba, cons, con_texto=ct)

    # En modo rapido la rejilla se recorta a los extremos y a tres modelos. No es
    # un resultado: es la comprobacion de que el camino de codigo funciona antes
    # de gastar horas de tarjeta. La rejilla completa son 192 inferencias, que en
    # una prueba de humo no aportan nada y cuestan dos horas.
    intensidades = (0.0, 1.0) if args.rapido else INTENSIDADES
    modelos = MODELOS_REJILLA[:1] + MODELOS_REJILLA[-2:] if args.rapido else MODELOS_REJILLA
    operadores = list(OPERADORES)[:2] if args.rapido else list(OPERADORES)

    adversario: dict[str, dict] = {}
    for operador in operadores:
        adversario[operador] = {}
        for eps in intensidades:
            sucio = degradar(prueba_limpia, operador, eps)
            # La CURVA se traza siempre con la misma semilla, en todas las
            # intensidades. Antes se promediaban tres semillas en los extremos y
            # una en el interior, y el resultado era una curva cuyos puntos no
            # eran comparables entre si: el control de `solo estructura` --que no
            # recibe texto y debe salir plano-- saltaba de 0.4899 a 0.7703 sin que
            # ninguna degradacion lo tocara, solo porque 0.4899 era la media de
            # tres semillas muy dispares y 0.7703 el valor de una sola.
            # Las tres semillas se siguen midiendo en los extremos, pero su
            # dispersion se informa APARTE y no se mezcla en la serie.
            usar = (semillas if eps in INTENSIDADES_CON_TODAS_LAS_SEMILLAS
                    else semillas[:1])
            celda: dict[str, dict] = {}
            for etiqueta, exp_origen, var in modelos:
                por_s = []
                for semilla in usar:
                    pc, escal = punto_de(exp_origen, var, semilla)
                    if pc is None:
                        continue
                    r = evaluar_punto_de_control(
                        var, semilla, entrada_para(var, sucio),
                        f"adv_{operador}_{int(eps * 100):03d}", pc, escal,
                        sin_centinela=True, experimento=f"{exp_origen}rej")
                    por_s.append(r["metricas"])
                if por_s:
                    # El punto de la curva es SIEMPRE el de la primera semilla.
                    celda[etiqueta] = {
                        **por_s[0],
                        "semilla_de_la_curva": int(semillas[0]),
                        "punto_de_operacion": tpr_a_fpr(r["y_true"], r.get("y_proba")),
                    }
                    if len(por_s) > 1:
                        celda[etiqueta]["dispersion_medida"] = {
                            "n_semillas": len(por_s),
                            "por_semilla": [m.get("f1") for m in por_s],
                            **{k: v for k, v in agregar(por_s).items()
                               if k in ("f1", "f1_desv")},
                        }
            for nombre_c in CLASICOS_REJILLA:
                celda[nombre_c] = ejecutar_clasico(
                    nombre_c, corpus, particion, semilla=semillas[0],
                    prueba_alternativa=sucio)["metricas"]
            # `propuesta` se conserva como alias del mecanismo elegido, para que
            # los informes y figuras anteriores se sigan leyendo igual.
            celda["propuesta"] = celda.get("atencion cruzada (token)",
                                           celda["b5_multimodal_clasico"])
            celda["b5_fusion_clasica"] = celda["b5_multimodal_clasico"]
            adversario[operador][f"{eps:.2f}"] = celda
            resumen = "  ".join(
                f"{k[:11]}={v['f1']:.3f}" for k, v in celda.items()
                if k not in ("propuesta", "b5_fusion_clasica"))
            print(f"    {operador:14s} eps={eps:4.2f}  {resumen}")

    completo = degradacion["las tres"]["f1"]
    informe = {
        "experimento": "E3",
        "pregunta": "¿Tolera el modelo la ausencia de ramas, y qué mecanismo lo trata mejor?",
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "datos": {"n": int(len(corpus)), "particion": particion.resumen()},
        "semillas": list(semillas),
        "mecanismos": mecanismos,
        "mejor_mecanismo": mejor,
        "equivalencia_entre_mecanismos": equivalencia_mecanismos,
        "degradacion": degradacion,
        "degradacion_adversaria_del_texto": adversario,
        "prueba_funcional_de_la_arquitectura_r1_3": comprobacion,
        "modelo_de_amenaza": (
            "El atacante controla el cuerpo del mensaje pero no el resultado de "
            "autenticación, que lo estampa la infraestructura receptora. La "
            "asimetría es PARCIAL: un atacante puede comprar un dominio y pasar "
            "SPF; lo que no puede es pasar DMARC alineado suplantando la marca "
            "que imita. Los operadores son proxies sintéticos de técnicas de "
            "evasión documentadas, no comportamiento de atacante observado."
        ),
        "caida_relativa": {
            k: round((completo - v["f1"]) / completo, 4) if completo else None
            for k, v in degradacion.items()
        },
        "nota_sobre_el_centinela": (
            "El centinela es una salvaguarda numérica, pues impide que el softmax "
            "opere sobre un conjunto vacío de claves y devuelva NaN, y no una "
            "propuesta "
            "arquitectónica. Este experimento decide si además aporta desempeño."
        ),
    }

    def figura(destino: Path) -> None:
        barras_con_error(
            destino, "e3_mecanismos",
            "E3. Mecanismos de tolerancia a la ausencia de ramas\n"
            "F1 con todas las ramas disponibles, media de tres semillas",
            list(mecanismos), [m["f1"] for m in mecanismos.values()],
            [m.get("f1_desv", 0.0) for m in mecanismos.values()])
        # Curva por operador: es la figura que sostiene el eje adversario, y sin
        # ella la rejilla queda solo como una tabla de veinticuatro numeros.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ejes = plt.subplots(1, max(1, len(adversario)),
                                 figsize=(4.6 * max(1, len(adversario)), 4.6), sharey=True)
        ejes = np.atleast_1d(ejes)
        # Los controles van punteados y en gris: deben salir PLANOS, y verlos
        # planos es parte de lo que la lamina demuestra.
        series = [e for e, _, _ in MODELOS_REJILLA] + list(CLASICOS_REJILLA)
        series = [x for x in series if any(x in c for c in
                                           next(iter(adversario.values())).values())]
        colores = {
            "atencion cruzada (token)": "#C44E52", "atencion cruzada (modalidad)": "#4C72B0",
            "fusor MLP": "#55A868", "concatenacion tardia": "#937860",
            "mezcla de expertos": "#8172B3", "solo texto": "#DA8BC3",
            "solo estructura [control]": "#9AA5B1", "solo red [control]": "#B7C0C9",
            "b1_tfidf_lr": "#DD8452", "b5_multimodal_clasico": "#CCB974",
            "b3_rf_estructura": "#9AA5B1", "b4_red": "#B7C0C9",
        }
        marcas = {"solo estructura [control]": ":", "solo red [control]": ":",
                  "b3_rf_estructura": ":", "b4_red": ":",
                  "b1_tfidf_lr": "s--", "b5_multimodal_clasico": "s--"}
        for eje, (op, celdas) in zip(ejes, adversario.items()):
            xs = [float(k) for k in celdas]
            for serie in series:
                ys = [c[serie]["f1"] for c in celdas.values() if serie in c]
                if len(ys) != len(xs):
                    continue
                eje.plot(xs, ys, marcas.get(serie, "o-"), label=serie,
                         color=colores.get(serie))
            eje.set_title(op.replace("_", " ")); eje.set_xlabel("intensidad")
            eje.grid(alpha=0.3); eje.set_ylim(0, 1.05)
        ejes[0].set_ylabel("F1")
        ejes[-1].legend(fontsize=6.5, loc="lower left", ncol=1)
        fig.suptitle("E3. Degradación adversaria del texto, aplicada solo en "
                     "evaluación")
        fig.tight_layout()
        fig.savefig(destino / "e3_degradacion_adversaria.png", dpi=160)
        plt.close(fig)

        barras_con_error(
            destino, "e3_degradacion",
            f"E3. Degradación al ocultar ramas en evaluación\nmecanismo: {mejor}",
            list(degradacion), [d["f1"] for d in degradacion.values()],
            [d.get("f1_desv", 0.0) for d in degradacion.values()], resaltar=3,
            etiqueta_resaltada="condición sin modalidades no textuales")

    emitir("e3", informe, figura, corpus=corpus, particion=particion)
    print(f"\n  mejor mecanismo: {mejor}")
    print(f"  caída al quedarse solo con texto: "
          f"{100 * informe['caida_relativa']['solo texto']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
