"""
Arma la carpeta de medios de verificación, una por resultado comprometido.

La tabla de la matriz de objetivos declara, para cada resultado, un medio de
verificación y un indicador objetivamente verificable. Este guion recorre esa
tabla y deja en `medios_de_verificacion/` una carpeta por resultado con lo que
ese resultado exige, más una nota que enuncia el medio y el indicador literales
y dice qué artefacto acredita cada parte.

La copia se hace desde `medios_de_verificacion/experimentos/`, que es la salida
cruda de la cola y la única fuente: las carpetas por resultado se reconstruyen
enteras en cada pasada, de modo que no pueden quedar describiendo una ejecución
distinta de la vigente. Las figuras que produce `scripts/figuras/` se escriben
directamente en su carpeta de resultado y no se tocan aquí: la limpieza previa
solo alcanza a los ficheros que esta misma pasada vuelve a poner, nombre por
nombre.

Si falta un artefacto, se dice cuál y el guion termina con código distinto de
cero: un medio de verificación incompleto que no avisa es peor que uno ausente.

    python scripts/experimentos/consolidar_iov.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comun import BASE  # noqa: E402

MV = BASE / "medios_de_verificacion"
EXP = MV / "experimentos"
R1_4 = MV / "R1.4_entrenamiento"

# Carpetas enteras que la cola escribe en su directorio de trabajo y que el medio
# de verificacion de R1.4 pide de forma literal: «logs de ejecucion y curvas de
# aprendizaje». Se trasladan, no se copian: duplicar trece megabytes en el
# repositorio no aporta nada y las dos copias acaban divergiendo.
CARPETAS = [
    (BASE / "data" / "reports" / "training_logs", R1_4 / "historiales", "*_history.json"),
    (BASE / "data" / "reports" / "experimentos", R1_4 / "registros", "*"),
]

# Los medios de verificación y los indicadores se transcriben LITERALMENTE de la
# matriz de objetivos del plan de tesis. No se parafrasean: el jurado contrasta
# contra ese texto y una paráfrasis convierte la comprobación en interpretación.
RESULTADOS: dict[str, dict] = {
    "R1.1_corpus_multimodal": {
        "titulo": "Dataset multimodal consolidado",
        "medio": ("Repositorio digital documentado con estructura clara de datos "
                  "crudos y procesados. Informe descriptivo del tratamiento de los "
                  "datasets."),
        "indicador": (
            "Corpus consolidado en el que al menos el 25% de las muestras integra "
            "simultáneamente texto, estructura HTML/DOM y metadatos de red, y al "
            "menos el 40% integra texto y al menos una modalidad no textual. "
            "Prevalencia de la clase positiva entre 0.40 y 0.60, tanto en el corpus "
            "como en el subconjunto de dos modalidades. Formato estándar (.parquet, "
            ".csv o .json) listo para ingesta. Prueba de humo automatizada que valide "
            "la integridad del corpus, incluida la ausencia de fuga por "
            "disponibilidad de campo."),
        "artefactos": [
            (EXP / "e0" / "e0.json", "e0_corpus_y_prueba_de_humo.json",
             "Criterios del indicador contrastados uno a uno y prueba de humo"),
            (EXP / "e0" / "e0_cobertura_modal.png", "cobertura_modal_por_coleccion.png",
             "Disponibilidad de cada modalidad en cada colección"),
            (BASE / "data" / "reports" / "corpus_real_report.json",
             "informe_descriptivo_del_corpus.json",
             "Informe descriptivo del tratamiento de los datasets"),
            (BASE / "data" / "reports" / "listas_correo_viabilidad.json",
             "viabilidad_de_las_listas_de_correo.json",
             "Medición que justifica incorporar archivos de listas de discusión"),
        ],
    },
    "R1.2_pipeline_de_procesamiento": {
        "titulo": "Pipeline de procesamiento estandarizado",
        "medio": "Código fuente y notebook de Jupyter/Colab (.ipynb) documentado.",
        "indicador": ("Ejecución sin errores del script de limpieza y extracción de "
                      "características del 90% de las muestras del dataset."),
        "artefactos": [
            (EXP / "e0" / "e0.json", "e0_cobertura_del_procesamiento.json",
             "Cobertura del procesamiento y comprobaciones de integridad"),
            (EXP / "e5" / "e5.json", "e5_validez_y_fugas.json",
             "Auditoría de qué mide cada característica y suelo de procedencia"),
            (BASE / "notebooks" / "pipeline_r1_1_r1_2.ipynb",
             "pipeline_r1_1_r1_2.ipynb",
             "Notebook documentado que recorre el pipeline de extremo a extremo"),
        ],
    },
    "R1.3_arquitectura": {
        "titulo": "Arquitectura funcional de atención cruzada",
        "medio": ("Informe descriptivo de la arquitectura. Diagrama de arquitectura "
                  "del modelo y código fuente de la red neuronal."),
        "indicador": ("Implementación en código (PyTorch/TensorFlow) de al menos dos "
                      "ramas de extracción modal y una capa de fusión jerárquica o "
                      "atención cruzada documentada. Prueba funcional de "
                      "entrenamiento."),
        "artefactos": [
            (BASE / "data" / "reports" / "model_sanity_check.json",
             "prueba_funcional_de_la_arquitectura.json",
             "Prueba funcional: formas, gradiente, aislamiento y recarga"),
            (EXP / "e3" / "e3.json", "e3_tolerancia_y_degradacion.json",
             "Tolerancia a la ausencia de modalidades y degradación adversaria"),
            (EXP / "e3" / "e3_degradacion.png", "degradacion_por_ausencia.png",
             "Desempeño al retirar modalidades en evaluación"),
            (EXP / "e3" / "e3_degradacion_adversaria.png", "degradacion_adversaria.png",
             "Curvas por operador de evasión e intensidad"),
            (BASE / "doc" / "ARQUITECTURA_EXPLICADA.md",
             "INFORME_DE_LA_ARQUITECTURA.md",
             "Informe descriptivo de la arquitectura"),
        ],
        "codigo": ["src/phishing_model/model.py",
                   "src/phishing_model/fusion/cross_attention.py",
                   "src/phishing_model/fusion/mixture_of_experts.py",
                   "src/phishing_model/fusion/modality_encoder.py",
                   "src/phishing_model/encoders/"],
    },
    "R1.4_entrenamiento": {
        "titulo": "Pipeline de entrenamiento automatizado",
        "medio": ("Scripts de entrenamiento automatizados. Logs de ejecución y curvas "
                  "de aprendizaje (Loss/Accuracy)."),
        "indicador": (
            "Pipeline automatizado que demuestre convergencia matemática y ausencia de "
            "sobreajuste crítico. Evaluación empírica, durante el entrenamiento, de la "
            "necesidad de aplicar ponderación de clases, con base en el comportamiento "
            "de la exhaustividad de la clase minoritaria. Esta decisión se documenta "
            "con evidencia experimental y no se asume a priori."),
        "artefactos": [
            (EXP / "e4" / "e4.json", "e4_generalizacion_y_ponderacion.json",
             "Curva de aprendizaje, protocolo de partición y decisión de ponderación"),
            (EXP / "e4" / "e4_curva_aprendizaje.png", "curva_por_cantidad_de_datos.png",
             "Desempeño en función de la cantidad de datos de entrenamiento"),
            (EXP / "e1" / "e1.json", "e1_multimodalidad.json",
             "Corridas de las que proceden los historiales de entrenamiento"),
        ],
        "generadas": {
            "historiales/": "Historial de cada corrida: pérdida por paso y validación por época",
            "registros/": "Registro de ejecución de la cola completa",
            "curvas_de_aprendizaje/": "Curvas de pérdida y de exactitud por corrida",
        },
        "codigo": ["scripts/experimentos/cola_servidor.sh",
                   "scripts/experimentos/entrenar.py",
                   "scripts/experimentos/ejecutor.py",
                   "src/phishing_model/train.py"],
    },
    "R2.1_lineas_base": {
        "titulo": "Selección e implementación de líneas base unimodales",
        "medio": "Informe de selección y código de implementación.",
        "indicador": ("Justificación e implementación de al menos dos enfoques "
                      "unimodales funcionales como líneas base."),
        "artefactos": [
            (EXP / "e1" / "e1.json", "e1_unimodales_y_piso_clasico.json",
             "Desempeño de los tres unimodales neuronales y las cuatro líneas base"),
            (EXP / "e2" / "e2.json", "e2_mecanismos_de_fusion.json",
             "Las mismas referencias bajo cada mecanismo de fusión"),
        ],
        "codigo": ["scripts/experimentos/clasicos.py",
                   "src/phishing_baseline/"],
    },
    "R2.2_comparacion": {
        "titulo": "Pipeline de comparación de modelos",
        "medio": ("Informe técnico y gráficos de rendimiento (curvas ROC, matriz de "
                  "confusión). Reporte de métricas comparativas."),
        "indicador": (
            "Cuadro comparativo que cuantifique y caracterice las diferencias en "
            "exactitud, precisión, exhaustividad, F1 y ROC-AUC entre el modelo "
            "propuesto y las líneas base unimodales, acompañado de pruebas de "
            "significancia estadística (McNemar o t de Student) que determinen si las "
            "diferencias observadas son estadísticamente significativas (p < 0.05) o "
            "producto de varianza aleatoria."),
        "artefactos": [
            (EXP / "e4" / "e4.json", "cuadro_comparativo_corpus_completo.json",
             "Cuadro comparativo de las doce arquitecturas con sus pruebas"),
            (EXP / "e4" / "e4_cuadro_comparativo.png", "cuadro_comparativo.png",
             "Desempeño de todas las arquitecturas sobre el corpus completo"),
            (EXP / "e1" / "e1.json", "e1_aporte_de_la_multimodalidad.json",
             "Contraste de McNemar frente al mejor unimodal neuronal"),
            (EXP / "e1" / "e1_multimodalidad.png", "aporte_de_la_multimodalidad.png",
             "Comparación sobre el subconjunto con las tres modalidades"),
            (EXP / "e2" / "e2.json", "e2_equivalencia_entre_mecanismos.json",
             "Pruebas de equivalencia entre mecanismos de fusión"),
            (EXP / "e2" / "e2_mecanismo_fusion.png", "mecanismos_de_fusion.png",
             "Desempeño por mecanismo de fusión"),
            (EXP / "e5" / "e5.json", "e5_validez_de_la_comparacion.json",
             "Qué parte de lo medido corresponde al fenómeno y qué a la procedencia"),
            (EXP / "e7" / "e7.json", "e7_codificador_multilingue.json",
             "Contraste entre codificador monolingüe y multilingüe, por idioma"),
            (EXP / "e7" / "e7_codificador.png", "codificador_multilingue.png",
             "Desempeño de cada codificador sobre los subconjuntos por idioma"),
        ],
        "generadas": {
            "matrices_de_confusion/": "Matriz de confusión de cada uno de los doce modelos",
            "curvas_roc_*.png": "Curvas ROC por familia y en la región de operación",
        },
    },
    "R2.3_modelo_final": {
        "titulo": "Modelo final optimizado y validado",
        "medio": ("Informe final junto a archivo de pesos algorítmicos exportados y "
                  "reporte de latencia."),
        "indicador": (
            "El modelo optimizado es funcional para realizar inferencias sobre datos "
            "nuevos no vistos, y reporta su tasa de falsos negativos observada en el "
            "conjunto de prueba, sin presuponer una reducción respecto de la línea "
            "base, conforme al enfoque de caracterización adoptado en R2.2."),
        "artefactos": [
            (EXP / "e6" / "e6.json", "e6_latencia_y_falsos_negativos.json",
             "Tamaño, latencia media y percentil 95, acuerdo de cuantización y "
             "tasa de falsos negativos de cada arquitectura"),
            (R1_4 / "registros" / "e6_modelo_optimizado.log",
             "registro_de_la_medicion.log",
             "Registro de la exportación, la cuantización y la medición"),
            (BASE / "data" / "model" / "onnx" / "atencion_cruzada_token"
             / "quantization_latency_report.json",
             "informe_de_exportacion_de_la_propuesta.json",
             "Exportación y cuantización de la arquitectura de la que habla el "
             "capítulo, con la ruta de los pesos resultantes"),
        ],
        "no_versionados": {
            "data/model/onnx/atencion_cruzada_token/model_int8.onnx":
                "Pesos exportados y cuantizados de la arquitectura propuesta, unos "
                "66 MB. No se versionan por tamaño; se regeneran con E6 y su huella "
                "queda en el informe de exportación.",
            "data/model/onnx/atencion_cruzada_token/model_fp32.onnx":
                "Los mismos pesos en coma flotante de 32 bits, unos 260 MB.",
        },
        "codigo": ["src/phishing_model/quantization.py",
                   "scripts/experimentos/e6_modelo_optimizado.py"],
    },
}

PENDIENTES = {
    "R3.1": "Selección del esquema técnico de interpretabilidad",
    "R3.2": "Módulo XAI integrado operativamente",
    "R3.3": "Reporte de validación de transparencia operativa",
}


def _nota(carpeta: Path, clave: str, spec: dict, copiados: list, ausentes: list) -> None:
    """La nota que encabeza cada carpeta, con el medio y el indicador literales."""
    lineas = [
        f"# {clave.split('_')[0]}: {spec['titulo']}",
        "",
        "## Medio de verificación comprometido",
        "",
        f"> {spec['medio']}",
        "",
        "## Indicador objetivamente verificable",
        "",
        f"> {spec['indicador']}",
        "",
        "## Qué contiene esta carpeta",
        "",
        "| Artefacto | Qué acredita |",
        "| --- | --- |",
    ]
    for destino, para_que in copiados:
        lineas.append(f"| `{destino}` | {para_que} |")
    for sub, para_que in (spec.get("generadas") or {}).items():
        lineas.append(f"| `{sub}` | {para_que} |")
    if spec.get("no_versionados"):
        lineas += ["", "## Artefactos que no se versionan por tamaño", ""]
        for ruta, para_que in spec["no_versionados"].items():
            lineas.append(f"- `{ruta}`: {para_que}")
    if spec.get("codigo"):
        lineas += ["", "## Código fuente que sustenta el resultado", ""]
        lineas += [f"- `{c}`" for c in spec["codigo"]]
    if ausentes:
        lineas += ["", "## Artefactos que faltan", ""]
        lineas += [f"- `{a}`" for a in ausentes]
    lineas += [
        "",
        "---",
        "",
        "Los artefactos de esta carpeta se copian desde "
        "`medios_de_verificacion/experimentos/`, que es la salida de la cola de "
        "experimentos, ejecutándola y después "
        "`python scripts/experimentos/consolidar_iov.py`.",
        "",
    ]
    (carpeta / "INDICADOR_Y_MEDIO_DE_VERIFICACION.md").write_text(
        "\n".join(lineas), encoding="utf-8")


def _indice(estado: dict) -> None:
    """El índice de la carpeta raíz, que es por donde entra quien revisa."""
    lineas = [
        "# Medios de verificación",
        "",
        "Una carpeta por resultado comprometido en la matriz de objetivos. Cada una "
        "contiene el medio de verificación que ese resultado exige y una nota que "
        "enuncia el indicador y dice qué artefacto acredita cada parte.",
        "",
        "| Resultado | Carpeta | Artefactos | Estado |",
        "| --- | --- | --- | --- |",
    ]
    for clave, datos in estado.items():
        carpeta = datos["carpeta"]
        n = len(datos["presentes"])
        total = n + len(datos["ausentes"])
        marca = "completo" if datos["cubierto"] else f"faltan {len(datos['ausentes'])}"
        lineas.append(f"| {clave.split('_')[0]} | [`{carpeta}/`]({carpeta}/) "
                      f"| {n} de {total} | {marca} |")
    for clave, titulo in PENDIENTES.items():
        lineas.append(f"| {clave} | — | — | pendiente, fase siguiente |")
    lineas += [
        "",
        "## Salida cruda de los experimentos",
        "",
        "`experimentos/` conserva lo que escribe la cola tal cual: un informe en "
        "formato JSON por experimento con su procedencia (huella del corpus y de la "
        "partición, semillas, versiones y unidad de procesamiento), las figuras que "
        "cada uno emite, los historiales de entrenamiento y los registros de "
        "ejecución. Las carpetas por resultado se construyen desde ahí y pueden "
        "reconstruirse en cualquier momento.",
        "",
        "## Cómo regenerarlo todo",
        "",
        "```bash",
        "bash scripts/experimentos/cola_servidor.sh",
        "python scripts/figuras/generar_figuras_mv.py",
        "python scripts/experimentos/consolidar_iov.py",
        "```",
        "",
    ]
    (MV / "README.md").write_text("\n".join(lineas), encoding="utf-8")


def _sincronizar_carpetas() -> None:
    """Traslada a la carpeta del resultado lo que la cola dejó en su directorio."""
    for origen, destino, patron in CARPETAS:
        if not origen.exists():
            continue
        destino.mkdir(parents=True, exist_ok=True)
        movidos = 0
        for f in sorted(origen.glob(patron)):
            if f.is_file():
                shutil.move(str(f), destino / f.name)
                movidos += 1
        if movidos:
            print(f"  {movidos} ficheros de {origen.relative_to(BASE)} "
                  f"-> {destino.relative_to(BASE)}")


def main() -> int:
    estado: dict[str, dict] = {}
    faltan_todos: list[str] = []

    _sincronizar_carpetas()

    for clave, spec in RESULTADOS.items():
        carpeta = MV / clave
        carpeta.mkdir(parents=True, exist_ok=True)
        # Se retiran SOLO las copias que esta funcion pone, nombre por nombre. La
        # version anterior borraba todo fichero de la carpeta, y con ello las
        # figuras que `scripts/figuras/generar_figuras_mv.py` escribe ahi
        # directamente: el diagrama de la arquitectura de R1.3 y las tres curvas
        # ROC de R2.2 desaparecian en cada consolidacion, pese a que el propio
        # encabezado de este modulo dice que no se tocan. Las subcarpetas se
        # salvaban por accidente, porque el borrado solo alcanzaba a ficheros.
        propios = {destino for _, destino, _ in spec["artefactos"]}
        propios.add("INDICADOR_Y_MEDIO_DE_VERIFICACION.md")
        for viejo in carpeta.glob("*"):
            if viejo.is_file() and viejo.name in propios:
                viejo.unlink()

        copiados, ausentes = [], []
        for origen, destino, para_que in spec["artefactos"]:
            if origen.exists():
                shutil.copy2(origen, carpeta / destino)
                copiados.append((destino, para_que))
            else:
                ausentes.append(origen.relative_to(BASE).as_posix())

        _nota(carpeta, clave, spec, copiados, ausentes)
        estado[clave] = {
            "carpeta": clave,
            "titulo": spec["titulo"],
            "medio_de_verificacion": spec["medio"],
            "indicador": spec["indicador"],
            "presentes": [d for d, _ in copiados],
            "ausentes": ausentes,
            "cubierto": not ausentes,
        }
        faltan_todos += [f"{clave}: {a}" for a in ausentes]

    _indice(estado)

    informe = {
        "generado_en": datetime.now(timezone.utc).isoformat(),
        "resultados": estado,
        "pendientes": PENDIENTES,
        "cubiertos": sum(1 for d in estado.values() if d["cubierto"]),
        "total": len(estado),
        "faltantes": faltan_todos,
    }
    (MV / "cobertura_iov.json").write_text(
        json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")

    print("MEDIOS DE VERIFICACIÓN POR RESULTADO")
    for clave, d in estado.items():
        marca = "OK  " if d["cubierto"] else "FALTA"
        print(f"  [{marca}] {clave:34s} {len(d['presentes'])} artefactos")
    for a in faltan_todos:
        print(f"     falta: {a}")
    print(f"\n{informe['cubiertos']}/{informe['total']} resultados con su medio completo")
    print(f"Índice en {(MV / 'README.md').relative_to(BASE)}")
    return 0 if not faltan_todos else 1


if __name__ == "__main__":
    raise SystemExit(main())
