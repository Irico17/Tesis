# Detección multimodal de phishing en correo electrónico

Código y evidencia de la tesis de licenciatura en Ingeniería Informática de
Cristhofer Alegre (PUCP), asesorada por el Dr. Edwin Villanueva.

El trabajo propone una arquitectura que analiza simultáneamente tres modalidades
del mensaje —el texto, la estructura del cuerpo en HTML y los metadatos de
tránsito y autenticación— y las combina mediante fusión jerárquica con atención
cruzada. Su objeto no es demostrar superioridad sobre un enfoque unimodal, sino
**caracterizar** el comportamiento de esa arquitectura y de las alternativas con
las que compite, incluidas líneas base clásicas, bajo un protocolo declarado de
antemano.

Este repositorio contiene lo necesario para reproducir esos resultados y la
evidencia que los sustenta. El documento de la tesis se redacta aparte.

---

## Estado

| Objetivo | Resultados | Estado |
| --- | --- | --- |
| 1. Diseñar e implementar el modelo multimodal | R1.1 a R1.4 | reportados |
| 2. Evaluar el desempeño frente a enfoques unimodales | R2.1 a R2.3 | reportados |
| 3. Integrar técnicas de IA explicable | R3.1 a R3.3 | fase siguiente |

La evidencia de cada resultado está en [`medios_de_verificacion/`](medios_de_verificacion/),
una carpeta por resultado, con una nota que enuncia el medio de verificación y el
indicador comprometidos y dice qué artefacto acredita cada parte.

---

## Estructura

```
src/phishing_pipeline/     construcción del corpus: descarga, limpieza, extracción
                           de las tres modalidades, deduplicación y partición
src/phishing_model/        la arquitectura: ramas de extracción, mecanismos de
                           fusión, entrenamiento, evaluación y cuantización
src/phishing_baseline/     líneas base clásicas B1, B3, B4 y B5

scripts/experimentos/      la cola de siete experimentos, de E0 a E6
scripts/figuras/           figuras que exigen los medios de verificación
scripts/prueba_integral.py prueba de integración de la cadena completa
notebooks/                 recorrido documentado del pipeline de datos

medios_de_verificacion/    evidencia, una carpeta por resultado comprometido
doc/                       informe de la arquitectura, pre-registro de hipótesis
                           y guía de lectura de los resultados
data/                      artefactos de trabajo; no se versionan
```

El corpus (589 MB), los puntos de control (74 GB) y las predicciones fila por
fila no se versionan: están muy por encima del límite por archivo de la
plataforma y se reconstruyen de forma íntegra con los comandos de abajo. La
huella criptográfica del corpus queda registrada en la procedencia de cada
informe, de modo que puede comprobarse que una reconstrucción coincide con la que
produjo los resultados publicados.

---

## Puesta en marcha

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # requirements.lock.txt fija las versiones exactas
export PYTHONPATH=src
```

Kaggle exige credenciales propias: colocar `kaggle.json` en `~/.kaggle/`.

## Reconstruir el corpus

Ocho colecciones públicas. Cinco se ingieren en formato de mensaje original
(`.eml` y buzones `mbox`), que es de donde procede toda la cobertura de las
modalidades no textuales; tres se publican en formato tabular y aportan solo
texto.

```bash
python -m phishing_pipeline.downloaders.correo_real      # phishing_pot, Nazario, SpamAssassin
python -m phishing_pipeline.downloaders.listas_correo    # Fedora, kernel_lists
python -m phishing_pipeline.downloaders.kaggle_phishing  # Kaggle
python -m phishing_pipeline.downloaders.phish_mmf        # CEAS_08, datacon2023
python -m phishing_pipeline.corpus_real                  # Dataset_Real.parquet
```

El cuaderno [`notebooks/pipeline_r1_1_r1_2.ipynb`](notebooks/pipeline_r1_1_r1_2.ipynb)
recorre este pipeline paso a paso y contrasta el indicador de R1.1. Puede leerse
sin haber descargado nada: si el corpus no está construido, lee las cifras del
informe de la última ejecución.

## Ejecutar los experimentos

Antes de comprometer tiempo de GPU conviene pasar la prueba de integración, que
recorre la misma cadena en unos minutos con unos cientos de filas:

```bash
python scripts/prueba_integral.py
```

La cola es estrictamente secuencial y en una sola tarjeta. No es una preferencia:
si dos corridas compartieran GPU, los tiempos dejarían de ser comparables entre
condiciones del mismo experimento.

```bash
bash scripts/experimentos/cola_servidor.sh
python scripts/figuras/generar_figuras_mv.py
python scripts/experimentos/consolidar_iov.py
```

| Experimento | Pregunta que responde | Resultados |
| --- | --- | --- |
| E0 | ¿Cumple el corpus su indicador y el pipeline procesa sin errores? | R1.1, R1.2 |
| E1 | ¿Aporta la multimodalidad frente a los unimodales y al piso clásico? | R1.4, R2.1, R2.2 |
| E2 | ¿Importa el mecanismo con que se fusionan las modalidades? | R2.1, R2.2 |
| E3 | ¿Tolera el modelo que falten o se degraden las entradas? | R1.3 |
| E4 | ¿Generaliza a correo no visto? Cuadro comparativo completo | R1.4, R2.2 |
| E5 | ¿Cuánto de lo medido es fenómeno y cuánto procedencia? | R1.2, R2.2 |
| E6 | ¿Es utilizable en el punto de integración declarado? | R2.3 |

Desde Windows, `scripts/estado_cola.ps1` consulta el avance de la cola en el
servidor.

---

## Cómo leer los resultados

[`doc/COMO_INTERPRETAR_LOS_RESULTADOS.md`](doc/COMO_INTERPRETAR_LOS_RESULTADOS.md)
explica qué significa cada cifra de cada informe y qué pregunta responde.

Dos advertencias que atraviesan todo el trabajo y conviene leer antes que
cualquier número:

1. **El desempeño absoluto no es la eficacia esperable en producción.** El corpus
   se ensambla a partir de colecciones públicas cuya composición hace que la
   clase coincida en buena medida con la procedencia del mensaje. Esa propiedad
   está medida, no supuesta: un clasificador que solo ve las banderas de
   disponibilidad, sin acceso al contenido, alcanza F1 0.6388. Las comparaciones
   entre modelos sí son válidas, porque todos se evalúan sobre las mismas filas.
2. **Las preguntas son exploratorias.** Se dirigen a caracterizar el desempeño
   de las arquitecturas sobre el material disponible y no llevan umbral que
   superar. Se responden con lo que la evidencia permite afirmar, y cuando no
   permite afirmar nada se declara así.
