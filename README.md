# Detección multimodal de phishing en correo electrónico

Código y evidencia de la tesis de licenciatura en Ingeniería Informática de
Cristhofer Alegre (PUCP), asesorada por el Dr. Edwin Villanueva.

El trabajo propone una arquitectura que analiza simultáneamente tres modalidades
del mensaje (el texto, la estructura del cuerpo en HTML y los metadatos de
tránsito y autenticación), las combina mediante fusión jerárquica con atención
cruzada y acompaña cada alerta de una explicación en lenguaje natural.

Sobre un corpus de 44 100 correos, la arquitectura supera a los enfoques
unimodales y a la fusión clásica con diferencias estadísticamente significativas
(F1 de 0.9927 frente a 0.9895 del mejor unimodal, p = 0.0044). La ventaja crece en
el punto de operación exigente: con una falsa alarma por cada mil correos
legítimos detecta el 95.3% del phishing, frente al 92.3% del mejor unimodal y al
76.2% de la fusión clásica. Ocultarle la estructura o la red le cuesta menos de
medio punto de F1, y el modelo cuantizado responde en 74 ms por correo en CPU.

Este repositorio contiene todo lo que se usó para obtener esos resultados y la
evidencia que los sustenta; nada más. El documento de la tesis se redacta aparte.

---

## Objetivos, resultados y dónde se acreditan

Los nombres de los resultados, sus medios de verificación y sus indicadores son
los de la matriz de objetivos de la tesis (Tabla 2).

| Objetivo específico | Resultado | Experimento | Evidencia | Estado |
| --- | --- | --- | --- | --- |
| **OE1.** Diseñar e implementar el modelo multimodal | R1.1 Corpus multimodal consolidado | E0 | [`R1.1_corpus_multimodal/`](medios_de_verificacion/R1.1_corpus_multimodal/) | reportado |
| | R1.2 Pipeline de procesamiento estandarizado | E0, E5 | [`R1.2_pipeline_de_procesamiento/`](medios_de_verificacion/R1.2_pipeline_de_procesamiento/) | reportado |
| | R1.3 Arquitectura funcional de atención cruzada | E3 | [`R1.3_arquitectura/`](medios_de_verificacion/R1.3_arquitectura/) | reportado |
| | R1.4 Pipeline de entrenamiento automatizado | E1, E4 | [`R1.4_entrenamiento/`](medios_de_verificacion/R1.4_entrenamiento/) | reportado |
| **OE2.** Evaluar el modelo frente a enfoques unimodales | R2.1 Modelos de referencia | E1, E2 | [`R2.1_lineas_base/`](medios_de_verificacion/R2.1_lineas_base/) | reportado |
| | R2.2 Cuadro comparativo de desempeño | E1, E2, E4, E5, E7 | [`R2.2_comparacion/`](medios_de_verificacion/R2.2_comparacion/) | reportado |
| | R2.3 Modelo final optimizado | E6 | [`R2.3_modelo_final/`](medios_de_verificacion/R2.3_modelo_final/) | reportado |
| **OE3.** Integrar técnicas de IA explicable | R3.1 Selección del esquema técnico de interpretabilidad | E8 | [`R3.1_seleccion_de_la_tecnica/`](medios_de_verificacion/R3.1_seleccion_de_la_tecnica/) | reportado |
| | R3.2 Módulo XAI integrado operativamente | E9 | [`R3.2_modulo_de_interpretacion/`](medios_de_verificacion/R3.2_modulo_de_interpretacion/) | reportado |
| | R3.3 Reporte de validación de transparencia operativa | E9 | [`R3.3_validacion_de_las_explicaciones/`](medios_de_verificacion/R3.3_validacion_de_las_explicaciones/) | verificación numérica completa; panel de analistas preparado y pendiente |

---

## Estructura del repositorio

```
.
├── src/                          código de la tesis, importable con PYTHONPATH=src
│   ├── phishing_pipeline/        OE1 · R1.1, R1.2   construcción del corpus
│   ├── phishing_model/           OE1 · R1.3, R1.4   arquitectura y entrenamiento
│   │                             OE2 · R2.3         exportación y cuantización
│   │                             OE3 · R3.1–R3.3    explicabilidad
│   └── phishing_baseline/        OE2 · R2.1, R2.2   líneas base clásicas y contraste
├── scripts/
│   ├── experimentos/             OE1–OE3            los diez experimentos, E0 a E9
│   ├── figuras/                  OE1–OE2            figuras de los medios de verificación
│   ├── prueba_integral.py        OE1–OE3            prueba de integración de la cadena
│   └── estado_cola.ps1                              seguimiento de la cola en el servidor
├── notebooks/                    OE1 · R1.1, R1.2   recorrido documentado del pipeline
├── doc/validacion_xai/           OE3 · R3.3         instrumento del panel de analistas
├── medios_de_verificacion/       OE1–OE3            evidencia, una carpeta por resultado
└── data/                                            datos y modelos; no se versionan
```

### `src/phishing_pipeline/`: el corpus (OE1, R1.1 y R1.2)

Descarga ocho colecciones públicas, extrae las tres modalidades de cada mensaje,
elimina duplicados exactos y casi duplicados, y construye el corpus con su
partición agrupada por campaña.

| Módulo | Para qué sirve |
| --- | --- |
| `downloaders/` | Descarga de las colecciones: `correo_real` (phishing_pot, Nazario, SpamAssassin), `listas_correo` (Fedora, kernel), `kaggle_phishing` y `phish_mmf` (CEAS_08, datacon2023); `validation` rechaza descargas incompletas. |
| `cleaners/` | Limpieza al esquema común: `correo_crudo` para `.eml` y `mbox`, `kaggle_phishing` y `phish_mmf` para las colecciones tabulares. |
| `features/` | Extracción de modalidades: `dom_parser` y `dom_stats` (estructura HTML), `network` (URL, IP y autenticación), `vectorizer` (tokenización y escalado, R1.2). |
| `dedup/near_duplicates.py` | Agrupación de casi duplicados por MinHash, base de la partición por campaña. |
| `auditoria_fugas.py` | Auditoría de fugas de etiqueta por disponibilidad de campo (R1.2). |
| `corpus_real.py` | Orquesta todo lo anterior y escribe `Dataset_Real.parquet` (R1.1). |
| `schema.py`, `config.py`, `logging_utils.py` | Esquema canónico, rutas y constantes, registro. |

### `src/phishing_model/`: la arquitectura (OE1, OE2 y OE3)

| Módulo | Para qué sirve | Resultado |
| --- | --- | --- |
| `encoders/` | Rama de texto (DistilBERT) y tokenización de las características de estructura y de red. | R1.3 |
| `fusion/` | Auto-atención entre modalidades no textuales, atención cruzada a nivel de token y mezcla de expertos. | R1.3 |
| `model.py` | Ensambla las ramas y los cinco mecanismos de fusión comparados, con máscaras de disponibilidad. | R1.3 |
| `sanity_check.py` | Prueba funcional de la arquitectura (forward y backward completos). | R1.3 |
| `dataset.py`, `losses.py`, `train.py` | Carga de datos, pérdida ponderada y bucle de entrenamiento con parada temprana. | R1.4 |
| `evaluate.py` | Evaluación y predicciones fila por fila. | R2.2 |
| `quantization.py` | Exportación a ONNX, cuantización a 8 bits y medición de latencia. | R2.3 |
| `xai/` | Las tres familias comparadas (`attention_explainer`, `shap_explainer`, `lime_explainer`), su fidelidad (`faithfulness`), la narrativa (`narrative`) y su verificación numérica (`fidelity_check`). | R3.1–R3.3 |
| `explicabilidad/` | Carga del modelo reportado y comprobación de su procedencia antes de explicarlo; atribución por modalidad. | R3.1, R3.2 |
| `config.py` | Hiperparámetros y tipos de fusión. | — |

### `src/phishing_baseline/`: las referencias (OE2, R2.1 y R2.2)

| Módulo | Para qué sirve |
| --- | --- |
| `b1_tfidf_lr.py` | B1: TF-IDF con regresión logística sobre el texto. |
| `b3_rf_structural.py` | B3: bosque aleatorio sobre la estructura. |
| `b4_network.py` | B4: bosque aleatorio sobre la red. |
| `b5_classical_multimodal.py` | B5: fusión clásica de las tres modalidades. |
| `group_cv.py`, `evaluation.py` | Validación agrupada y métricas comunes. |
| `compare_models.py` | Pruebas de McNemar entre modelos (R2.2). |

### `scripts/experimentos/`: los experimentos (OE1, OE2 y OE3)

Cada experimento responde a una pregunta y deja su informe en
`medios_de_verificacion/experimentos/eN/`, con la huella del corpus y de la
partición, las semillas y el entorno de cada cifra.

| Experimento | Pregunta que responde | Resultados |
| --- | --- | --- |
| `e0_corpus_y_pipeline.py` | ¿Cumple el corpus su indicador y el pipeline procesa sin errores? | R1.1, R1.2 |
| `e1_multimodalidad.py` | ¿Aporta la multimodalidad frente a los unimodales y al piso clásico? | R1.4, R2.1, R2.2 |
| `e2_mecanismo_fusion.py` | ¿Importa el mecanismo con que se fusionan las modalidades? | R2.1, R2.2 |
| `e3_ausencia_ramas.py` | ¿Tolera el modelo que falten o se degraden las entradas? | R1.3 |
| `e4_generalizacion.py` | ¿Generaliza a correo no visto? Cuadro comparativo completo. | R1.4, R2.2 |
| `e5_validez.py` | ¿Cuánto de lo medido es fenómeno y cuánto procedencia? | R1.2, R2.2 |
| `e6_modelo_optimizado.py` | ¿Es utilizable en el punto de integración declarado? | R2.3 |
| `e7_codificador_multilingue.py` | ¿Cambia el resultado con un codificador multilingüe? | R2.2 |
| `e8_explicabilidad.py` | ¿Qué familia de explicabilidad conviene a la arquitectura? | R3.1 |
| `e9_narrativa_y_validacion.py` | ¿Se explica toda alerta y coinciden las cifras de la narrativa con su atribución? | R3.2, R3.3 |

Piezas compartidas: `comun.py` (partición, métricas y emisión de informes),
`ejecutor.py` y `entrenar.py` (entrenamiento de cada variante), `clasicos.py`
(líneas base clásicas B1, B3, B4 y B5), `degradacion.py` (perturbaciones adversarias del texto, en E3),
`cola_servidor.sh` (cola E0 a E7), `manifiesto_de_ejecucion.py` (ventana temporal
de la cola) y `consolidar_iov.py` (arma las carpetas de medios de verificación).

### `scripts/figuras/`: las figuras (OE1 y OE2)

`generar_figuras_mv.py` dibuja desde los informes las figuras que exigen los
medios de verificación: el diagrama de la arquitectura y la degradación
adversaria (R1.3), las curvas de aprendizaje (R1.4), las matrices de confusión, las curvas ROC y las láminas
comparativas (R2.2). Genera además la estructura de descomposición del trabajo y
el diagrama PRISMA de la revisión. `formato.py` fija tamaños y tipografía, y
`estructura_del_trabajo.py` define la EDT del plan de proyecto.

### `medios_de_verificacion/`: la evidencia (OE1, OE2 y OE3)

Una carpeta por resultado (R1.1 a R3.3) con lo que su medio de verificación exige
y una nota que enuncia el indicador comprometido y dice qué artefacto acredita
cada parte. `experimentos/` es la salida cruda de la cola, de la que se
construyen las demás carpetas; `EJECUCION_DE_LA_COLA.json` registra cuándo se
ejecutó cada experimento.

---

## Herramientas

Python 3.10 o superior, con PyTorch y Transformers (DistilBERT) para el modelo,
scikit-learn y SciPy para las líneas base y las pruebas estadísticas, datasketch
para los casi duplicados, lxml y BeautifulSoup para el HTML, Captum y LIME para la
explicabilidad, Jinja2 para la narrativa, y ONNX y ONNX Runtime para el modelo
optimizado. `requirements.txt` declara las cotas mínimas y
`requirements.lock.txt` fija las versiones exactas del entorno de ejecución.

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

El corpus (589 MB), los puntos de control (74 GB) y las predicciones fila por
fila no se versionan por tamaño. La huella criptográfica del corpus queda en la
procedencia de cada informe, de modo que puede comprobarse que una reconstrucción
coincide con la que produjo los resultados publicados.

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

La cola es estrictamente secuencial y en una sola tarjeta: si dos corridas
compartieran GPU, los tiempos dejarían de ser comparables entre condiciones del
mismo experimento. E8 y E9 se ejecutan después, sobre el punto de control que la
cola deja.

```bash
bash scripts/experimentos/cola_servidor.sh                              # E0 a E7
python scripts/experimentos/e8_explicabilidad.py --muestra 60
python scripts/experimentos/e9_narrativa_y_validacion.py --verificar 100
python scripts/figuras/generar_figuras_mv.py
python scripts/experimentos/consolidar_iov.py
```

Desde Windows, `scripts/estado_cola.ps1` consulta el avance de la cola en el
servidor.

---

## Cómo leer los resultados

Cada carpeta de [`medios_de_verificacion/`](medios_de_verificacion/) enuncia el
indicador de su resultado y qué artefacto lo acredita; los informes en JSON de
`medios_de_verificacion/experimentos/` registran, junto a cada cifra, la huella del
corpus y de la partición, las semillas y el entorno con que se calculó.

Dos advertencias que atraviesan todo el trabajo y conviene leer antes que
cualquier número:

1. **El desempeño absoluto no es la eficacia esperable en producción.** El corpus
   se ensambla a partir de colecciones públicas cuya composición hace que la
   clase coincida en buena medida con la procedencia del mensaje. Esa propiedad
   está medida, no supuesta: un clasificador que solo ve las banderas de
   disponibilidad, sin acceso al contenido, alcanza F1 0.6388, 35 puntos por
   debajo del modelo propuesto. Las comparaciones entre modelos sí son válidas,
   porque todos se evalúan sobre las mismas filas.
2. **Las preguntas son exploratorias.** Se dirigen a caracterizar el desempeño
   de las arquitecturas sobre el material disponible y no llevan umbral que
   superar. Se responden con lo que la evidencia permite afirmar.
