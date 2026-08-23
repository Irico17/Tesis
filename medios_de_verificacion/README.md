# Medios de verificación y evidencia de los indicadores

Paquete de artefactos que sustentan el Capítulo 4 del documento de avance.
Cristhofer Alegre · 20210577 · PUCP · 2026-08-23

Cada apartado enuncia el **IOV** (indicador objetivamente verificable) y el **MV** (medio
de verificación) tal como figuran en la matriz de objetivos del documento, y a
continuación el artefacto que los sustenta. Todas las cifras fueron re-verificadas
ejecutando código contra estos mismos archivos.

---

## R1.1 · Dataset multimodal consolidado — `R1.1_dataset/`

**IOV.** Dataset consolidado con al menos 30,000 muestras que contengan simultáneamente
texto, estructura HTML/DOM y metadatos de red, en formato estándar listo para ingesta.
Prueba de humo que valide la integridad del corpus.

**MV.** Repositorio digital documentado e informe descriptivo del tratamiento.

| Evidencia | Archivo |
|---|---|
| 43,517 correos con las tres modalidades (45% sobre el umbral de 30,000) | `r1_1_iov_verification.json` |
| Corpus consolidado: 110,152 correos, 61,210 phishing y 48,942 legítimos | `Dataset_Unificado_report.json` |
| **Prueba de humo**: cobertura multimodal 86.80% sobre umbral de 85%, etiquetas nulas 0.00%, cobertura textual 100.00% | `smoke_test.json` |
| Figura de cobertura multimodal | `Dataset_Unificado_multimodal.png` |

## R1.2 · Pipeline de procesamiento — `R1.2_pipeline/`

**IOV.** Ejecución sin errores de la limpieza y extracción de características sobre al
menos el 90% de las muestras.

**MV.** Código fuente y cuaderno documentado.

| Evidencia | Archivo |
|---|---|
| Análisis de utilidad y variabilidad por columna | `column_analysis.json` |
| Agrupación de casi-duplicados por MinHash | `near_duplicate_clusters.json` |
| Código y cuadernos | repositorio del proyecto (Anexo C) |

## R1.3 · Arquitectura de atención cruzada — `R1.3_arquitectura/`

**IOV.** Implementación de al menos dos ramas de extracción modal y una capa de fusión
jerárquica o atención cruzada documentada, más una prueba funcional de entrenamiento.

**MV.** Informe descriptivo, diagrama de arquitectura y código fuente.

| Evidencia | Archivo |
|---|---|
| **Siete comprobaciones automatizadas**, todas superadas, incluida la de pasos de entrenamiento sobre datos reales | `model_sanity_check.json` |
| Diagrama de la arquitectura | `arquitectura_multimodal.png` |

Las tres comprobaciones incorporadas tras la revisión del modelo son el aislamiento de
modalidad ausente, la acotación del escalado y la persistencia de la configuración en el
punto de control.

## R1.4 · Pipeline de entrenamiento — `R1.4_entrenamiento/`

**IOV.** Convergencia matemática y ausencia de sobreajuste crítico. Evaluación empírica
de la necesidad de aplicar ponderación de clases, con base en el comportamiento del
recall de la clase minoritaria a lo largo de los pliegues.

**MV.** Scripts automatizados, registros de ejecución y curvas de aprendizaje.

| Evidencia | Archivo |
|---|---|
| **36 historiales** con pérdida por paso, métricas por época y diagnóstico sobre la fuente excluida | `historiales/` |
| **45 curvas de aprendizaje** (pérdida y exactitud) | `curvas_de_aprendizaje/` |

Los historiales incluyen el campo `diagnostico_dominio_no_observado`, que registra el
desempeño sobre la fuente excluida al término de cada época. Es estrictamente diagnóstico
y no interviene en la selección del punto de control.

> **Pendiente.** La evaluación de la ponderación de clases bajo la arquitectura vigente
> se encuentra en ejecución (seis corridas: ponderación de clases y pérdida focal, con
> tres semillas cada una). El resto del indicador está cubierto.

## R2.1 · Modelos de referencia — `R2.1_referencia/`

**IOV.** Justificación e implementación de al menos dos enfoques unimodales funcionales.

**MV.** Informe de selección y código de implementación.

| Evidencia | Archivo |
|---|---|
| Cinco modelos de referencia, en partición convencional y por fuente no observada | `baselines_results.json` |
| B2 sobre el corpus íntegro, bajo el mismo protocolo que la propuesta | `b2_loso_results.json` |
| Ablación de características estructurales | `b3_feature_ablation.json` |
| Curvas ROC y matrices de confusión de los modelos de referencia | `roc_curves_test.png`, `confusion_matrix_B*.png` |

## R2.2 · Comparación de modelos — `R2.2_comparacion/`

**IOV.** Cuadro comparativo que cuantifique las diferencias en exactitud, precisión,
recall, F1 y AUC-ROC entre el modelo propuesto y los modelos de referencia unimodales,
acompañado de pruebas de significancia estadística que determinen si son significativas
(p < 0.05) o producto de varianza aleatoria.

**MV.** Informe técnico, gráficos de rendimiento (curvas ROC y matriz de confusión) y
reporte de métricas comparativas.

| Evidencia | Archivo |
|---|---|
| **15 corridas** por fuente no observada: 4 configuraciones × 3 semillas, más 3 de RoBERTa | `*_loso.json` |
| **Área bajo la curva ROC** por configuración, con desviación entre semillas | `roc_auc_por_configuracion.json` |
| **Prueba exacta de McNemar** frente a B1 y B2, por pliegue | `significancia_mcnemar.json` |
| Matrices de confusión del modelo propuesto, por pliegue | `matrices_confusion.json`, `matriz_confusion_*.png` |
| Predicciones fila a fila para reproducir cualquier prueba pareada | `predicciones/` |

### Resultado de la prueba de significancia

Las seis comparaciones alcanzan significancia al nivel de 0.05, de modo que las
diferencias **no** son atribuibles a varianza aleatoria. Su dirección es mayoritariamente
desfavorable al modelo propuesto:

| Comparación | Fuente excluida | Solo el propuesto | Solo la referencia | p | Favorece a |
|---|---|---:|---:|---:|---|
| Frente a B1 | Kaggle_Phishing_Email | 126 | 543 | <0.0001 | B1 |
| | PhishMMF | 451 | 923 | <0.0001 | B1 |
| | Spam_Genuine_Mail | 9,064 | 6,533 | <0.0001 | El propuesto |
| Frente a B2 | Kaggle_Phishing_Email | 32 | 15 | 0.0186 | El propuesto |
| | PhishMMF | 182 | 364 | <0.0001 | B2 |
| | Spam_Genuine_Mail | 2,312 | 2,705 | <0.0001 | B2 |

El único pliegue en que supera a B1 es aquel con mayor volumen de entrenamiento, y el
único en que supera a B2 es aquel en que ambos rinden de forma prácticamente nula, de
modo que esa ventaja carece de relevancia práctica.

---

## Reproducción

```bash
python -m phishing_model.sanity_check          # R1.3, siete comprobaciones
python scripts/agregar_semillas.py             # R2.2, agregado entre semillas
python scripts/veredicto_final.py              # R2.2, comparación pareada
python scripts/generar_iov_mv.py               # significancia, ROC-AUC y matrices
```

---

## Relación con `data/reports/`

El capítulo de resultados cita rutas bajo `data/reports/`, que es el directorio donde los
guiones escriben durante la ejecución. Los artefactos que sustentan cada afirmación se
conservan aquí en su versión curada, de modo que la correspondencia es la siguiente:

| Ruta citada en el documento | Ubicación versionada |
|---|---|
| `data/reports/r1_1_iov_verification.json` | `R1.1_dataset/` |
| `data/reports/smoke_test.json` | `R1.1_dataset/` |
| `data/reports/model_sanity_check.json` | `R1.3_arquitectura/` |
| `data/reports/training_logs/` | `R1.4_entrenamiento/historiales/` |
| `data/reports/baselines_results.json` | `R2.1_modelos_referencia/` |
| `data/reports/loso/` | `R2.2_comparacion/` |
| `data/reports/iov_mv/` | `R2.2_comparacion/` |

Las salidas de trabajo bajo `data/reports/loso/`, `data/reports/training_logs/` y
`data/reports/iov_mv/` no se versionan por duplicar unos 31 MB sin aportar nada: su
contenido vigente es exactamente el de esta carpeta.
