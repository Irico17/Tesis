# R2.2: Pipeline de comparación de modelos

## Medio de verificación comprometido

> Informe técnico y gráficos de rendimiento (curvas ROC, matriz de confusión). Reporte de métricas comparativas.

## Indicador objetivamente verificable

> Cuadro comparativo que cuantifique y caracterice las diferencias en exactitud, precisión, exhaustividad, F1 y ROC-AUC entre el modelo propuesto y las líneas base unimodales, acompañado de pruebas de significancia estadística (McNemar o t de Student) que determinen si las diferencias observadas son estadísticamente significativas (p < 0.05) o producto de varianza aleatoria.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `cuadro_comparativo_corpus_completo.json` | Cuadro comparativo de las doce arquitecturas con sus pruebas |
| `cuadro_comparativo.png` | Desempeño de todas las arquitecturas sobre el corpus completo |
| `e1_aporte_de_la_multimodalidad.json` | Contraste de McNemar frente al mejor unimodal neuronal |
| `aporte_de_la_multimodalidad.png` | Comparación sobre el subconjunto con las tres modalidades |
| `e2_equivalencia_entre_mecanismos.json` | Pruebas de equivalencia entre mecanismos de fusión |
| `mecanismos_de_fusion.png` | Desempeño por mecanismo de fusión |
| `e5_validez_de_la_comparacion.json` | Qué parte de lo medido corresponde al fenómeno y qué a la procedencia |
| `e7_codificador_multilingue.json` | Contraste entre codificador monolingüe y multilingüe, por idioma |
| `codificador_multilingue.png` | Desempeño de cada codificador sobre los subconjuntos por idioma |
| `matrices_de_confusion/` | Matriz de confusión de cada uno de los doce modelos |
| `curvas_roc_*.png` | Curvas ROC por familia y en la región de operación |

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
