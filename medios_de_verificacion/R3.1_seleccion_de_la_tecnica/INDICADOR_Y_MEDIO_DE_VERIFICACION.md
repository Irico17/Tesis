# R3.1: Selección del esquema técnico de interpretabilidad

## Medio de verificación comprometido

> Matriz de decisiones técnicas y informe de selección

## Indicador objetivamente verificable

> Cuadro comparativo que justifique la selección de al menos una técnica XAI idónea para la arquitectura.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `e8_comparacion_de_familias_prueba_completa.json` | Comparación de la atención intrínseca, los valores de Shapley y la aproximación local en fidelidad, estabilidad, alcance y coste sobre los 4 411 correos del conjunto de prueba, con resúmenes por clase predicha; es la comparación que sustenta la selección |
| `e8_comparacion_de_familias.json` | Primera comparación, sobre sesenta correos clasificados como phishing; se conserva como antecedente |

## Código fuente que sustenta el resultado

- `src/phishing_model/xai/`
- `src/phishing_model/explicabilidad/`
- `scripts/experimentos/e8_explicabilidad.py`

## Cómo se obtuvo

La metodología fija cuatro criterios de decisión: coste computacional, compatibilidad nativa con la arquitectura, estabilidad y respaldo teórico. La fidelidad (exhaustividad y suficiencia) se mide como comprobación de validez. En los correos clasificados como phishing, que son los que explica el módulo de R3.2, las tres familias no se distinguen en fidelidad; en los legítimos, los gradientes integrados son más fieles. Se selecciona la atención: gana en coste y compatibilidad, empata en estabilidad y pierde en respaldo teórico. La comparación se ejecuta en fragmentos paralelos con `e8_explicabilidad.py --fragmento K/N` y se une con `--unir`.

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
