# R3.1: Selección del esquema técnico de interpretabilidad

## Medio de verificación comprometido

> Matriz de decisiones técnicas y informe de selección

## Indicador objetivamente verificable

> Cuadro comparativo que justifique la selección de al menos una técnica XAI idónea para la arquitectura.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `e8_comparacion_de_familias.json` | Comparación de la atención intrínseca, los valores de Shapley y la aproximación local en fidelidad, estabilidad, alcance y coste, sobre el punto de control reportado, con la técnica seleccionada |

## Código fuente que sustenta el resultado

- `src/phishing_model/xai/`
- `src/phishing_model/explicabilidad/`
- `scripts/experimentos/e8_explicabilidad.py`

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
