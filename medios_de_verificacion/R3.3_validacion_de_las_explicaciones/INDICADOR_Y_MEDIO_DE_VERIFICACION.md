# R3.3: Reporte de validación de transparencia operativa

## Medio de verificación comprometido

> Formularios de evaluación UX y resultados de validación vía LLM

## Indicador objetivamente verificable

> Reporte estructurado por modalidad que traduzca el peso matemático en una justificación técnica comprensible. El desempeño del panel de evaluación se contrasta contra dos referencias: una meta objetivo del 90% de comprensión y un umbral mínimo aceptable del 80%, este último como criterio formal de cumplimiento del resultado.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `e9_validacion_y_panel.json` | Verificación numérica de las narrativas frente a su vector de atribución, instrumento del panel y ejemplos seleccionados |

## Código fuente que sustenta el resultado

- `src/phishing_model/xai/fidelity_check.py`
- `src/phishing_model/explicabilidad/procedencia.py`
- `doc/validacion_xai/script_scoring.py`

## Parte pendiente

El panel de analistas, criterio primario del indicador, está preparado y pendiente de aplicación. Su instrumento, la guía de reclutamiento y el cálculo de la puntuación están en `doc/validacion_xai/`.

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
