# R3.2: Módulo XAI integrado operativamente

## Medio de verificación comprometido

> Código fuente del módulo acoplado.

## Indicador objetivamente verificable

> Extracción exitosa de pesos de importancia en la muestra de prueba de predicciones positivas.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `e9_narrativas_de_la_muestra_critica.json` | Cobertura de la extracción y de las narrativas sobre toda la muestra clasificada como fraudulenta, con su legibilidad y una muestra de explicaciones |

## Código fuente que sustenta el resultado

- `src/phishing_model/xai/narrative.py`
- `scripts/experimentos/e9_narrativa_y_validacion.py`

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
