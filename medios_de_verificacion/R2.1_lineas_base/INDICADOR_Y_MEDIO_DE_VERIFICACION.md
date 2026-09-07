# R2.1: Selección e implementación de líneas base unimodales

## Medio de verificación comprometido

> Informe de selección y código de implementación.

## Indicador objetivamente verificable

> Justificación e implementación de al menos dos enfoques unimodales funcionales como líneas base.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `e1_unimodales_y_piso_clasico.json` | Desempeño de los tres unimodales neuronales y las cuatro líneas base |
| `e2_mecanismos_de_fusion.json` | Las mismas referencias bajo cada mecanismo de fusión |

## Código fuente que sustenta el resultado

- `scripts/experimentos/clasicos.py`
- `src/phishing_baseline/`

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
