# R1.2: Pipeline de procesamiento estandarizado

## Medio de verificación comprometido

> Código fuente y notebook de Jupyter/Colab (.ipynb) documentado.

## Indicador objetivamente verificable

> Ejecución sin errores del script de limpieza y extracción de características del 90% de las muestras del dataset.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `e0_cobertura_del_procesamiento.json` | Cobertura del procesamiento y comprobaciones de integridad |
| `e5_validez_y_fugas.json` | Auditoría de qué mide cada característica y suelo de procedencia |
| `pipeline_r1_1_r1_2.ipynb` | Notebook documentado que recorre el pipeline de extremo a extremo |

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
