# R2.3: Modelo final optimizado y validado

## Medio de verificación comprometido

> Informe final junto a archivo de pesos algorítmicos exportados y reporte de latencia.

## Indicador objetivamente verificable

> El modelo optimizado es funcional para realizar inferencias sobre datos nuevos no vistos, y reporta su tasa de falsos negativos observada en el conjunto de prueba, sin presuponer una reducción respecto de la línea base, conforme al enfoque de caracterización adoptado en R2.2.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `e6_latencia_y_falsos_negativos.json` | Tamaño, latencia media y percentil 95, acuerdo de cuantización y tasa de falsos negativos de cada arquitectura |
| `registro_de_la_medicion.log` | Registro de la exportación, la cuantización y la medición |
| `informe_de_exportacion_de_la_propuesta.json` | Exportación y cuantización de la arquitectura de la que habla el capítulo, con la ruta de los pesos resultantes |

## Artefactos que no se versionan por tamaño

- `data/model/onnx/atencion_cruzada_token/model_int8.onnx`: Pesos exportados y cuantizados de la arquitectura propuesta, unos 66 MB. No se versionan por tamaño; se regeneran con E6 y su huella queda en el informe de exportación.
- `data/model/onnx/atencion_cruzada_token/model_fp32.onnx`: Los mismos pesos en coma flotante de 32 bits, unos 260 MB.

## Código fuente que sustenta el resultado

- `src/phishing_model/quantization.py`
- `scripts/experimentos/e6_modelo_optimizado.py`

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
