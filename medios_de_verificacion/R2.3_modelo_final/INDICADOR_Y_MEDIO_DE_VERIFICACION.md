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

## Código fuente que sustenta el resultado

- `src/phishing_model/quantization.py`
- `scripts/experimentos/e6_modelo_optimizado.py`

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
