# R1.4: Pipeline de entrenamiento automatizado

## Medio de verificación comprometido

> Scripts de entrenamiento automatizados. Logs de ejecución y curvas de aprendizaje (Loss/Accuracy).

## Indicador objetivamente verificable

> Pipeline automatizado demostrando convergencia y estabilidad empíricas y ausencia de sobreajuste crítico (overfitting). Evaluación empírica, durante el entrenamiento, de la necesidad de aplicar class weighting, con base en el comportamiento del recall de la clase minoritaria a lo largo de las particiones de validación agrupadas por campaña. Esta decisión se documentará con evidencia experimental y no se asume a priori.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `e4_generalizacion_y_ponderacion.json` | Curva de aprendizaje, protocolo de partición y decisión de ponderación |
| `curva_por_cantidad_de_datos.png` | Desempeño en función de la cantidad de datos de entrenamiento |
| `e1_multimodalidad.json` | Corridas de las que proceden los historiales de entrenamiento |
| `historiales/` | Historial de cada corrida: pérdida por paso y validación por época |
| `registros/` | Registro de ejecución de la cola completa |
| `curvas_de_aprendizaje/` | Curvas de pérdida y de exactitud por corrida |

## Código fuente que sustenta el resultado

- `scripts/experimentos/cola_servidor.sh`
- `scripts/experimentos/entrenar.py`
- `scripts/experimentos/ejecutor.py`
- `src/phishing_model/train.py`

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
