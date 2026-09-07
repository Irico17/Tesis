# R1.3: Arquitectura funcional de atención cruzada

## Medio de verificación comprometido

> Informe descriptivo de la arquitectura. Diagrama de arquitectura del modelo y código fuente de la red neuronal.

## Indicador objetivamente verificable

> Implementación en código (PyTorch/TensorFlow) de al menos dos ramas de extracción modal y una capa de fusión jerárquica o atención cruzada documentada. Prueba funcional de entrenamiento.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `prueba_funcional_de_la_arquitectura.json` | Prueba funcional: formas, gradiente, aislamiento y recarga |
| `e3_tolerancia_y_degradacion.json` | Tolerancia a la ausencia de modalidades y degradación adversaria |
| `degradacion_por_ausencia.png` | Desempeño al retirar modalidades en evaluación |
| `degradacion_adversaria.png` | Curvas por operador de evasión e intensidad |
| `INFORME_DE_LA_ARQUITECTURA.md` | Informe descriptivo de la arquitectura |

## Código fuente que sustenta el resultado

- `src/phishing_model/model.py`
- `src/phishing_model/fusion/cross_attention.py`
- `src/phishing_model/fusion/mixture_of_experts.py`
- `src/phishing_model/fusion/modality_encoder.py`
- `src/phishing_model/encoders/`

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
