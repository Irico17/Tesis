# R1.1: Corpus multimodal consolidado

## Medio de verificación comprometido

> Repositorio digital documentado con estructura clara de datos crudos y procesados. Informe descriptivo del tratamiento de los datasets.

## Indicador objetivamente verificable

> Corpus consolidado en el que al menos el 25% de las muestras integra simultáneamente texto, estructura HTML/DOM y metadatos de red, y al menos el 40% integra texto y al menos una modalidad no textual. Prevalencia de la clase positiva entre 0.40 y 0.60, tanto en el corpus como en el subconjunto de dos modalidades. Formato estándar (.parquet, .csv o .json) listo para ingesta. Prueba de humo automatizada que valide la integridad del corpus, incluida la ausencia de fuga por disponibilidad de campo.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `e0_corpus_y_prueba_de_humo.json` | Criterios del indicador contrastados uno a uno y prueba de humo |
| `cobertura_modal_por_coleccion.png` | Disponibilidad de cada modalidad en cada colección |
| `informe_descriptivo_del_corpus.json` | Informe descriptivo del tratamiento de los datasets |
| `viabilidad_de_las_listas_de_correo.json` | Medición que justifica incorporar archivos de listas de discusión |

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
