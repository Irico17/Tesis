# Medios de verificación

Una carpeta por resultado comprometido en la matriz de objetivos. Cada una contiene el medio de verificación que ese resultado exige y una nota que enuncia el indicador y dice qué artefacto acredita cada parte.

| Resultado | Carpeta | Artefactos | Estado |
| --- | --- | --- | --- |
| R1.1 | [`R1.1_corpus_multimodal/`](R1.1_corpus_multimodal/) | 4 de 4 | completo |
| R1.2 | [`R1.2_pipeline_de_procesamiento/`](R1.2_pipeline_de_procesamiento/) | 3 de 3 | completo |
| R1.3 | [`R1.3_arquitectura/`](R1.3_arquitectura/) | 5 de 5 | completo |
| R1.4 | [`R1.4_entrenamiento/`](R1.4_entrenamiento/) | 3 de 3 | completo |
| R2.1 | [`R2.1_lineas_base/`](R2.1_lineas_base/) | 2 de 2 | completo |
| R2.2 | [`R2.2_comparacion/`](R2.2_comparacion/) | 9 de 9 | completo |
| R2.3 | [`R2.3_modelo_final/`](R2.3_modelo_final/) | 3 de 3 | completo |
| R3.1 | [`R3.1_seleccion_de_la_tecnica/`](R3.1_seleccion_de_la_tecnica/) | 1 de 1 | completo |
| R3.2 | [`R3.2_modulo_de_interpretacion/`](R3.2_modulo_de_interpretacion/) | 1 de 1 | completo |
| R3.3 | [`R3.3_validacion_de_las_explicaciones/`](R3.3_validacion_de_las_explicaciones/) | 1 de 1 | completo; panel humano pendiente |

## Salida cruda de los experimentos

`experimentos/` conserva lo que escribe la cola tal cual: un informe en formato JSON por experimento con su procedencia (huella del corpus y de la partición, semillas, versiones y unidad de procesamiento), las figuras que cada uno emite, los historiales de entrenamiento y los registros de ejecución. Las carpetas por resultado se construyen desde ahí y pueden reconstruirse en cualquier momento.

## Cómo regenerarlo todo

```bash
bash scripts/experimentos/cola_servidor.sh
python scripts/figuras/generar_figuras_mv.py
python scripts/experimentos/consolidar_iov.py
```
