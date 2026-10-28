# R3.3: Reporte de validación de transparencia operativa

## Medio de verificación comprometido

> Formularios de evaluación UX y resultados de la verificación automatizada de las narrativas

## Indicador objetivamente verificable

> Reporte estructurado por modalidad que traduzca el peso matemático en una justificación técnica comprensible. El desempeño del panel de evaluación se contrasta contra dos referencias: una meta objetivo del 90% de comprensión y un umbral mínimo aceptable del 80%, este último como criterio formal de cumplimiento del resultado. El resultado se clasifica en tres bandas, que son meta alcanzada, mínimo cumplido o no cumplido, y se reporta con esa precisión, sin ajustar el instrumento con posterioridad a su aplicación.

## Qué contiene esta carpeta

| Artefacto | Qué acredita |
| --- | --- |
| `e9_validacion_y_panel.json` | Verificación numérica de las narrativas frente a su vector de atribución (100 de 100 correctas) |
| `ejemplos_del_panel.json` | Correos del panel y su explicación, con los criterios de selección y cuántos candidatos superó cada filtro |
| `formulario_del_panel.html` | Formulario de evaluación tal como lo ve el evaluador: consentimiento, perfil, inducción, valoración de la claridad y la utilidad de cada explicación, seis preguntas de comprensión, SUS y la subescala de calidad de la información del PSSUQ |
| `crear_formulario.gs` | Apps Script que crea ese mismo formulario en Google Forms |
| `clave_de_respuestas.json` | Respuesta correcta de cada pregunta de comprensión, derivada del reporte |
| `resaltado_induccion.png` | Correo del panel resaltado según el peso de cada palabra, como aparece en el formulario |
| `resaltado_phishing_1.png` | Correo del panel resaltado según el peso de cada palabra, como aparece en el formulario |
| `resaltado_legitimo_1.png` | Correo del panel resaltado según el peso de cada palabra, como aparece en el formulario |
| `resaltado_phishing_2.png` | Correo del panel resaltado según el peso de cada palabra, como aparece en el formulario |
| `resaltado_legitimo_2.png` | Correo del panel resaltado según el peso de cada palabra, como aparece en el formulario |

## Código fuente que sustenta el resultado

- `src/phishing_model/xai/fidelity_check.py`
- `src/phishing_model/explicabilidad/procedencia.py`
- `scripts/experimentos/e9_ejemplos_del_panel.py`
- `doc/validacion_xai/generar_formulario.py`
- `doc/validacion_xai/respuestas_a_csv.py`
- `doc/validacion_xai/script_scoring.py`

## Cómo se obtuvo

El panel lee cuatro correos reales del conjunto de prueba, dos de phishing y dos legítimos, con el reporte que el modelo reportado generó para cada uno, y responde diez preguntas de comprensión; después completa el SUS y el PSSUQ. Los correos se eligieron con `scripts/experimentos/e9_ejemplos_del_panel.py`, con criterios fijos declarados antes de aplicar el instrumento: bien clasificados, en inglés o español, cuerpo legible, tres palabras decisivas legibles y visibles en el correo mostrado y, en los legítimos, distintas de un nombre propio. Los enlaces se desactivan y se ocultan direcciones y teléfonos. `ejemplos_del_panel.json` registra cuántos candidatos superó cada filtro. El formulario se genera con `doc/validacion_xai/generar_formulario.py` a partir del instrumento (`doc/validacion_xai/instrumento_evaluacion.md`), y las respuestas exportadas se califican con `doc/validacion_xai/respuestas_a_csv.py` y `doc/validacion_xai/script_scoring.py`.

## Parte pendiente

El panel de evaluadores, criterio primario del indicador, tiene su instrumento listo y está pendiente de resultados. Cuando lleguen, el resultado de `script_scoring.py` se añadirá a esta carpeta.

---

Los artefactos de esta carpeta se copian desde `medios_de_verificacion/experimentos/`, que es la salida de la cola de experimentos, ejecutándola y después `python scripts/experimentos/consolidar_iov.py`.
