# Instrumento de evaluación de usabilidad y comprensibilidad del módulo XAI

**Proyecto:** Detección de phishing multimodal con IA Explicable — Tesis PUCP
**Fase del plan:** E0 (preparación del panel), resultado de tesis **R3.3**
**Duración estimada:** unos 25 minutos por evaluador, en un formulario autoadministrado de Google Forms
**Aplicable a:** cualquier técnica XAI candidata (SHAP, LIME o atención intrínseca). Este instrumento **no depende de cuál gane la matriz de decisión de R3.1** — las preguntas se refieren al *reporte de explicación* que ve el evaluador, no a la técnica interna que lo generó. No se debe revelar al evaluador qué técnica generó el reporte (evita sesgo de expectativa); si se evalúan varias técnicas en paralelo, aplicar este mismo instrumento una vez por técnica, con evaluadores o ejemplos distintos para evitar efecto de aprendizaje.

---

## 0. Datos de la sesión

| Campo | Valor |
|---|---|
| ID de evaluador (anónimo, ej. `EVAL_01`) | ______________________ |
| Fecha | ______________________ |
| Modalidad de sesión (presencial/remota) | ______________________ |
| Técnica XAI evaluada (uso interno del tesista, no se muestra al evaluador) | ______________________ |
| Duración real de la sesión | ______________________ |

El evaluador **no necesita saber** qué técnica XAI generó el reporte. Ese dato lo completa el tesista después de la sesión, para fines de análisis agregado por técnica si se evalúa más de una.

---

## 1. Orden de aplicación y tiempos sugeridos

El instrumento se aplica como formulario en línea, autoadministrado, que generan `generar_formulario.py` y el Apps Script `formulario/crear_formulario.gs`.

| Paso | Contenido | Tiempo estimado |
|---|---|---|
| 1 | Presentación, consentimiento y tres preguntas de perfil | 2 min |
| 2 | Inducción escrita: cómo leer el reporte, con un ejemplo guiado que no se usa en las preguntas | 3–4 min |
| 3 | Sección C — Comprensión (10 preguntas sobre 4 correos reales) | 10–12 min |
| 4 | Sección A — SUS adaptado (10 ítems) | 3–4 min |
| 5 | Sección B — PSSUQ adaptado (16 ítems) | 4–5 min |
| 6 | Cierre y comentario opcional | 1 min |

**Total:** unos 25 minutos.

**Por qué la comprensión va antes que las escalas de opinión.** El SUS y el PSSUQ se diseñaron para responderse después de usar el sistema; aquí, usarlo es leer los reportes. Si se respondieran tras el ejemplo guiado, medirían la opinión sobre un único reporte. La numeración de las secciones (A, B, C) se conserva para que coincida con el scoring.

**Nota metodológica sobre la escala:** para mantener la sesión corta y evitar confundir al evaluador con dos escalas distintas, tanto el SUS como el PSSUQ adaptados de este instrumento usan una **escala Likert unificada de 1 a 5** (1 = totalmente en desacuerdo, 5 = totalmente de acuerdo), con el mismo sentido en ambos cuestionarios (mayor puntaje = mejor percepción de usabilidad). Esto es una adaptación deliberada de los originales (SUS ya usa 1–5; el PSSUQ clásico usa 1–7 con sentido invertido). Se documenta aquí para que el capítulo de metodología de la tesis lo declare explícitamente como decisión de diseño, no como error de transcripción. Las comparaciones con la literatura que reporta PSSUQ en escala 1–7 deben hacerse con cautela (comparar patrones relativos, no valores absolutos).

---

## 2. Sección A — SUS adaptado (System Usability Scale, Brooke 1996)

Instrucciones para el evaluador: *"A continuación hay 10 afirmaciones sobre el reporte de explicación que acabas de revisar. Para cada una, marca qué tan de acuerdo estás, en una escala de 1 (totalmente en desacuerdo) a 5 (totalmente de acuerdo). No hay respuestas correctas o incorrectas — nos interesa tu percepción honesta."*

| # | Afirmación | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 1 | Creo que usaría este reporte de explicación con frecuencia si tuviera que revisar correos sospechosos regularmente. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | Encontré que este reporte de explicación es innecesariamente complejo. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | Pensé que este reporte de explicación era fácil de entender. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | Creo que necesitaría el apoyo de una persona técnica para poder interpretar este reporte de explicación. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | Encontré que los distintos elementos de este reporte de explicación (factores, pesos, justificación) estaban bien integrados entre sí. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 6 | Pensé que había demasiada inconsistencia en cómo se presenta este reporte de explicación. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 7 | Me imagino que la mayoría de las personas aprendería a interpretar este reporte de explicación muy rápidamente. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 8 | Encontré este reporte de explicación muy engorroso/incómodo de leer. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 9 | Sentí mucha confianza al interpretar este reporte de explicación. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 10 | Necesité aprender muchas cosas antes de poder entender este reporte de explicación. | ☐ | ☐ | ☐ | ☐ | ☐ |

### Scoring SUS (estándar, adaptado a escala 1–5 que ya usa el original de Brooke)

Para cada ítem impar (1, 3, 5, 7, 9): `puntaje_contribución = valor_respuesta − 1`
Para cada ítem par (2, 4, 6, 8, 10): `puntaje_contribución = 5 − valor_respuesta`

`SUS_score = (suma de las 10 contribuciones) × 2.5`

Resultado en el rango 0–100. **Punto de referencia habitual en la literatura (Bangor, Kortum & Miller, 2008; Sauro, 2011): SUS > 68 se considera "por encima del promedio".** Un SUS ≥ 80 suele considerarse "excelente"; por debajo de 51, "pobre". Este umbral es una referencia complementaria del instrumento SUS en sí — el criterio de aceptación formal de R3.3 de la tesis es el 80% de comprensión narrativa de la Sección C, no el SUS ni el PSSUQ (ver nota al final de la Sección C).

---

## 3. Sección B — PSSUQ adaptado (Post-Study System Usability Questionnaire, basado en Lewis 1992/2002)

Instrucciones para el evaluador: *"Las siguientes 16 afirmaciones son sobre distintos aspectos del reporte de explicación: qué tan útil es, qué tan clara es la información que contiene, y qué tan agradable es de leer. Usa la misma escala de 1 a 5."*

**Nota de adaptación:** las 16 preguntas siguientes son una traducción y adaptación al español, en el contexto de "explicaciones de clasificación de phishing", de la estructura estándar del PSSUQ (versión de 16 ítems, Lewis 2002), agrupada en sus 3 subescalas clásicas: Utilidad del Sistema (SysUse), Calidad de la Información (InfoQual) y Calidad de la Interfaz (IntQual).

#### B.1 — Subescala "Utilidad del Sistema" (SysUse) — ítems 1 a 6

| # | Afirmación | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 1 | En general, la facilidad de uso de este reporte de explicación me resulta satisfactoria. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | Fue sencillo entender por qué el sistema clasificó el correo como lo hizo, usando este reporte. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | Pude identificar rápidamente los factores clave que llevaron a la clasificación. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | Pude interpretar el reporte de forma eficiente, sin pasos innecesarios. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 5 | Usar este reporte de explicación me resultó cómodo. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 6 | Fue fácil aprender a interpretar este tipo de reporte. | ☐ | ☐ | ☐ | ☐ | ☐ |

#### B.2 — Subescala "Calidad de la Información" (InfoQual) — ítems 7 a 12

| # | Afirmación | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 7 | La información presentada en el reporte (factores, pesos, justificación textual) fue clara. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 8 | Fue fácil encontrar la información que necesitaba dentro del reporte para entender la decisión. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 9 | La información del reporte fue fácil de entender. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 10 | La información fue efectiva para ayudarme a entender por qué el correo fue clasificado así. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 11 | La organización de la información dentro del reporte (orden de los factores, jerarquía de pesos) fue clara. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 12 | Confío en que la información del reporte refleja razones reales de la clasificación, no una justificación genérica. | ☐ | ☐ | ☐ | ☐ | ☐ |

#### B.3 — Subescala "Calidad de la Interfaz" (IntQual) — ítems 13 a 15

| # | Afirmación | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 13 | La forma en que se presenta visualmente el reporte (formato, íconos, resaltado de factores) fue agradable. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 14 | Me gustó la manera en que el reporte comunica los factores y sus pesos. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 15 | El reporte tiene toda la información que esperaría encontrar en una explicación de este tipo. | ☐ | ☐ | ☐ | ☐ | ☐ |

#### B.4 — Ítem de satisfacción general — ítem 16

| # | Afirmación | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 16 | En general, este reporte de explicación me resulta satisfactorio. | ☐ | ☐ | ☐ | ☐ | ☐ |

### Scoring PSSUQ

Cada subescala se calcula como el **promedio simple de los ítems que la componen** (escala 1–5, mayor = mejor):

- `SysUse = promedio(ítems 1–6)`
- `InfoQual = promedio(ítems 7–12)`
- `IntQual = promedio(ítems 13–15)`
- `Overall = promedio(ítems 1–16)` (incluye el ítem 16 de satisfacción general)

No existe un "umbral aprobatorio" estándar único para el PSSUQ en la literatura (a diferencia del SUS); se reporta como evidencia descriptiva complementaria y para comparar entre variantes de fusión / técnicas XAI si se evalúa más de una. El criterio de aceptación duro de R3.3 es exclusivamente el de la Sección C.

---

## 4. Sección C — Comprensión narrativa (criterio primario de R3.3; meta 90%, mínimo 80%)

Esta sección es la que responde directamente al resultado **R3.3** de la tesis: *"Diseño y aplicación de cuestionarios estructurados (SUS/PSSUQ) a analistas de seguridad como criterio primario de comprensibilidad... requiriendo un umbral mínimo del 80% en los índices de claridad y confianza cognitiva operativa."* El SUS y el PSSUQ (Secciones A y B) miden *percepción* de usabilidad; esta sección mide **comprensión objetiva verificable** — si el evaluador de hecho entendió qué factores llevaron a la clasificación, no solo si el reporte "se sintió" claro.

### Ejemplos y preguntas

La sección usa **cuatro correos reales** del conjunto de prueba, todos correctamente clasificados por el modelo reportado: dos de phishing y dos legítimos, presentados en orden alterno. Un quinto correo de phishing se usa solo en la inducción. Los selecciona `scripts/experimentos/e9_ejemplos_del_panel.py` con criterios fijos y registrados en su informe (`medios_de_verificacion/experimentos/e9/ejemplos_del_panel.json`): idioma inglés o español, cuerpo legible, palabras decisivas legibles y visibles en el correo mostrado, ninguna repetida y, en los legítimos, ninguna que sea un nombre propio. Los enlaces se desactivan y se ocultan direcciones y teléfonos.

No hay ejemplo de confianza intermedia: el modelo casi nunca duda, y ningún correo bien clasificado con esa confianza tiene una explicación legible. La pregunta prevista para él, qué factor empuja hacia cada clase, tampoco podría responderse, porque la atención no tiene signo.

Las diez preguntas, y su respuesta correcta, se derivan del propio reporte que el evaluador lee (`generar_formulario.py`); la clave queda en `formulario/clave_de_respuestas.json`. Cubren lo que un analista necesita leer en el reporte:

| Correo | Preguntas | Qué miden |
|---|---|---|
| 1 (phishing) | C1, C2, C3 | Palabra de mayor peso; peso del texto frente a la estructura; nivel de confianza |
| 2 (legítimo) | C4, C5 | Palabra de mayor peso; modalidad que sustenta la decisión |
| 3 (phishing) | C6, C7 | Qué palabra no cita el reporte; orden de dos palabras citadas |
| 4 (legítimo) | C8, C9, C10 | Decisión y confianza; palabra entre las tres más influyentes; aporte de la estructura |

Las opciones incorrectas son palabras que aparecen en el correo pero que el reporte no cita, de modo que no se acierta adivinando qué palabra «suena» a phishing; la correcta nunca va en primer lugar, y las cuatro preguntas de verdadero o falso se reparten dos y dos.

### Cómo se calcula el % de comprensión y cómo se agrega al umbral de R3.3

Por evaluador:

```
% comprensión (evaluador) = (número de respuestas correctas de C1–C10) / 10 × 100
```

A nivel de panel (todos los evaluadores):

```
% comprensión (panel) = promedio del % de comprensión de todos los evaluadores
```

**Criterio de aceptación de R3.3 — dos referencias, tres bandas.** A raíz de la observación del Jurado 1 en la revisión del E3 (*"se puede iniciar con 90 e ir bajando"*, pág. 12 del PDF), el resultado del panel se reporta contra **dos** referencias en lugar de una sola:

| Banda | Rango | Interpretación |
|---|---|---|
| **Meta alcanzada** | ≥ 90% | Supera la meta objetivo sugerida por el jurado y, por tanto, también el mínimo formal. |
| **Mínimo cumplido** | 80% – 90% | Cumple el IOV comprometido en la Tabla 2 sin llegar a la meta más exigente. **Resultado válido para acreditar R3.3**, reportado con esta precisión. |
| **No cumple** | < 80% | No satisface el IOV. Se reporta como hallazgo honesto. |

El **mínimo de 80%** es el compromiso formal del IOV de R3.3 en la Tabla 2 y no se renegocia; el **90%** es una meta objetivo más exigente que permite partir de un estándar alto sin alterar el compromiso ya evaluado.

En ningún caso se "ajusta" el instrumento post-hoc para forzar el umbral: si el panel queda por debajo del 80%, se documenta como limitación y como motivo para iterar sobre el diseño del módulo de narrativa (R3.2) antes de una segunda ronda de validación. El script `script_scoring.py` clasifica automáticamente en estas tres bandas y emite la interpretación correspondiente.

Complementariamente (no sustituye este chequeo), R3.3 incluye una verificación automatizada de las narrativas, resuelta con una regla determinista, que comprueba que cada cifra del texto generado corresponde a la del vector de atribución que lo originó (E9) — ese chequeo es independiente de este instrumento y se aplica sobre el propio texto generado, no sobre el panel humano. Este instrumento cubre exclusivamente la comprensión humana.

---

## 5. Cierre

*"Gracias por tu tiempo y por tus respuestas. Esta información se usará de forma agregada y anónima para evaluar la comprensibilidad del módulo de explicabilidad de un proyecto de tesis de ingeniería. Si tienes algún comentario adicional sobre el reporte que revisaste, puedes escribirlo aquí:"*

```
[COMENTARIOS_ABIERTOS_DEL_EVALUADOR]
```

---

## 6. Notas para el tesista (no forman parte del cuestionario aplicado)

1. **Agnosticismo respecto a la técnica XAI:** las preguntas se refieren al reporte que ve el evaluador, no a la técnica que lo generó, y el formulario no la nombra en las preguntas.
2. **Regeneración:** el formulario no se edita a mano. Si cambian los ejemplos o los ítems, se ejecuta `python doc/validacion_xai/generar_formulario.py`, que reescribe el Apps Script, la vista previa y la clave de respuestas a partir de este instrumento y del informe de E9.
3. **Consistencia entre evaluadores:** todos los evaluadores ven los mismos cuatro correos, en el mismo orden.
4. **Congelación antes de aplicar:** el instrumento y el formulario se fijan antes de enviar el enlace y no se modifican durante la aplicación, conforme al IOV de R3.3.
5. **Captura y scoring:** la exportación de Google Forms se convierte con `respuestas_a_csv.py` (descarta a quien no acepta, califica la sección C con la clave y asigna identificadores anónimos) y se procesa con `script_scoring.py` (columnas `sus_q1..sus_q10`, `pssuq_q1..pssuq_q16`, `comprension_q1..q10`, `comprension_correcta_q1..q10`).
