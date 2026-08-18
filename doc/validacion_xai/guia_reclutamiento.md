# Guía de reclutamiento del panel de evaluadores — validación del módulo XAI

**Proyecto:** Detección de phishing multimodal con IA Explicable — Tesis PUCP
**Fase del plan:** E0 (arranca en paralelo a las Fases A-D, no espera al checkpoint entrenado)
**Referencia normativa interna:** Anexo A de la tesis — "Grupo de Control de Usuarios"

---

## 1. Perfil de evaluador sugerido

Según el Anexo A de la tesis, el panel se compone de **10 evaluadores externos voluntarios**, estudiantes o profesionales de ingeniería. Perfil concreto recomendado:

- **Requisito base (según Anexo A):** estudiantes o profesionales de ingeniería informática, de sistemas, o carreras afines. **No se exige experiencia previa en ciberseguridad** — el objetivo es medir comprensibilidad para un usuario técnico general, no solo para un experto en seguridad.
- **Recomendación, no requisito duro:** si es posible conseguir **3 a 4 personas con algo de trasfondo en seguridad/redes** (cursos de seguridad informática, experiencia laboral en SOC/redes, CTFs, etc.) dentro de los 10, la validación queda más rica — permite comparar informalmente si la comprensión difiere entre perfiles con y sin trasfondo de seguridad. Esto es deseable pero **no debe retrasar el reclutamiento** si no se consigue.
- **Diversidad deseable dentro de lo posible:** mezclar años de carrera / niveles de experiencia (algunos estudiantes de pregrado, algunos profesionales egresados) para no sesgar el panel hacia un solo perfil.
- **Exclusión:** no reclutar a personas que hayan participado en el desarrollo del pipeline o del modelo (compañeros de tesis, asesor, etc.) — invalidaría la independencia de la evaluación.

---

## 2. Plantilla de mensaje de invitación

Usar por correo electrónico o mensaje de texto/WhatsApp. Ajustar el saludo según el canal.

### Versión email

```
Asunto: Invitación a participar en una validación de usabilidad (tesis de ingeniería, ~30 min)

Hola [NOMBRE],

Estoy desarrollando mi tesis de ingeniería en la PUCP sobre detección de
phishing con inteligencia artificial explicable (XAI): un sistema que, además
de clasificar un correo como phishing o legítimo, genera un reporte que
explica en lenguaje simple qué factores llevaron a esa decisión (por ejemplo,
"el dominio recién creado pesó 45% en la decisión").

Te escribo para invitarte a participar como evaluador voluntario de qué tan
clara y comprensible es esa explicación. La participación consiste en:

  1. Una breve inducción de 15 minutos donde te explico cómo funciona el
     sistema y cómo leer el reporte.
  2. Revisar 2-3 ejemplos reales de correos ya clasificados con su reporte
     de explicación.
  3. Responder un cuestionario corto (escalas de opinión + algunas preguntas
     de comprensión).

En total, la sesión toma aproximadamente 30 minutos, [presencial en
[LUGAR] / de forma remota por videollamada — completar según corresponda].

Es una participación completamente voluntaria y no remunerada. Tu aporte
sería muy valioso para la validación de este trabajo. Si te interesa,
respóndeme este mensaje y coordinamos un horario que te acomode.

¡Gracias de antemano!

[TU NOMBRE]
Estudiante de Ingeniería [ESPECIALIDAD] — PUCP
[TU CORREO DE CONTACTO]
```

### Versión mensaje corto (WhatsApp/texto)

```
Hola [NOMBRE]! Estoy haciendo mi tesis sobre detección de phishing con IA
explicable, y necesito 10 voluntarios para evaluar qué tan claras son las
explicaciones que genera el sistema. Es una sesión de ~30 min (15 min de
inducción + un cuestionario corto), voluntaria y sin costo para ti. ¿Te
animarías a participar? Si sí, coordinamos día y hora. Gracias! 🙌
```

**Nota:** el Anexo A no contempla incentivo económico para los evaluadores. No ofrecer pagos, sorteos ni compensación monetaria al invitar — si se desea, un gesto no monetario (agradecimiento público, compartir el resultado de la tesis) es aceptable, pero no está previsto como requisito.

---

## 3. Guion de la inducción de 15 minutos

Aplicar **antes** de entregar el instrumento de evaluación (`instrumento_evaluacion.md`). Objetivo: nivelar el conocimiento del evaluador sin sesgar sus respuestas (no adelantar juicios de valor sobre si el sistema es "bueno" o "claro" — solo explicar cómo funciona y cómo leerlo).

| Minuto | Contenido | Puntos clave a cubrir |
|---|---|---|
| 0–3 | **Qué es el sistema.** | Es un modelo que recibe un correo (texto, estructura HTML, metadatos de red) y lo clasifica como "phishing" o "legítimo". No es un producto comercial, es un prototipo de investigación de tesis. |
| 3–7 | **Qué es una "explicación XAI" en términos simples.** | El sistema no solo da un veredicto — también genera un reporte que dice *qué factores del correo pesaron más en esa decisión* (ej. "un dominio recién creado", "lenguaje de urgencia en el asunto", "un enlace con IP en vez de dominio"). Explicar con una analogía simple: "es como si un profesor no solo pusiera una nota, sino que explicara qué partes del examen influyeron más en esa nota". **No profundizar en la técnica interna (SHAP/LIME/atención)** — el evaluador no necesita saber cómo se calculan los pesos, solo cómo leer el resultado. |
| 7–11 | **Cómo leer el reporte.** | Mostrar la estructura típica del reporte con un ejemplo neutral (no uno de los que se usará en la Sección C, para no adelantar respuestas): qué significa el porcentaje de peso de cada factor, cómo se ordenan (mayor a menor peso), qué significa que un factor "empuje" hacia phishing o hacia legítimo. |
| 11–14 | **Qué se le va a pedir al evaluador.** | Explicar la mecánica de la sesión: primero un cuestionario de percepción (SUS + PSSUQ, escalas de acuerdo/desacuerdo), luego 2-3 ejemplos reales con preguntas de comprensión donde se busca ver si el reporte comunicó correctamente los factores de la decisión. Aclarar que **no hay respuestas "correctas o incorrectas" en las escalas de opinión**, pero que **sí las hay en la sección de comprensión** (busca medir si el reporte comunica bien, no evaluar al participante). |
| 14–15 | **Preguntas y consentimiento.** | Resolver dudas rápidas del evaluador. Confirmar consentimiento de participación voluntaria (ver checklist §4) antes de iniciar el cuestionario. |

**Importante:** no revelar durante la inducción cuál de las 3 técnicas XAI candidatas (SHAP/LIME/atención) generó los reportes que se van a mostrar — mantener la sesión agnóstica a la técnica, tal como el instrumento mismo.

---

## 4. Checklist de logística

- [ ] **Cantidad a reclutar:** mínimo 10 evaluadores (según Anexo A). Reclutar **11-12** para tener 1-2 de margen ante abandonos o sesiones incompletas, sin comprometer el mínimo de 10 completos.
- [ ] **Formato de sesión:** definir presencial vs. remoto según disponibilidad. Remoto (videollamada + formulario compartido en pantalla, o instrumento enviado en PDF/Google Forms) es más flexible para coordinar 10-12 agendas distintas; presencial permite resolver dudas más naturalmente durante la inducción. Puede combinarse (algunos presenciales, otros remotos), documentando el modo en el campo "Modalidad de sesión" del instrumento.
- [ ] **Consentimiento informado:** no se requiere comité de ética formal (estudio de usabilidad de bajo riesgo, sin datos sensibles de los evaluadores, participación anónima con ID tipo `EVAL_01`). Sí es buena práctica dejar constancia simple de consentimiento voluntario antes de empezar. Usar una frase corta de consentimiento verbal registrada o un checkbox/firma simple, por ejemplo:
  > *"Confirmo que participo de forma voluntaria en esta evaluación de usabilidad para fines de tesis, que puedo retirarme en cualquier momento sin consecuencias, y que mis respuestas se usarán de forma anónima y agregada."*
  Registrar el nombre del evaluador solo en una lista de contacto separada del cuestionario (para poder contactarlo si hace falta aclarar algo), nunca junto a sus respuestas — el cuestionario en sí usa solo el ID anónimo.
- [ ] **Materiales listos antes de la primera sesión:** `instrumento_evaluacion.md` con los placeholders de ejemplos ya completados (`[EJEMPLO_N_CORREO_AQUI]`, `[REPORTE_XAI_GENERADO_AQUI_EJEMPLO_N]`, opciones y respuestas correctas) — esto depende de que exista un checkpoint entrenado con explicaciones reales (Fase E1 del plan), así que **la logística de reclutamiento puede avanzar ya, pero las sesiones reales solo pueden aplicarse una vez completado ese checkpoint**.
- [ ] **Registro de resultados:** transcribir cada sesión al CSV que consume `script_scoring.py` inmediatamente después de cada sesión (evitar acumular cuestionarios en papel/PDF sin transcribir).
- [ ] **Cronograma sugerido:**

| Etapa | Cuándo | Depende de |
|---|---|---|
| Enviar invitaciones y confirmar 10-12 evaluadores | Puede empezar ya (en paralelo a Fases A-D del plan técnico) | Nada — es trabajo de coordinación independiente del código |
| Agendar horarios de sesión | Tan pronto se confirmen evaluadores | Disponibilidad del panel |
| Completar placeholders del instrumento con ejemplos reales | Cuando exista checkpoint entrenado con explicaciones (Fase E1) | Fases B/C/E1 del plan técnico (arquitectura entrenada + módulo de narrativa R3.2) |
| Aplicar las 10-12 sesiones | Idealmente concentradas en 1-2 semanas para minimizar deriva de versión del modelo entre evaluadores | Placeholders completos |
| Scoring y análisis agregado | Inmediatamente después de la última sesión | `script_scoring.py` + CSV de respuestas completo |
| **Fecha límite recomendada** | **Antes de fines de septiembre de 2026**, dejando margen dentro de la ventana de la Fase 5 del Gantt de la tesis (módulo XAI, 31 ago–02 oct 2026), para tener resultados listos a tiempo para el capítulo de resultados | — |

  Sugerencia práctica: dado que reclutar y agendar 10-12 personas externas suele tomar 2-3 semanas de calendario real (aunque cada sesión dure solo 30 min), conviene **enviar las invitaciones cuanto antes** — no esperar a que el checkpoint esté listo, ya que el cuello de botella real es la coordinación de agendas, no la disponibilidad del instrumento.

---

## 5. Notas finales

- Evitar aplicar la sesión a un mismo evaluador dos veces con distintas técnicas XAI (si se decide comparar SHAP vs. LIME vs. atención con el mismo panel) sin dejar al menos unos días entre sesiones y sin usar exactamente los mismos ejemplos — para reducir efecto de aprendizaje/memoria en la Sección C de comprensión.
- Si algún evaluador abandona la sesión a la mitad, registrar la sesión como incompleta y no incluirla en el promedio del panel salvo que se documente explícitamente como dato parcial.
- Mantener la lista de contacto de evaluadores (nombre + medio de contacto) en un archivo separado del CSV de respuestas, por buena práctica de anonimización de datos de investigación.
