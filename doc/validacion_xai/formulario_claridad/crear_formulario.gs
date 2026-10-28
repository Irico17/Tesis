/**
 * Crea el formulario del panel de R3.3 en Google Forms.
 * Uso: script.google.com > Nuevo proyecto > pegar este código > Ejecutar
 * la función crearFormulario y autorizar. El enlace para compartir y el de
 * edición aparecen en el registro de ejecución.
 * Generado por doc/validacion_xai/generar_formulario.py; no editar a mano.
 */
function crearFormulario() {
  var form = FormApp.create("Evaluación de explicaciones de un detector de phishing");
  form.setDescription("Este formulario es parte de una tesis de Ingeniería Informática (Cristhofer Alegre, PUCP). Construimos un sistema que detecta correos de phishing (correos que intentan engañarte) y que explica por qué tomó cada decisión. Queremos saber si esas explicaciones se entienden.\n\nVerás un ejemplo resuelto y luego 4 correos, cada uno con su explicación. Por cada correo responderás 2 o 3 preguntas cortas; al final, dos cuestionarios de opinión. Toma unos 15 a 20 minutos.\n\nNo necesitas saber de seguridad ni entender todo el inglés de los correos: las preguntas se responden mirando la explicación. No te evaluamos a ti, sino a la explicación.\n\nEs voluntario y anónimo. Los enlaces de los correos están desactivados; no intentes abrirlos.\n\nConsultas: a20210577@pucp.edu.pe");
  form.setCollectEmail(false);
  try { form.setRequireLogin(false); } catch (e) { Logger.log('Abrir a mano: ' + e); }
  form.setProgressBar(true);
  form.setShowLinkToRespondAgain(false);
  form.setPublishingSummary(false);
  form.setConfirmationMessage('Gracias por participar. Tus respuestas se registraron.');
  var item;
  item = form.addMultipleChoiceItem();
  item.setTitle("Consentimiento").setHelpText("Acepto participar de forma voluntaria y anónima, y sé que puedo dejar el formulario cuando quiera.").setRequired(true);
  item.setChoices([item.createChoice("Acepto participar", FormApp.PageNavigationType.CONTINUE),
                   item.createChoice("No acepto", FormApp.PageNavigationType.SUBMIT)]);
  form.addPageBreakItem().setTitle("Sobre ti").setHelpText("Estas preguntas solo sirven para describir al grupo de participantes.");
  item = form.addMultipleChoiceItem();
  item.setTitle("P1. ¿Cuál es tu situación actual?").setChoiceValues(["Estudiante de pregrado", "Egresado o egresada", "Profesional en ejercicio", "Otra"]).setRequired(true);
  item = form.addMultipleChoiceItem();
  item.setTitle("P2. ¿Cuál es tu carrera o área de formación?").setChoiceValues(["Ingeniería Informática", "Ingeniería de Sistemas", "Ingeniería de Telecomunicaciones o Electrónica", "Ciencias de la Computación", "Otra"]).setRequired(true);
  item = form.addMultipleChoiceItem();
  item.setTitle("P3. ¿Tienes formación o experiencia en ciberseguridad (cursos, trabajo en un centro de operaciones de seguridad, competencias tipo CTF)?").setChoiceValues(["Sí", "No"]).setRequired(true);
  form.addPageBreakItem().setTitle("Cómo leer la explicación").setHelpText("Cada explicación tiene tres partes:\n1. La decisión: si el correo es phishing o legítimo, y qué tan seguro está el sistema (en %).\n2. En qué se basó: qué partes del correo pesaron más, en orden. Casi siempre es el texto; a veces también el diseño del correo o sus enlaces.\n3. La imagen del correo: las palabras más sombreadas son las que más pesaron.\n\nAbajo tienes un ejemplo ya explicado. No se evalúa.");
  form.addImageItem().setTitle("Ejemplo resuelto").setHelpText("Asunto: Protect Your Account").setImage(UrlFetchApp.fetch("https://raw.githubusercontent.com/Irico17/Tesis/master/doc/validacion_xai/formulario_claridad/resaltado_induccion.png"));
  form.addSectionHeaderItem().setTitle("Explicación del sistema: ejemplo resuelto").setHelpText("Decisión del sistema: PHISHING (intento de fraude), con 99.8% de confianza.\n\n¿En qué se basó?\n• Texto del correo: 100.0% del peso. Lo que más pesó:\n    1. «asp»: 26.5%\n    2. «tdcanadatrust»: 22.0%\n    3. «securityandprivacy»: 19.4%\n    4. el signo de puntuación «.»: 16.3%\n    5. «your»: 15.7%\n• Diseño del correo (HTML): no influyó en esta decisión.\n• Enlaces y verificación del remitente: no influyó en esta decisión.");
  form.addSectionHeaderItem().setTitle("Cómo se lee este ejemplo").setHelpText("En este ejemplo, el sistema decidió que el correo es phishing, con 99.8% de confianza. Se basó casi solo en el texto: las palabras que más pesaron fueron «asp», «tdcanadatrust» y «securityandprivacy», que en la imagen aparecen con recuadro. El diseño del correo y sus enlaces no influyeron.");
  form.addPageBreakItem().setTitle("Correo 1 de 4").setHelpText("Ahora, 4 correos reales. Responde según lo que dice la explicación, no según lo que tú opinas del correo.");
  form.addImageItem().setTitle("Correo 1").setHelpText("Asunto: USAA REQUEST YOUR APPROVAL FOR AN INCOMING ACCOUNT DEPOSIT").setImage(UrlFetchApp.fetch("https://raw.githubusercontent.com/Irico17/Tesis/master/doc/validacion_xai/formulario_claridad/resaltado_correo_1.png"));
  form.addSectionHeaderItem().setTitle("Explicación del sistema: correo 1").setHelpText("Decisión del sistema: PHISHING (intento de fraude), con 99.6% de confianza.\n\n¿En qué se basó?\n• Texto del correo: 100.0% del peso. Lo que más pesó:\n    1. «inbox»: 29.9%\n    2. «transffered»: 19.6%\n    3. «authomated»: 19.1%\n    4. la cifra «0514»: 16.4%\n    5. «your»: 15.0%\n• Diseño del correo (HTML): no influyó en esta decisión.\n• Enlaces y verificación del remitente: no influyó en esta decisión.");
  item = form.addMultipleChoiceItem();
  item.setTitle("C1. Según la explicación, ¿qué palabra pesó más para decidir que este correo es phishing?").setChoiceValues(["account", "transffered", "inbox", "payment"]).setRequired(true);
  item = form.addMultipleChoiceItem();
  item.setTitle("C2. Según la explicación, el diseño del correo (HTML) pesó más que su texto.").setChoiceValues(["Verdadero", "Falso"]).setRequired(true);
  item = form.addMultipleChoiceItem();
  item.setTitle("C3. Según la explicación, ¿qué tan seguro está el sistema de su decisión?").setChoiceValues(["Menos del 60%", "Entre 60% y 90%", "Entre 90% y 99%", "Más del 99%"]).setRequired(true);
  form.addPageBreakItem().setTitle("Correo 2 de 4");
  form.addImageItem().setTitle("Correo 2").setHelpText("Asunto: Re: how to enable/start \"git@.service\" via systemd?").setImage(UrlFetchApp.fetch("https://raw.githubusercontent.com/Irico17/Tesis/master/doc/validacion_xai/formulario_claridad/resaltado_correo_2.png"));
  form.addSectionHeaderItem().setTitle("Explicación del sistema: correo 2").setHelpText("Decisión del sistema: LEGÍTIMO (correo normal), con 100.0% de confianza.\n\n¿En qué se basó?\n• Texto del correo: 100.0% del peso. Lo que más pesó:\n    1. «fedora»: 45.0%\n    2. «invocations»: 14.8%\n    3. «wrote»: 14.7%\n    4. «libexec»: 13.0%\n    5. la abreviatura «cdt»: 12.6%\n• Diseño del correo (HTML): no influyó en esta decisión.\n• Enlaces y verificación del remitente: no influyó en esta decisión.");
  item = form.addMultipleChoiceItem();
  item.setTitle("C4. Según la explicación, ¿qué palabra pesó más para decidir que este correo es legítimo?").setChoiceValues(["socket", "fedora", "daemon", "invocations"]).setRequired(true);
  item = form.addMultipleChoiceItem();
  item.setTitle("C5. Según la explicación, la decisión se basó sobre todo en el texto del correo.").setChoiceValues(["Verdadero", "Falso"]).setRequired(true);
  form.addPageBreakItem().setTitle("Correo 3 de 4");
  form.addImageItem().setTitle("Correo 3").setHelpText("Asunto: monkey.org Password update").setImage(UrlFetchApp.fetch("https://raw.githubusercontent.com/Irico17/Tesis/master/doc/validacion_xai/formulario_claridad/resaltado_correo_3.png"));
  form.addSectionHeaderItem().setTitle("Explicación del sistema: correo 3").setHelpText("Decisión del sistema: PHISHING (intento de fraude), con 100.0% de confianza.\n\n¿En qué se basó?\n• Texto del correo: 100.0% del peso. Lo que más pesó:\n    1. «notifications»: 24.5%\n    2. «expires»: 23.5%\n    3. «myaccount»: 19.0%\n    4. el signo de puntuación «,»: 17.3%\n    5. la cifra «2022»: 15.8%\n• Diseño del correo (HTML): no influyó en esta decisión.\n• Enlaces y verificación del remitente: no influyó en esta decisión.");
  item = form.addMultipleChoiceItem();
  item.setTitle("C6. ¿Cuál de estas palabras NO aparece en la lista de la explicación?").setChoiceValues(["myaccount", "password", "expires", "notifications"]).setRequired(true);
  item = form.addMultipleChoiceItem();
  item.setTitle("C7. Según la explicación, la palabra «myaccount» pesó más que «notifications».").setChoiceValues(["Verdadero", "Falso"]).setRequired(true);
  form.addPageBreakItem().setTitle("Correo 4 de 4");
  form.addImageItem().setTitle("Correo 4").setHelpText("Asunto: Protect Your Account").setImage(UrlFetchApp.fetch("https://raw.githubusercontent.com/Irico17/Tesis/master/doc/validacion_xai/formulario_claridad/resaltado_correo_4.png"));
  form.addSectionHeaderItem().setTitle("Explicación del sistema: correo 4").setHelpText("Decisión del sistema: PHISHING (intento de fraude), con 99.8% de confianza.\n\n¿En qué se basó?\n• Texto del correo: 100.0% del peso. Lo que más pesó:\n    1. «asp»: 26.5%\n    2. «tdcanadatrust»: 22.0%\n    3. «securityandprivacy»: 19.4%\n    4. el signo de puntuación «.»: 16.3%\n    5. «your»: 15.7%\n• Diseño del correo (HTML): no influyó en esta decisión.\n• Enlaces y verificación del remitente: no influyó en esta decisión.");
  item = form.addMultipleChoiceItem();
  item.setTitle("C8. Según la explicación, ¿qué decidió el sistema sobre este correo?").setChoiceValues(["Legítimo, con entre 60% y 90% de confianza", "Phishing, con más del 99% de confianza", "Legítimo, con más del 99% de confianza", "Phishing, con entre 60% y 90% de confianza"]).setRequired(true);
  item = form.addMultipleChoiceItem();
  item.setTitle("C9. ¿Cuál de estas palabras está entre las tres que más pesaron, según la explicación?").setChoiceValues(["access", "tdcanadatrust", "agreement", "account"]).setRequired(true);
  item = form.addMultipleChoiceItem();
  item.setTitle("C10. Según la explicación, el diseño del correo (HTML) no influyó en esta decisión.").setChoiceValues(["Verdadero", "Falso"]).setRequired(true);
  form.addPageBreakItem().setTitle("Tu opinión (1 de 2)").setHelpText("¿Qué te parecieron las explicaciones? Marca de 1 (totalmente en desacuerdo) a 5 (totalmente de acuerdo). No hay respuestas correctas.");
  item = form.addGridItem();
  item.setTitle("Grado de acuerdo (1 a 5)").setRows(["A1. Creo que usaría esta explicación con frecuencia si tuviera que revisar correos sospechosos regularmente.", "A2. Encontré que esta explicación es innecesariamente compleja.", "A3. Pensé que esta explicación era fácil de entender.", "A4. Creo que necesitaría el apoyo de una persona técnica para poder interpretar esta explicación.", "A5. Encontré que las partes de esta explicación encajaban bien entre sí.", "A6. Pensé que había demasiada inconsistencia en cómo se presenta esta explicación.", "A7. Me imagino que la mayoría de las personas aprendería a interpretar esta explicación muy rápidamente.", "A8. Encontré esta explicación muy engorrosa o incómoda de leer.", "A9. Sentí mucha confianza al interpretar esta explicación.", "A10. Necesité aprender muchas cosas antes de poder entender esta explicación."]).setColumns(["1 (totalmente en desacuerdo)", "2", "3", "4", "5 (totalmente de acuerdo)"]).setRequired(true);
  form.addPageBreakItem().setTitle("Tu opinión (2 de 2)").setHelpText("Unas preguntas más de opinión, con la misma escala de 1 a 5.");
  item = form.addGridItem();
  item.setTitle("Grado de acuerdo (1 a 5)").setRows(["B1. La información de la explicación fue clara.", "B2. Fue fácil encontrar la información que necesitaba dentro de la explicación para entender la decisión.", "B3. La información de la explicación fue fácil de entender.", "B4. La información fue efectiva para ayudarme a entender por qué el correo fue clasificado así.", "B5. El orden de la información (de lo que más pesó a lo que menos) fue claro.", "B6. Confío en que la información de la explicación refleja razones reales de la clasificación, no una justificación genérica."]).setColumns(["1 (totalmente en desacuerdo)", "2", "3", "4", "5 (totalmente de acuerdo)"]).setRequired(true);
  form.addPageBreakItem().setTitle("Cierre").setHelpText("Gracias por tu tiempo. Si quieres, deja un comentario sobre los reportes.");
  form.addParagraphTextItem().setTitle("Comentarios (opcional)");
  Logger.log('Enlace para responder: ' + form.getPublishedUrl());
  Logger.log('Enlace para editar: ' + form.getEditUrl());
}
