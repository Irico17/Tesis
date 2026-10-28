/**
 * Crea el formulario del panel de R3.3 en Google Forms.
 * Uso: script.google.com > Nuevo proyecto > pegar este código > Ejecutar
 * la función crearFormulario y autorizar. El enlace para compartir y el de
 * edición aparecen en el registro de ejecución.
 * Generado por doc/validacion_xai/generar_formulario.py; no editar a mano.
 */
function crearFormulario() {
  var form = FormApp.create("Evaluación de explicaciones de un detector de phishing");
  form.setDescription("Este formulario es parte de una tesis de Ingeniería Informática (Cristhofer Alegre, PUCP). Construimos un sistema que detecta correos de phishing (correos que intentan engañarte) y que explica por qué tomó cada decisión. Queremos saber si esas explicaciones se entienden.\n\nVerás un ejemplo resuelto y luego 4 correos, cada uno con su explicación. Por cada correo responderás 1 o 2 preguntas cortas y dirás si la explicación se entiende y si te sería útil; al final, dos cuestionarios de opinión. Toma unos 15 minutos.\n\nNo necesitas saber de seguridad ni entender todo el inglés de los correos: las preguntas se responden mirando la explicación. No te evaluamos a ti, sino a la explicación.\n\nEs voluntario y anónimo. Los enlaces de los correos están desactivados; no intentes abrirlos.\n\nConsultas: a20210577@pucp.edu.pe");
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
  form.addImageItem().setTitle("Ejemplo resuelto").setHelpText("Asunto: monkey.org Password update\nEl correo original tiene formato HTML (diseño), con 2 enlaces; aquí se muestra solo su texto.").setImage(UrlFetchApp.fetch("https://raw.githubusercontent.com/Irico17/Tesis/master/doc/validacion_xai/formulario/resaltado_induccion.png"));
  form.addSectionHeaderItem().setTitle("Explicación del sistema: ejemplo resuelto").setHelpText("Decisión del sistema: PHISHING (intento de fraude), con 100.0% de confianza.\n\n¿En qué se basó?\n• Texto del correo: 100.0% del peso. Lo que más pesó:\n    1. «notifications»: 24.5%\n    2. «expires»: 23.5%\n    3. «myaccount»: 19.0%\n    4. el signo de puntuación «,»: 17.3%\n    5. la cifra «2022»: 15.8%\n• Diseño del correo (HTML): no influyó en esta decisión.\n• Enlaces y verificación del remitente: no influyó en esta decisión.");
  form.addSectionHeaderItem().setTitle("Cómo se lee este ejemplo").setHelpText("En este ejemplo, el sistema decidió que el correo es phishing, con 100.0% de confianza. Se basó casi solo en el texto: las palabras que más pesaron fueron «notifications», «expires» y «myaccount», que en la imagen aparecen con recuadro. El diseño del correo y sus enlaces no influyeron.");
  form.addPageBreakItem().setTitle("Correo 1 de 4").setHelpText("Ahora, 4 correos reales. Responde las preguntas según lo que dice la explicación, no según lo que tú opinas del correo. Al final de cada correo, marca de 1 (totalmente en desacuerdo) a 5 (totalmente de acuerdo) qué te pareció su explicación.");
  form.addImageItem().setTitle("Correo 1").setHelpText("Asunto: USAA REQUEST YOUR APPROVAL FOR AN INCOMING ACCOUNT DEPOSIT\nEl correo original tiene formato HTML (diseño), con 12 imágenes y 11 enlaces; aquí se muestra solo su texto.").setImage(UrlFetchApp.fetch("https://raw.githubusercontent.com/Irico17/Tesis/master/doc/validacion_xai/formulario/resaltado_phishing_1.png"));
  form.addSectionHeaderItem().setTitle("Explicación del sistema: correo 1").setHelpText("Decisión del sistema: PHISHING (intento de fraude), con 99.6% de confianza.\n\n¿En qué se basó?\n• Texto del correo: 100.0% del peso. Lo que más pesó:\n    1. «inbox»: 29.9%\n    2. «transffered»: 19.6%\n    3. «authomated»: 19.1%\n    4. la cifra «0514»: 16.4%\n    5. «your»: 15.0%\n• Diseño del correo (HTML): no influyó en esta decisión.\n• Enlaces y verificación del remitente: no influyó en esta decisión.");
  item = form.addMultipleChoiceItem();
  item.setTitle("C1. Según la explicación, ¿qué palabra pesó más para decidir que este correo es phishing?").setChoiceValues(["account", "transffered", "inbox", "payment"]).setRequired(true);
  item = form.addMultipleChoiceItem();
  item.setTitle("C2. Según la explicación, ¿qué parte del correo pesó más en esta decisión?").setChoiceValues(["Enlaces y verificación del remitente", "Las tres pesaron lo mismo", "Texto del correo", "Diseño del correo (HTML)"]).setRequired(true);
  item = form.addGridItem();
  item.setTitle("Tu valoración de la explicación del correo 1 (1 a 5)").setRows(["E1. Entendí por qué el sistema tomó esta decisión.", "E2. Esta explicación me ayudaría a decidir si confiar en la decisión del sistema."]).setColumns(["1 (totalmente en desacuerdo)", "2", "3", "4", "5 (totalmente de acuerdo)"]).setRequired(true);
  form.addPageBreakItem().setTitle("Correo 2 de 4");
  form.addImageItem().setTitle("Correo 2").setHelpText("Asunto: Re: how to enable/start \"git@.service\" via systemd?").setImage(UrlFetchApp.fetch("https://raw.githubusercontent.com/Irico17/Tesis/master/doc/validacion_xai/formulario/resaltado_legitimo_1.png"));
  form.addSectionHeaderItem().setTitle("Explicación del sistema: correo 2").setHelpText("Decisión del sistema: LEGÍTIMO (correo normal), con 100.0% de confianza.\n\n¿En qué se basó?\n• Texto del correo: 100.0% del peso. Lo que más pesó:\n    1. «fedora»: 45.0%\n    2. «invocations»: 14.8%\n    3. «wrote»: 14.7%\n    4. «libexec»: 13.0%\n    5. la abreviatura «cdt»: 12.6%\n• Diseño del correo (HTML): no influyó en esta decisión.\n• Enlaces y verificación del remitente: no influyó en esta decisión.");
  item = form.addMultipleChoiceItem();
  item.setTitle("C3. Según la explicación, ¿qué palabra pesó más para decidir que este correo es legítimo?").setChoiceValues(["socket", "invocations", "fedora", "daemon"]).setRequired(true);
  item = form.addGridItem();
  item.setTitle("Tu valoración de la explicación del correo 2 (1 a 5)").setRows(["E3. Entendí por qué el sistema tomó esta decisión.", "E4. Esta explicación me ayudaría a decidir si confiar en la decisión del sistema."]).setColumns(["1 (totalmente en desacuerdo)", "2", "3", "4", "5 (totalmente de acuerdo)"]).setRequired(true);
  form.addPageBreakItem().setTitle("Correo 3 de 4");
  form.addImageItem().setTitle("Correo 3").setHelpText("Asunto: RE: ITS HELP DESK: PASSWORD UPDATE").setImage(UrlFetchApp.fetch("https://raw.githubusercontent.com/Irico17/Tesis/master/doc/validacion_xai/formulario/resaltado_phishing_2.png"));
  form.addSectionHeaderItem().setTitle("Explicación del sistema: correo 3").setHelpText("Decisión del sistema: PHISHING (intento de fraude), con 95.6% de confianza.\n\n¿En qué se basó?\n• Texto del correo: 99.6% del peso. Lo que más pesó:\n    1. el signo de puntuación «*»: 23.3%\n    2. «expires»: 20.5%\n    3. «account»: 20.3%\n    4. «helpdesk»: 18.1%\n    5. «the»: 17.5%\n• Diseño del correo (HTML): no influyó en esta decisión.\n• Enlaces y verificación del remitente: 0.4% del peso. Lo que más pesó:\n    1. la cantidad de subdominios en las URLs: 0.2%\n    2. la cantidad de URLs reportadas: 0.1%");
  item = form.addMultipleChoiceItem();
  item.setTitle("C4. Además del texto, ¿qué otra cosa influyó en esta decisión, según la explicación?").setChoiceValues(["la longitud del mensaje", "el uso de un acortador de enlaces", "la presencia de código HTML", "la cantidad de subdominios en las URLs"]).setRequired(true);
  item = form.addMultipleChoiceItem();
  item.setTitle("C5. Según la explicación, la palabra «helpdesk» pesó más que «expires».").setChoiceValues(["Verdadero", "Falso"]).setRequired(true);
  item = form.addGridItem();
  item.setTitle("Tu valoración de la explicación del correo 3 (1 a 5)").setRows(["E5. Entendí por qué el sistema tomó esta decisión.", "E6. Esta explicación me ayudaría a decidir si confiar en la decisión del sistema."]).setColumns(["1 (totalmente en desacuerdo)", "2", "3", "4", "5 (totalmente de acuerdo)"]).setRequired(true);
  form.addPageBreakItem().setTitle("Correo 4 de 4");
  form.addImageItem().setTitle("Correo 4").setHelpText("Asunto: Mysterious autostart behaviour").setImage(UrlFetchApp.fetch("https://raw.githubusercontent.com/Irico17/Tesis/master/doc/validacion_xai/formulario/resaltado_legitimo_2.png"));
  form.addSectionHeaderItem().setTitle("Explicación del sistema: correo 4").setHelpText("Decisión del sistema: LEGÍTIMO (correo normal), con 100.0% de confianza.\n\n¿En qué se basó?\n• Texto del correo: 100.0% del peso. Lo que más pesó:\n    1. «kde»: 23.9%\n    2. «qbittorrent»: 21.8%\n    3. «poc»: 19.0%\n    4. «autostart»: 17.8%\n    5. «ktorrent»: 17.4%\n• Diseño del correo (HTML): no influyó en esta decisión.\n• Enlaces y verificación del remitente: no influyó en esta decisión.");
  item = form.addMultipleChoiceItem();
  item.setTitle("C6. Según la explicación, ¿qué decidió el sistema sobre este correo?").setChoiceValues(["Phishing, con más del 99% de confianza", "Legítimo, con entre 60% y 90% de confianza", "Legítimo, con más del 99% de confianza", "Phishing, con entre 60% y 90% de confianza"]).setRequired(true);
  item = form.addGridItem();
  item.setTitle("Tu valoración de la explicación del correo 4 (1 a 5)").setRows(["E7. Entendí por qué el sistema tomó esta decisión.", "E8. Esta explicación me ayudaría a decidir si confiar en la decisión del sistema."]).setColumns(["1 (totalmente en desacuerdo)", "2", "3", "4", "5 (totalmente de acuerdo)"]).setRequired(true);
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
