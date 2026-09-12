# La tesis explicada

Documento interno, escrito para el autor. No forma parte del entregable y no se
cita en él. Su objeto es que puedas explicar tu propio trabajo sin recurrir al
capítulo, entendiendo por qué cada pieza está donde está y qué sostiene de
verdad cada cifra.

Corresponde a la ejecución del 8 y 9 de septiembre de 2026, que es la que el
documento reporta. Todas las cifras que aparecen aquí salen de los informes bajo
`medios_de_verificacion/`.

---

## 1. De qué va esto, en una página

Un correo de phishing no es solo texto. Trae, además, un cuerpo con marcado
(HTML), unos enlaces, y unas cabeceras de tránsito que dicen por dónde pasó el
mensaje y si superó las comprobaciones de autenticación (SPF, DKIM, DMARC). La
mayor parte de los detectores publicados mira solo el texto.

La tesis construye un modelo que mira las tres cosas a la vez y decide con las
tres. Eso es lo que quiere decir **multimodal**: tres fuentes de información
distintas que se combinan dentro del mismo modelo.

El problema de fondo, y lo que da sentido a toda la arquitectura, es que **esas
tres fuentes casi nunca están todas presentes**. Un correo reenviado pierde las
cabeceras. Un correo en texto plano no tiene marcado. Un modelo multimodal
ingenuo, que concatena las tres representaciones y las pasa por una capa densa,
se rompe cuando una falta: le llegan ceros y no sabe distinguir «esta modalidad
vale cero» de «esta modalidad no existe».

**La contribución no es la multimodalidad. Es el mecanismo de fusión que sabe
que una modalidad puede faltar.** Esto conviene tenerlo muy claro, porque es la
respuesta a la primera pregunta que te van a hacer.

---

## 2. El corpus, y la verdad incómoda

### Cómo está hecho

Ocho colecciones públicas, 44 100 correos, prevalencia 0.402.

| Colección | Formato | Mensajes | Clase |
|---|---|---:|---|
| Kaggle | tabular (CSV) | 17 409 | ambas |
| Fedora | mbox | 13 158 | legítimo |
| phishing_pot | .eml | 5 960 | phishing |
| Nazario | mbox | 4 164 | phishing |
| datacon2023 | tabular (JSONL) | 1 084 | phishing |
| SpamAssassin | .eml | 812 | legítimo |
| kernel_lists | mbox | 759 | legítimo |
| CEAS_08 | tabular (JSONL) | 754 | legítimo |

De esos 44 100, **20 098 tienen las tres modalidades** (45.57 %) y 28 558 tienen
texto y al menos una no textual (64.76 %). Eso es lo que exige el indicador de
R1.1 y por eso el corpus se aceptó.

### El submuestreo, que hay que declarar

Las colecciones disponibles aportan mucho más correo legítimo que fraudulento.
Para que la prevalencia cayera en el rango comprometido (0.40 a 0.60), el
pipeline **descarta 206 774 correos legítimos** al azar, con semilla fija. No se
descarta ningún phishing ni se genera nada sintético.

Esto tiene dos consecuencias que el capítulo ahora declara y que tú debes
declarar también si te preguntan:

1. El criterio de prevalencia **se cumple por construcción, no por medición**.
   Acredita que el corpus se ensambló como se prometió; no dice nada sobre cómo
   se reparten las clases en el correo real, donde el phishing es una fracción
   mucho menor.
2. Ninguna cifra absoluta del capítulo describe una pasarela de correo real. Por
   eso se informa además la detección a tasas de falsos positivos del 1 % y del
   0.1 %, que sí conserva sentido operativo.

Si te preguntan por qué equilibraste: porque el trabajo **compara arquitecturas
entre sí**, y con una prevalencia muy baja la varianza de las métricas se
concentra en unos pocos positivos, de modo que las diferencias entre modelos
quedarían dominadas por el ruido de muestreo antes que por la arquitectura.

### La verdad incómoda: clase ≈ procedencia

Esta es la limitación central de la tesis y conviene que la digas tú antes de que
la diga el jurado.

**De las ocho colecciones, solo una (Kaggle) contiene mensajes de ambas clases.**
Las otras siete son de clase única. Eso significa que, en la mayor parte del
corpus, saber de dónde viene un mensaje equivale a saber su clase.

Se midió de tres maneras y las tres dan lo mismo:

- **Auditoría de características.** Para cada característica no textual se
  compara cuánta información aporta sobre la clase (fijada la colección) contra
  cuánta aporta sobre la colección. En las **quince**, la segunda aplasta a la
  primera. El mayor cociente es 0.4338, y corresponde al uso de un dominio de
  primer nivel infrecuente. Ninguna llega a 1.
- **Suelo del confusor.** Un clasificador que solo ve **siete banderas de
  disponibilidad** (si el mensaje trae estructura, red, SPF, DKIM, DMARC,
  remitente, fecha), **sin mirar el contenido**, alcanza F1 0.6388 y ROC-AUC
  **0.8378**. Ese es el suelo: cualquier modelo debe superarlo con holgura para
  que su cifra signifique algo.
- **Idioma.** El 16.37 % del corpus no está en inglés. En inglés la prevalencia
  es 0.3959; en siete de los nueve idiomas restantes con muestra suficiente es
  **≥ 0.95**, y en varios exactamente 1.0000. Estar escrito en otro idioma
  equivale casi a ser phishing. El idioma informa 33 veces más sobre la colección
  que sobre la clase.

La consecuencia práctica: **tu 0.9927 de F1 no es «detecta phishing el 99.27 % de
las veces»**. Es «sobre este material, separa las clases con ese acierto, y una
parte de esa separación es reconocimiento de procedencia». Decirlo tú te da
credibilidad; que te lo saquen te la quita.

---

## 3. La arquitectura, pieza a pieza

Todo lo que sigue está en `src/phishing_model/model.py`. La dimensión interna de
trabajo es **d = 256** y la atención usa **8 cabezas**.

### 3.1 Tres ramas de entrada

**Rama de texto.** Un DistilBERT (`distilbert-base-uncased`) que se ajusta
completo durante el entrenamiento. Lo importante: **no se toma el vector [CLS],
se expone la secuencia completa de estados ocultos**. Eso es lo que permite que
la atención cruzada apunte a palabras concretas y no a un resumen del mensaje.

**Rama de estructura.** Seis características del árbol del documento (número de
nodos, profundidad, enlaces, imágenes, si hay marcado, si hay enlaces por
dirección numérica).

**Rama de red.** Nueve características continuas más varias categóricas (los
resultados de SPF, DKIM y DMARC).

Las dos ramas no textuales no se aplanan en un vector: **cada característica se
convierte en un token propio**, siguiendo la tokenización de datos tabulares de
Gorishniy et al. (2021). Así la atención puede señalar *qué* característica pesó,
que es lo que hará falta cuando llegue la fase de explicabilidad.

### 3.2 La fusión, en dos etapas

**Etapa 1, autoatención entre modalidades.** Los tokens de estructura y de red se
miran entre sí. Aquí es donde el modelo puede aprender, por ejemplo, que «muchos
enlaces» importa más cuando además «SPF falla».

**Etapa 2, atención cruzada.** El texto actúa como **consulta** y el resultado de
la etapa 1 como **clave y valor**. Es decir: el texto pregunta, las otras
modalidades responden.

### 3.3 La compuerta modal, que es la pieza clave

La salida de la fusión no es lo que devuelve la atención cruzada. Es esto:

```
fusionada = texto + tanh(g) · (atención_cruzada(texto, memoria) − texto)
```

donde `g` es un parámetro aprendido.

Léelo despacio, porque es la pieza que más preguntas genera:

- Si `tanh(g) = 0`, la fusión devuelve el texto tal cual. El modelo se comporta
  como un unimodal de texto.
- Si `tanh(g) = 1`, devuelve la atención cruzada completa.
- En medio, interpola.

**El texto es la base residual.** Las modalidades no textuales no aportan una
representación propia: aportan una *corrección* sobre la del texto. Y como `g`
es un único escalar aprendido, `tanh(g)` es una **medida directa de cuánto usa el
modelo las modalidades no textuales**, comparable entre corridas.

De aquí sale el tercer hallazgo del capítulo: **la arquitectura está anclada al
texto por diseño**. Si anulas el texto por completo, no queda camino desde las
otras modalidades hasta la decisión y el desempeño cae al de un clasificador
trivial. Eso no es un defecto: la investigación se define sobre correo
electrónico, que siempre tiene cuerpo. Pero delimita el sobre operativo, y por
eso está en las limitaciones.

### 3.4 El centinela, que es una salvaguarda numérica

Si un mensaje no trae ninguna modalidad no textual, la memoria de atención queda
vacía. Un softmax sobre un conjunto vacío de claves devuelve **NaN** y el
entrenamiento revienta.

La solución: un **token centinela** aprendido, que se antepone siempre a la
memoria y nunca se enmascara. Así ninguna fila puede quedarse sin claves.

Hay un detalle que conviene que sepas porque habla bien del trabajo. La versión
anterior resolvía lo mismo desenmascarando la posición 0 de la memoria. Esa
posición es el token de la primera característica estructural, que es un vector
**con contenido aprendido**, no un relleno neutro. El modelo atendía a un token
con contenido para filas que por definición no tenían contenido que mostrar. En
este corpus no transportaba información, porque esa primera característica es
`has_html` y la disponibilidad estructural se define justamente como
`has_html == 1`, de modo que su valor era constante en las filas afectadas. Pero
**la corrección dependía de esa coincidencia**: reordenar las columnas la
convertía en una fuga real y silenciosa. Con el centinela, el aislamiento se
sostiene por construcción.

Si te preguntan «¿el centinela mejora el desempeño?»: **no, y se midió**. Con
todas las modalidades presentes, centinela y descarte (0.9980), descarte sin
centinela (0.9975) y mezcla de expertos (0.9970) son equivalentes dentro del
margen de 0.005. El centinela es una salvaguarda numérica, no una propuesta
arquitectónica.

### 3.5 Dropout de modalidad

Durante el entrenamiento, cada rama disponible se suprime con probabilidad 0.15.
Es lo que enseña al modelo a operar con modalidades ausentes, en lugar de
descubrirlo en despliegue.

Hay una alternativa implementada, `randomize`, que sortea el patrón de
disponibilidad de la distribución marginal del corpus. Se midió: la información
mutua entre el patrón de disponibilidad y la etiqueta (el atajo) baja de 0.0790
nats a 0.0235 con aleatorización, frente a 0.0602 con descarte independiente. **La
aleatorización reduce el atajo un 70 %, pero no lo elimina**, porque el patrón
sorteado se intersecta con la disponibilidad real y no puede añadirse una
modalidad que la fila no tiene.

### 3.6 Las variantes que se comparan

Seis arquitecturas, entrenadas todas bajo condiciones idénticas:

| Variante | Qué hace |
|---|---|
| Atención cruzada por token | La propuesta: el texto consulta token a token |
| Atención cruzada por modalidad | Igual, pero el texto se resume antes en un vector |
| Mezcla de expertos | Cuatro expertos con una compuerta que pondera según las banderas |
| Fusor MLP | Concatenación y capa densa, sin atención |
| Concatenación tardía | Se concatenan las decisiones, no las representaciones |
| Unimodal de texto | Solo la rama textual |

Más cuatro líneas base clásicas: B1 (TF-IDF con regresión logística), B3 (bosque
aleatorio sobre estructura), B4 (bosque aleatorio sobre red) y B5 (fusión clásica
de las tres).

---

## 4. Qué dicen los experimentos, en cristiano

Ocho experimentos, cada uno con una pregunta. Van en una cola secuencial porque
comparten GPU y los tiempos deben ser comparables.

### E0 — ¿Sirve el corpus?

Sí, en los cinco criterios. Ya está comentado arriba, incluido el matiz del
submuestreo.

### E1 — ¿Aporta la multimodalidad?

**Aquí está el resultado que más te van a discutir, y tiene dos caras.**

Sobre el **subconjunto trimodal** (2 011 mensajes de prueba), la propuesta supera
al unimodal de texto por 0.0012 de F1, con p = 0.6250. No significativo.

Pero mira el dato que importa: esa prueba se calcula sobre **4 discordancias**.
Cuatro mensajes en los que los dos modelos difieren. Con cuatro casos ninguna
prueba podría alcanzar significancia, exista o no una diferencia real. **Lo que
se observa no es que sean equivalentes: es que este subconjunto no permite
distinguirlos.**

Sobre el **corpus completo** (4 411 mensajes de prueba), la misma comparación da
una diferencia de 0.0037, **p = 0.0044**, sobre **19 discordancias** (16 aciertos
exclusivos de la propuesta frente a 3 del unimodal), con intervalo de confianza
[0.0014; 0.0060], **por encima** del margen de indiferencia de 0.005.

¿Por qué la diferencia? Dos motivos, y los dos cuentan:

1. **Resolución.** El trimodal aporta menos de la cuarta parte de las
   discordancias.
2. **Composición.** En el trimodal, las tres modalidades están presentes en todos
   los mensajes. En el corpus completo faltan de forma desigual, **que es
   exactamente la situación que la fusión propuesta gestiona**.

Y la magnitud es tres veces mayor sobre el corpus completo, así que no es solo
cuestión de tener más mensajes para medir.

**La frase que debes tener preparada:** «La fusión no obtiene ventaja donde todas
las modalidades están presentes y el texto ya resuelve la tarea, porque ahí no
hay margen donde demostrarla. Sí la obtiene sobre el conjunto que describe el
problema, donde las modalidades faltan de forma desigual.»

### E2 — ¿Importa el mecanismo de fusión?

No. Los cuatro mecanismos contrastados frente al fusor MLP resultan
**equivalentes** dentro del margen de 0.005.

La lectura correcta no es que la fusión dé igual, sino que **sobre un conjunto
donde el texto ya resuelve la tarea con altísimo acierto, ningún mecanismo
dispone de margen donde demostrar superioridad**. La consecuencia práctica: en
condiciones parecidas, conviene el mecanismo más simple.

### E3 — ¿Tolera que falten entradas?

Sí, para las no textuales. Ocultando modalidades solo en evaluación:

| Condición | F1 | Caída relativa |
|---|---:|---:|
| Las tres | 0.9980 | 0.0000 |
| Sin estructura | 0.9980 | 0.0000 |
| Sin red | 0.9943 | 0.0037 |
| Solo texto | 0.9931 | 0.0049 |

Menos de medio punto porcentual en el peor caso. **Esto es lo que la arquitectura
existe para lograr.**

Ante manipulación adversaria del texto (homóglifos, caracteres de ancho cero,
entidades HTML, truncamiento) la caída **sí** es sustancial, y **no menor que la
de la fusión clásica**. Eso es un resultado negativo y el capítulo lo informa como
tal: la robustez adversaria no es una fortaleza del trabajo.

### E4 — ¿Generaliza?

Sobre correo no visto de las mismas colecciones, sí. Y es estable: tres
reparticiones independientes del corpus dan una desviación de **0.0008** en F1.

La curva de aprendizaje (25 %, 50 %, 75 %, 100 % → 0.9870, 0.9907, 0.9901,
0.9932) crece de forma amortiguada y no muestra sobreajuste. **Cuidado**: no
digas que «ha saturado». El valor más alto es el último y con cuatro puntos y una
corrida por punto no se puede afirmar saturación. El capítulo ya se corrigió en
esto.

El cuadro comparativo completo, que es la tabla que caracteriza al sistema:

| Modelo | F1 | ROC-AUC | TPR@FPR 1 % | TPR@FPR 0.1 % |
|---|---:|---:|---:|---:|
| Fusor MLP | 0.9931 | 0.9996 | 0.9961 | 0.9464 |
| **Atención cruzada por token (propuesta)** | **0.9927** | 0.9995 | 0.9966 | 0.9532 |
| Mezcla de expertos | 0.9923 | 0.9996 | 0.9961 | **0.9695** |
| Atención cruzada por modalidad | 0.9916 | 0.9996 | 0.9972 | 0.9560 |
| Concatenación tardía | 0.9915 | 0.9990 | 0.9972 | 0.8866 |
| Unimodal de texto | 0.9895 | 0.9992 | 0.9944 | 0.9233 |
| B5 fusión clásica | 0.9823 | 0.9985 | 0.9791 | 0.7625 |
| B1 TF-IDF + regresión logística | 0.9815 | 0.9982 | 0.9780 | 0.6588 |
| B4 bosque sobre red | 0.7280 | 0.9001 | 0.5544 | 0.4963 |
| B3 bosque sobre estructura | 0.6949 | 0.8473 | 0.4687 | 0.2521 |
| Unimodal de estructura | 0.6050 | 0.7261 | 0.2459 | 0.0519 |
| Unimodal de red | 0.6010 | 0.8164 | 0.4337 | 0.2989 |

**Mira la última columna, no la primera.** Al umbral de 0.5, todos los modelos
con texto parecen iguales. Al exigir una tasa de falsos positivos del 0.1 %, que
es la condición de una pasarela real, se abren: los neuronales con texto van de
0.8866 a 0.9695, los clásicos con texto de 0.6588 a 0.7625, y los que no reciben
texto no pasan de 0.4963.

**Sé honesto con esto:** tu modelo obtiene 0.9532 en ese punto, que **no es el
más alto**. El máximo, 0.9695, es de la mezcla de expertos. La lectura
transferible es que informar solo F1 en una tarea saturada oculta esa distancia,
no que tu arquitectura domine.

### E5 — ¿Cuánto es fenómeno y cuánto procedencia?

Ya está en la sección 2. Es la sección que califica a todas las demás.

Un detalle sobre el diagnóstico por colección: se deja fuera cada colección
entera y se evalúa sobre ella. **Siete de las ocho no son evaluables**, porque
contienen una sola clase. La única evaluable, Kaggle, da F1 **0.7448** frente al
**0.9815** que el mismo tipo de clasificador obtiene sobre la partición agrupada.

Ojo: ese diagnóstico usa una **sonda ligera** (TF-IDF con regresión logística),
no el modelo propuesto, porque hay que reentrenar una vez por colección. Por eso
se compara contra B1 y no contra tu 0.9927. El capítulo tenía mal esta
comparación y se corrigió.

### E6 — ¿Es desplegable?

Sí. La propuesta, exportada a ONNX y cuantizada a enteros de 8 bits:

- **73.6 ms** de latencia media por mensaje, **78.6 ms** en el percentil 95, sobre
  CPU y sin agrupar mensajes.
- **66.0 MB** frente a los 259.3 MB de la versión en coma flotante (−74.5 %).
- Aceleración de 1.45× en la mediana.
- **Acuerdo total** entre ambas versiones, salvo el fusor MLP (0.9967).

Dos matices que ahora el capítulo declara: la latencia se mide sobre **50
mensajes**, de modo que el percentil 95 es poco estable; y los modelos exportados
proceden de **una sola semilla (42)**, mientras que las métricas de R2.2 son
medias de tres.

### E7 — ¿Convendría un codificador multilingüe?

No con este material. Se comparó DistilBERT contra `distilbert-base-multilingual-cased`,
dos arquitecturas × dos codificadores × tres semillas, evaluando por separado los
3 932 mensajes en inglés y los 430 que no lo están.

**Ninguno de los seis contrastes alcanza significancia.** Sobre el subconjunto en
inglés, que es donde una ventaja sería interpretable, la diferencia es +0.0019 y
la prueba no la distingue del azar.

Y hay un hallazgo colateral buenísimo: **el codificador monolingüe obtiene F1
0.9961 sobre mensajes que no puede ni segmentar** (chino, vietnamita). Un modelo
que no puede leer un mensaje no debería clasificarlo casi perfecto. Que lo
consiga demuestra que **no está leyendo el contenido, sino reconociendo que el
texto le resulta ajeno** — y en este corpus, resultar ajeno equivale a ser
phishing. Es la coincidencia entre clase y procedencia, vista por otra vía.

---

## 5. Los tres hallazgos, y por qué son uno solo

El capítulo declara tres hallazgos no previstos:

1. **El mecanismo de fusión no determina el desempeño.** Todos equivalentes.
2. **El punto de operación separa a los modelos por su representación del texto**,
   no por cómo fusionan.
3. **La arquitectura está anclada al texto por diseño.**

Y la discusión los une: **los tres tienen la misma causa**, que es que la clase
coincide en gran medida con la procedencia. No son tres limitaciones
independientes, sino una sola observada por tres caminos.

---

## 6. Qué puedes afirmar y qué no

**Puedes afirmar:**

- Que construiste un corpus multimodal de 44 100 correos con trazabilidad
  completa y cobertura declarada.
- Que la arquitectura tolera la ausencia de modalidades con una caída inferior al
  0.5 %, y que eso está medido.
- Que sobre el corpus completo la propuesta supera al unimodal de texto de forma
  significativa (p = 0.0044) y fuera del margen de indiferencia.
- Que mediste la validez de tu propia evaluación y encontraste que una parte
  sustancial de lo medido es procedencia, no fenómeno. **Esto es lo más valioso
  del trabajo**, aunque no lo parezca.
- Que el sistema es desplegable con latencia y tamaño medidos.

**No puedes afirmar:**

- Que el sistema detecte phishing al 99 % en producción.
- Que la atención cruzada sea superior a otros mecanismos de fusión.
- Que el modelo generalice a colecciones no vistas: no es medible con este
  material y el capítulo lo declara.
- Que haya saturado la curva de datos.
- Que sea más robusto que la fusión clásica ante ataques adversarios.

---

## 7. Lo que falta

Los tres resultados del tercer objetivo, R3.1 a R3.3, que son la explicabilidad,
**no se reportan**: están planificados para la fase siguiente.

Conviene que sepas decir esto: la arquitectura **ya expone los pesos de atención
cruzada** entre el texto y las modalidades no textuales, de modo que el candidato
intrínseco que R3.1 debe evaluar está disponible sin tocar el modelo. Esa
disponibilidad es un resultado del trabajo ya hecho, aunque su evaluación
comparativa corresponda a después.

---

## 8. Dónde está cada cosa

| Qué | Dónde |
|---|---|
| El modelo | `src/phishing_model/model.py` |
| Los hiperparámetros | `src/phishing_model/config.py` |
| Construcción del corpus | `src/phishing_pipeline/corpus_real.py` |
| Descarga de correo crudo | `src/phishing_pipeline/downloaders/` |
| Los ocho experimentos | `scripts/experimentos/e0_*.py` … `e7_*.py` |
| La cola que los ejecuta | `scripts/experimentos/cola_servidor.sh` |
| Figuras de los medios de verificación | `scripts/figuras/generar_figuras_mv.py` |
| La evidencia, por resultado | `medios_de_verificacion/R1.1_*` … `R2.3_*` |

Para reconstruirlo todo desde cero:

```bash
python -m phishing_pipeline.downloaders.correo_real
python -m phishing_pipeline.downloaders.listas_correo
python -m phishing_pipeline.corpus_real
bash scripts/experimentos/cola_servidor.sh
```

La cola tarda unas diecinueve horas en una RTX A5500 y, al terminar sin fallos,
consolida los medios de verificación y emite las figuras por su cuenta.
