# Transformers, atención cruzada y la arquitectura de esta tesis

Documento de referencia técnica. La primera mitad explica el mecanismo general —qué es
la atención, qué la distingue de la atención cruzada, cómo se compone un Transformer—.
La segunda describe la arquitectura concreta de este trabajo, con las formas de los
tensores, los ficheros donde vive cada pieza y las decisiones que la explican.

Todo lo que aquí se afirma sobre el modelo se corresponde con el código en
`src/phishing_model/`; los recuentos de parámetros están medidos, no estimados.

---

## Parte I — El mecanismo

### 1. El problema que la atención resuelve

Antes de los Transformers, procesar una secuencia significaba recorrerla. Una red
recurrente lee el token 1, actualiza un estado oculto, lee el token 2, lo actualiza otra
vez, y así hasta el final. Eso impone dos costos:

1. **Secuencialidad.** El token 500 no puede calcularse antes que el 499. No hay
   paralelismo dentro de una secuencia.
2. **Cuello de botella del estado.** Toda la información de los primeros 400 tokens tiene
   que caber en un único vector de estado para influir en el 401. Cuanto más lejos, más
   diluida llega.

La atención sustituye el recorrido por una consulta directa: **cada posición mira a todas
las demás a la vez** y decide, para cada una, cuánto le importa. No hay estado que
acumular ni distancia que recorrer: el token 500 accede al token 1 con exactamente el
mismo costo con que accede al 499.

### 2. Consulta, clave y valor

La operación central se formula con tres proyecciones de la entrada. Dada una secuencia
de vectores `X` de forma `[n, d]`, se calculan

```
Q = X · W_Q      (consultas — "qué estoy buscando")
K = X · W_K      (claves    — "qué ofrezco para ser encontrado")
V = X · W_V      (valores   — "qué entrego si me eligen")
```

La analogía que mejor funciona es la de un índice. Cada posición emite una **consulta**
que describe lo que necesita, y cada posición publica una **clave** que describe lo que
tiene. El producto escalar `q_i · k_j` mide cuánto encaja lo que busca `i` con lo que
ofrece `j`. Esos productos se normalizan a pesos que suman uno y se usan para promediar
los **valores**:

```
Atención(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

Tres detalles que importan:

- **El `√d_k`.** Sin él, el producto escalar de dos vectores de dimensión grande tiene
  varianza proporcional a `d_k`, el softmax se satura y el gradiente se anula. Dividir
  por `√d_k` mantiene la varianza en torno a uno.
- **La matriz `Q · Kᵀ` es de tamaño `[n, n]`.** De ahí el costo cuadrático en longitud de
  secuencia, que es la razón práctica de truncar los correos a 512 sub-palabras.
- **La salida tiene la misma forma que la entrada**, `[n, d]`. Cada posición sale
  reescrita como una mezcla ponderada de todas, ella incluida.

### 3. Múltiples cabezas

Una sola atención impone un único criterio de relevancia por capa. La atención
multicabeza divide `d` en `h` subespacios, ejecuta el mecanismo en cada uno con
proyecciones propias y concatena:

```
cabeza_i = Atención(X·W_Q^i, X·W_K^i, X·W_V^i)     con W^i de dimensión d/h
salida   = [cabeza_1 ; … ; cabeza_h] · W_O
```

Cada cabeza puede especializarse: una en concordancia sintáctica, otra en correferencia,
otra en proximidad. No se les asigna ese papel, lo adquieren si resulta útil. En esta
tesis, `n_heads = 8` sobre `d_model = 256`, de modo que cada cabeza opera en 32
dimensiones.

### 4. El bloque completo

Un bloque Transformer no es solo atención. Es:

```
x = x + Atención(LayerNorm(x))          # subcapa 1: mezcla entre posiciones
x = x + FFN(LayerNorm(x))               # subcapa 2: transforma cada posición por separado
```

donde `FFN(z) = W₂ · GELU(W₁ · z + b₁) + b₂` se aplica **idéntica e independientemente a
cada posición**. La división de trabajo es nítida: la atención mueve información *entre*
posiciones, la red densa la procesa *dentro* de cada una.

Las dos piezas restantes no son adorno:

- **La conexión residual** (`x + …`) crea un camino por el que el gradiente viaja sin
  atenuarse hasta las capas bajas. Sin ella, apilar bloques deja de funcionar.
- **La normalización por capa** estabiliza la escala de las activaciones.

**Pre-normalización frente a post-normalización.** La formulación original de Vaswani et
al. (2017) normaliza *después* de la subcapa (`x = LayerNorm(x + Atención(x))`). Esa
variante exige un calentamiento cuidadoso de la tasa de aprendizaje: con
post-normalización, la magnitud del gradiente que llega a las capas inferiores crece con
la profundidad y el entrenamiento diverge con facilidad. La pre-normalización, que es lo
que aquí se usa (`norm_first=True`), deja la trayectoria residual sin normalizar de
extremo a extremo y es la formulación estándar desde entonces. En esta arquitectura
importa por una razón adicional: las capas de fusión se inicializan al azar y se conectan
a un codificador ya preentrenado, de modo que la estabilidad de los primeros pasos decide
cuánto se degradan los pesos aprendidos.

### 5. Posición

La atención es **permutación-equivariante**: barajar la entrada baraja la salida sin
cambiar nada más. Para un texto eso es inaceptable, así que se suma a cada token un
vector que codifica su posición. DistilBERT usa posiciones aprendidas, una fila por índice
hasta 512.

Este punto tiene una consecuencia directa en la parte II: **los tokens de modalidad de
esta tesis no llevan codificación posicional**, porque no forman una secuencia. Lo que
necesitan es identidad, no orden, y esa identidad se les da de otro modo (§8).

### 6. Auto-atención frente a atención cruzada

Es la distinción que sostiene toda la arquitectura de este trabajo.

**Auto-atención.** `Q`, `K` y `V` se derivan de la *misma* secuencia. Cada posición mira
a sus compañeras. Es lo que hace un codificador BERT: contextualizar cada palabra con el
resto de su frase.

**Atención cruzada.** `Q` viene de una secuencia y `K`, `V` de **otra**:

```
CrossAttn(A, B) = softmax( (A·W_Q) · (B·W_K)ᵀ / √d_k ) · (B·W_V)
```

La secuencia `A` conserva su longitud y su papel —sigue siendo la que se está
transformando— y `B` actúa como memoria que se consulta. La salida tiene la forma de `A`,
no la de `B`.

Tres propiedades que se aprovechan aquí:

1. **Las dos secuencias pueden tener longitudes distintas.** 512 tokens de texto contra 19
   de modalidad, sin necesidad de igualarlas.
2. **La dirección es asimétrica.** Que el texto atienda a la modalidad no es lo mismo que
   lo inverso. Aquí el texto es la consulta porque es la representación que alimenta al
   clasificador: la modalidad la enriquece, no al revés.
3. **La memoria admite máscara por elemento y por fila.** Si un correo no tiene metadatos
   de red, se marcan esas posiciones como ausentes y quedan fuera del softmax. Esto es lo
   que permite manejar modalidades faltantes sin imputar valores. Es también el punto
   donde aparece el fallo numérico que la §12 explica: si *todas* las claves de una fila
   quedan enmascaradas, el softmax opera sobre un conjunto vacío y produce `NaN`.

El decodificador original de Vaswani et al. combina las dos: auto-atención sobre lo
generado hasta el momento, y atención cruzada sobre la salida del codificador. Esta tesis
usa **esa misma capa**, `nn.TransformerDecoderLayer`, pero no para generar secuencias sino
como mecanismo de fusión entre modalidades.

---

## Parte II — La arquitectura de esta tesis

### 7. Vista de conjunto

```
correo
  │
  ├── cuerpo íntegro ──► tokenizador ──► DistilBERT ──► proyección
  │                       (512 sub-palabras)   [B,512,768] ──► [B,512,256]
  │                                                                │
  ├── 6 características estructurales ──► StructuralTokenizer ──► [B,6,256]
  │                                                                │
  └── 9 continuas + 3 categóricas de red ──► NetworkTokenizer ──► [B,12,256]
                                                                   │
                                        [B,18,256] ── + centinela ──► [B,19,256]
                                                                   │
                                         ETAPA 1 · auto-atención entre modalidades
                                                                   │
                                                              [B,19,256]
                                                                   │
       texto [B,512,256] ══ ETAPA 2 · atención cruzada ═══════════►│
              (Q = texto, K/V = modalidad)                         │
                                                                   ▼
                                     texto fusionado [B,512,256]
                                                   │
                          compuerta: texto + g·(fusionado − texto)
                                                   │
                             promediado enmascarado ──► [B,256]
                                                   │
                                             clasificador ──► [B,2]
```

Ficheros: `encoders/text_encoder.py`, `encoders/structural_tokenizer.py`,
`encoders/network_tokenizer.py`, `fusion/modality_encoder.py`,
`fusion/cross_attention.py`, y el ensamblado en `model.py`.

### 8. Cómo entran los datos no textuales

Aquí está la decisión de diseño menos obvia y la que conviene poder defender.

Lo habitual con características tabulares es concatenarlas en un vector y pasarlo por una
red densa. Eso produce **un** vector por correo, que la atención cruzada solo podría
consultar como una única posición: no habría nada que atender, solo un valor que sumar.

En su lugar se aplica la tokenización de FT-Transformer (Gorishniy et al., 2021): **cada
característica se convierte en su propio token**. Para la característica `i` con valor
escalar `x_i`:

```
token_i = x_i · w_i + b_i + e_i
```

donde `w_i` y `b_i` son una proyección escalar→vector **propia de esa columna** —no una
matriz compartida— y `e_i` es un vector de identidad, entrada `i` de una tabla de
inmersión. El término de identidad es lo que permite al modelo distinguir «este token es
`num_links`» de «este token es `word_count`» aunque ambos valgan lo mismo. Es el análogo
funcional de la codificación posicional del texto, salvo que codifica *qué* característica
es y no *dónde* está.

Las tres variables categóricas de autenticación (`spf_result`, `dkim_result`,
`dmarc_result`) no se proyectan: cada una tiene su propia tabla de inmersión con **una
categoría explícita para el valor ausente**. Que «no hay resultado SPF» sea una categoría
aprendida y no un cero disfrazado importa: un cero es indistinguible de un resultado real
que valga cero.

Resultado: 6 tokens estructurales + 9 continuos de red + 3 categóricos = **18 tokens de
modalidad**, cada uno de dimensión 256.

**Escalado.** Las continuas se transforman por cuantiles con los percentiles calculados
**solo sobre la partición de entrenamiento** y aplicados a validación y prueba. Calcularlos
sobre el conjunto completo sería una fuga: la partición de prueba influiría en cómo se
representa a sí misma.

### 9. Etapa 1 — la fusión jerárquica

Los 18 tokens de modalidad pasan por un `nn.TransformerEncoderLayer`: **auto-atención
entre ellos**, antes de que el texto los vea.

Esto es lo que hace la fusión *jerárquica* y no una simple concatenación. La estructura y
la red se relacionan primero entre sí —`url_has_ip_link` puede modular lo que significa
`spf_result`, `num_links` puede contextualizar `total_nodos_dom`— y solo el resultado de
esa interacción entra en contacto con la dimensión semántica. Sin esta etapa, cada
característica llegaría al texto aislada de las demás.

### 10. Etapa 2 — la atención cruzada

`nn.TransformerDecoderLayer` con:

- `tgt` = la secuencia textual completa, `[B, 512, 256]` — **no** un vector agregado.
- `memory` = los 19 tokens de modalidad ya auto-atendidos.

Que el texto entre como secuencia completa y no agregado es el punto que hace que la
atención cruzada sea real *a nivel de token*: cada sub-palabra del correo consulta por
separado a los tokens de modalidad. La palabra «verificar» puede atender al resultado SPF
mientras «adjunto» atiende al número de enlaces. Si el texto entrara agregado, habría una
sola consulta para todo el correo y el mecanismo degeneraría en algo muy próximo a una
suma ponderada.

La capa aporta además su propia auto-atención sobre el texto antes de la cruzada, que es
la estructura estándar del decodificador.

### 11. La compuerta modal

La salida de la etapa 2 no reemplaza al texto: **se interpola con él**.

```
fusionada = texto + tanh(g) · (atención_cruzada(texto, memoria) − texto)
```

`g` es un único parámetro aprendido. Cuando `tanh(g) = 0`, el modelo es exactamente el de
solo texto; cuando vale 1, es la salida cruzada pura.

Esto no es una florritura de diseño, es **instrumentación**. El valor de `tanh(g)` es una
medida directa y comparable entre pliegues de cuánto emplea el modelo las modalidades no
textuales: evidencia cuantitativa sobre la pregunta central de la tesis, en lugar de una
inferencia indirecta a partir de métricas agregadas.

**Sobre la inicialización.** Flamingo (Alayrac et al., 2022), de donde procede la idea,
inicializa la compuerta en cero, de modo que el modelo arranca siendo exactamente el
textual. Aquí eso **no funciona**, y se comprobó antes de descartarlo: el gradiente que
llega a la atención cruzada es proporcional a `tanh(g)`, luego con `g = 0` las capas de
fusión reciben gradiente exactamente nulo en el primer paso. La compuerta sí recibe
gradiente (0.0184 medido), pero solo puede evaluar una atención que sigue en su
inicialización aleatoria y que, por serlo, no aporta señal: tras 30 pasos la compuerta se
había movido a −0.000276, es decir, se cerraba en lugar de abrirse. Flamingo tolera ese
arranque frío porque entrena con órdenes de magnitud más pasos; aquí el presupuesto es de
tres épocas. Se inicializa por tanto en 0.5, punto medio que deja a la compuerta libre de
crecer o decrecer y hace que su valor final sea informativo por no estar sesgado por el
arranque.

**Lo que midió.** La compuerta termina en 0.5005 de media sobre las nueve corridas, con
recorrido de 0.4995 a 0.5014. Teniendo libertad para aumentar el peso de las modalidades,
el entrenamiento la dejó donde estaba.

### 12. Qué ocurre cuando faltan ramas

Es la parte que el asesor pidió explícitamente y la que más consecuencias tiene. Opera en
tres niveles.

**(a) Máscara de disponibilidad.** Cada correo trae dos banderas, `has_structure` y
`has_network`. La bandera se replica a lo largo de todos los tokens de su rama, de modo
que una rama ausente se excluye **en bloque** del cálculo de atención. No se imputa nada:
las posiciones sencillamente no participan en el softmax, y su peso es cero por
construcción y no por aprendizaje.

Se descartaron expresamente las alternativas:

| Alternativa | Por qué no |
|---|---|
| Imputar valores | Introduce observaciones que el modelo no puede distinguir de las reales, y atribuye evidencia a metadatos inexistentes. Inaceptable en un sistema que debe explicar sus decisiones. |
| Usar solo registros completos | El subconjunto con las tres modalidades tenía prevalencia 0.979 antes de incorporar los archivos de listas de discusión, es decir, era casi todo phishing e inservible para entrenar un clasificador binario. Incorporar esas colecciones es lo que lo equilibra, pero el corpus completo sigue conteniendo mensajes sin estructura ni red y descartarlos perdería el 54% de las muestras. |
| Un modelo por subconjunto | Multiplica los modelos a mantener y no produce representación compartida, que es justamente el objeto de la arquitectura. |

**(b) Token centinela.** Si una fila carece de *toda* modalidad no textual, la memoria
queda enteramente enmascarada y el softmax opera sobre un conjunto vacío: `NaN`. Se
resuelve anteponiendo un vector aprendido de «sin modalidad» que **acompaña siempre a la
memoria y nunca se enmascara**, de modo que la memoria nunca está vacía.

Merece la pena registrar qué sustituyó, porque es un fallo instructivo. La salvaguarda
anterior desenmascaraba la posición 0 de la memoria. Pero esa posición es, en la variante
a nivel de token, el token de la primera característica estructural: un vector con
contenido aprendido —norma L2 de 16.17 medida incluso con entrada nula—, no un relleno
neutro. El modelo atendía así a un token con contenido para filas que, por definición, no
tienen contenido que mostrar. La fuga no transportaba información en el corpus vigente,
porque la primera característica estructural es `has_html` y la disponibilidad estructural
se define precisamente como `has_html == 1`, de modo que su valor era constante en las
filas afectadas. Pero la propiedad dependía de ese detalle: reordenar las columnas o
cambiar la definición de disponibilidad la habría convertido en una fuga real y
silenciosa. Con el centinela el aislamiento se sostiene por construcción y no por
coincidencia, y una prueba automatizada verifica que alterar las características de una
rama declarada ausente no modifica la predicción.

**(c) Aleatorización del patrón durante el entrenamiento.** Este es el nivel que responde
al riesgo de fondo. Si la disponibilidad de modalidad correlaciona con la etiqueta —y en
correo público correlaciona—, el modelo puede aprender a leer *la presencia del campo* en
lugar de su contenido, que es aprender la procedencia del dato y no el fenómeno.

Se implementaron dos formulaciones:

- **Descarte independiente.** Cada rama disponible se suprime con probabilidad fija.
- **Aleatorización del patrón** (la que se usa). Para cada correo se sortea un patrón
  completo de disponibilidad extraído de la distribución marginal del corpus, **con
  independencia de su etiqueta**.

La diferencia no es de intensidad sino de naturaleza. El descarte suprime ramas, pero deja
intacta la asociación entre patrón y etiqueta en las filas que no suprime: con
probabilidad 0.15, el 85% de las observaciones conserva su patrón original y con él su
correlación con la clase. Se comprobó: elevar la probabilidad a 0.30, 0.45 y 0.60 no
mejora nada y degrada el área bajo la curva, porque destruye señal útil sin romper la
asociación. **Descartar no equivale a aleatorizar.**

Un matiz que conviene declarar: el mecanismo **no anula** la dependencia. El patrón
sorteado no puede *añadir* una modalidad que la fila no posee —no hay contenido que
mostrar—, de modo que la máscara resultante se intersecta con la disponibilidad natural, y
esa intersección deja pasar la parte de la dependencia que proviene de la disponibilidad
real. La reducción es del 70%, no la eliminación.

En inferencia ambos mecanismos quedan desactivados: las máscaras naturales se devuelven
sin modificar.

### 13. ¿Se está reentrenando DistilBERT?

**Sí. Ajuste fino completo, de principio a fin, en cada paso.** `freeze_text_encoder =
False` y ningún parámetro tiene `requires_grad = False`.

Las cifras, medidas sobre el modelo instanciado:

| Componente | Parámetros | Cuota |
|---|---:|---:|
| DistilBERT | 66,362,880 | 97.7% |
| Proyección 768→256 | 196,864 | 0.29% |
| Tokenizador estructural | 4,608 | 0.007% |
| Tokenizador de red | 12,800 | 0.019% |
| Etapa 1 (auto-atención modal) | 527,104 | 0.78% |
| Etapa 2 (atención cruzada) | 790,784 | 1.16% |
| Clasificador | 1,026 | 0.002% |
| **Total** | **67,896,323** | 100% |

**Toda la maquinaria de fusión que esta tesis propone son 1,533,443 parámetros: el 2.3%
del modelo.** La variante de solo texto tiene 66,560,770 parámetros; la multimodal,
67,896,323. Difieren en un 2%.

Ese dato no es anecdótico: es una de las claves para leer el resultado del Capítulo 4. Que
dos modelos que comparten el 97.7% de sus pesos y el mismo régimen de optimización
alcancen 0.9003 y 0.9004 de F1 no debería sorprender tanto como sorprende a primera vista.
La pregunta que el experimento responde no es si el 2% adicional puede aprender algo, sino
si tiene algo que aprender en estos datos —y la respuesta medida es que no.

**Tasas de aprendizaje diferenciadas.** El codificador se ajusta a `2e-5` y todo lo demás
a `1e-4`, cinco veces más. La razón es que DistilBERT llega con representaciones útiles ya
formadas y las capas nuevas llegan con ruido: aplicar a ambos la misma tasa alta destruiría
lo primero antes de que lo segundo aprendiera nada. El calentamiento lineal durante el
primer 10% de los pasos atiende el mismo riesgo desde otro ángulo, evitando que los
gradientes de gran magnitud de las capas aleatorias lleguen al codificador en los primeros
pasos.

**Exenciones del decaimiento de peso.** Los sesgos y los parámetros de normalización por
capa quedan exentos. El decaimiento penaliza la norma para limitar la capacidad efectiva,
razonamiento válido para las matrices de pesos pero no para estos: la escala y el
desplazamiento de una normalización son parámetros de calibración cuyo valor de reposo es
1 y 0, de modo que empujarlos hacia cero altera la normalización en lugar de regularizarla.
Afecta a 84 de los 147 tensores entrenables.

**Punto de partida.** `distilbert-base-uncased`, seis capas, 768 dimensiones ocultas, 12
cabezas, 66M de parámetros: una destilación de BERT-base que conserva en torno al 97% de
su desempeño con el 60% de su tamaño. La elección responde a la restricción de cómputo
declarada en la viabilidad del proyecto.

### 14. Las cuatro variantes

`fusion_type` selecciona entre cuatro configuraciones que comparten todo lo demás. Existen
porque cada una refuta una explicación alternativa concreta del resultado.

| Variante | Qué hace | Qué explicación descarta |
|---|---|---|
| `cross_attention_token_level` | La propuesta: etapas 1 y 2 completas | — |
| `cross_attention_modality_level` | Cada rama se colapsa a 1 token antes de la cruzada; sin etapa 1 | Que la riqueza de tokens individuales sea lo que aporta |
| `concat_late_fusion` | Las tres ramas se agregan por separado y se concatenan; sin atención ni compuerta | Que el mecanismo de fusión importe frente a la mera disponibilidad de las modalidades |
| `text_only` | Solo la rama textual | Que las modalidades no textuales aporten algo |

`concat_late_fusion` merece una nota. No es una variante menor de la atención cruzada: es
arquitectónicamente distinta, con una cabeza de clasificación de entrada triple (`3·256`)
y **vectores aprendidos de ausencia** en lugar del vector nulo que devolvería el promediado
enmascarado —un vector de ceros sería indistinguible de una rama cuyas características son
legítimamente nulas—. Que dos maneras tan distintas de incorporar las modalidades lleguen
al mismo resultado que no incorporarlas es lo que descarta que la atención cruzada
estuviese mal implementada o mal parametrizada.

### 15. Hiperparámetros

| Parámetro | Valor | Nota |
|---|---|---|
| `d_model` | 256 | Dimensión interna común a todas las ramas |
| `n_heads` | 8 | 32 dimensiones por cabeza |
| `n_fusion_layers` | 1 | Una capa en cada etapa |
| `dim_feedforward` | 512 | 2× `d_model` |
| `dropout` | 0.1 | |
| `norm_first` | `True` | Pre-normalización (§4) |
| `fusion_activation` | `gelu` | Coincide con DistilBERT; el defecto de PyTorch es ReLU |
| Longitud máxima | 512 sub-palabras | Límite posicional de DistilBERT |
| `backbone_lr` / `head_lr` | 2e-5 / 1e-4 | §13 |
| `warmup_ratio` | 0.1 | Calentamiento lineal, luego decaimiento a cero |
| `max_grad_norm` | 1.0 | Recorte de gradiente |
| Épocas / lote | 3 / 32 | |
| Pérdida | Entropía cruzada sin pesos | Decisión justificada con evidencia; ver §16 |
| Selección del punto de control | Área ROC de validación | Ver §16 |

### 16. Dos decisiones de entrenamiento que conviene poder defender

**Por qué no se pondera por clase.** El indicador de R1.4 exige decidirlo con evidencia y
no a priori. La clase minoritaria representa entre el 36.3% y el 40.8% de cada pliegue
—desbalance leve— y su exhaustividad no se desploma en ninguno: 0.9896, 0.9114 y 0.7484.
No se observa el colapso hacia la clase mayoritaria que justificaría intervenir. Tanto la
ponderación como la pérdida focal están implementadas y disponibles: la decisión es de
evidencia, no de capacidad.

**Por qué el criterio de selección es el área ROC y no la pérdida.** En las nueve corridas,
la pérdida de validación deja de bajar tras la primera o segunda época mientras la de
entrenamiento sigue descendiendo —firma clásica del sobreajuste—. Pero el área bajo la
curva de validación **sigue mejorando** en ese mismo tramo. Ambas cosas son compatibles: la
entropía cruzada penaliza la confianza mal calibrada, de modo que un modelo que ordena
mejor los ejemplos puede empeorar en pérdida si además se vuelve más confiado. Lo que se
degrada es la calibración, no la discriminación. Seleccionar por pérdida descartaría un
modelo que ordena mejor.

---

## Referencias

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł.,
  & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information
  Processing Systems*, 30, 5998–6008.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of
  BERT: smaller, faster, cheaper and lighter. *arXiv:1910.01108*.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning
  models for tabular data. *Advances in Neural Information Processing Systems*, 34,
  18932–18943.
- Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., … Simonyan, K.
  (2022). Flamingo: a visual language model for few-shot learning. *Advances in Neural
  Information Processing Systems*, 35, 23716–23736.
- Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., … Liu, T.-Y. (2020). On
  layer normalization in the Transformer architecture. *Proceedings of the 37th
  International Conference on Machine Learning*, 10524–10533.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization.
  *International Conference on Learning Representations*.
