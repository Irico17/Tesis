# Experimentos E0–E9

Diez preguntas, cada una con su protocolo, sus líneas base y sus artefactos. E0
comprueba el corpus y el pipeline. El orden de los siguientes importa: **E1 a E3
comparten partición fija** y responden si el mecanismo funciona; E4 pregunta si
generaliza; E5 no mide rendimiento, sino cuánto de lo medido es procedencia y no
fenómeno. E6 mide el modelo optimizado, E7 contrasta el codificador, y E8 y E9
cubren la explicabilidad. Esta página detalla de E1 a E5; los demás se describen
en la cabecera de su propio guion.

Separarlos así responde a la indicación del asesor: *«hay que separar los
experimentos de forma sistemática, porque si combinamos todo no vamos a saber qué
ha hecho efecto»*.

```bash
python scripts/experimentos/e0_corpus_y_pipeline.py
python scripts/experimentos/e1_multimodalidad.py
python scripts/experimentos/e2_mecanismo_fusion.py
python scripts/experimentos/e3_ausencia_ramas.py
python scripts/experimentos/e4_generalizacion.py
python scripts/experimentos/e5_validez.py
python scripts/experimentos/e6_modelo_optimizado.py
python scripts/experimentos/e7_codificador_multilingue.py
python scripts/experimentos/e8_explicabilidad.py --muestra 60
python scripts/experimentos/e9_narrativa_y_validacion.py --verificar 100
```

Cada uno escribe su JSON y su figura en `medios_de_verificacion/experimentos/<eN>/`
**en la misma ejecución**. No se generan después con guiones aparte: así fue como
una tabla del capítulo llegó a declarar seis mediciones cuando ya había nueve.

---

## E1 · ¿La multimodalidad aporta?

| | |
|---|---|
| **Datos** | Subconjunto con las tres modalidades, prevalencia ≈ 0.49 |
| **Partición** | Agrupada por campaña, **fija y compartida** por los cuatro modelos |
| **Modelos** | Multimodal · solo texto · solo estructura · solo red |
| **Sin** | Centinela ni descarte de modalidad |
| **Contraste** | McNemar pareado: multimodal frente al mejor unimodal |

**Concluye:** hay sinergia si el multimodal supera al mejor unimodal más allá de la
dispersión entre semillas *y* McNemar es significativo.

⚠️ El legítimo trimodal procede casi entero de una comunidad. La cifra absoluta
**no** es «rendimiento de detección de phishing»; la comparación sí es válida,
porque los cuatro modelos ven exactamente los mismos datos.

## E2 · ¿La atención cruzada aporta sobre una fusión simple?

Mismos datos y **las mismas particiones** que E1. Variantes: atención cruzada a
nivel de token, a nivel de modalidad, **fusor MLP** y concatenación tardía.

El fusor MLP es la línea base que pidió el asesor: *«puede ser el fusor que tú
propones, que es cross attention, pero puede ser algo más simple, tipo MLP»*.

⚠️ **E2 depende de E1.** Si E1 concluye que la multimodalidad no aporta, E2 compara
cuatro maneras de no aportar nada. Sigue valiendo —demuestra que el resultado nulo
no es un fallo de implementación de la atención— pero su lectura cambia y el
capítulo debe decirlo.

## E3 · ¿Tolera la ausencia de ramas?

Cuatro maneras de tratar la ausencia, decididas por medición y no por argumento:

| Variante | Qué prueba |
|---|---|
| Centinela + descarte aleatorio | La actual |
| Sin centinela, solo filas completas | La que pidió el asesor |
| Compuerta condicionada | Enrutamiento por ejemplo, ya implementado |
| **Mezcla de expertos** | Enrutamiento por patrón de disponibilidad |

Además, **curva de degradación**: se ocultan 1 o 2 ramas en inferencia a correos
que sí las tienen, y se mide la caída.

Aquí el centinela se gana o se pierde su sitio. Conviene recordar que **no es una
propuesta arquitectónica sino una salvaguarda numérica**: impide que el softmax
opere sobre un conjunto vacío de claves y devuelva `NaN`.

## E4 · ¿Generaliza a correo no visto?

Sobre el **corpus completo**, no solo el trimodal.

- Métricas con partición agrupada **y sin agrupar**, para cuantificar cuánto infla
  no agrupar —que es lo que hace la mayor parte de la literatura del área—
- Estratificado **por tamaño de conglomerado**: ¿detecta campañas vistas una sola
  vez tan bien como las masivas?
- **Curva de aprendizaje** al 25/50/75/100%: ¿limitado por datos o ya saturado?
- Varias semillas de partición → estabilidad de la estimación

Se declara, con la medición que lo demuestra, que los tres ejes de cambio de
distribución —tiempo, remitente, colección— no son medibles con corpus público.
Las cifras quedan en la clave `ejes_no_medibles` del informe de E4.

## E5 · Validez (sección, no experimento)

Califica todo lo anterior. Auditoría de características, control de procedencia y
pliegues por colección **como diagnóstico**. Ocupa una fracción del espacio de
E1–E4 y no sostiene ninguna afirmación de rendimiento.
