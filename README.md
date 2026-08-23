# Detección multimodal de phishing con IA explicable

Código y evidencia de la tesis **«Diseño y desarrollo de un modelo multimodal para la
detección de phishing integrado con IA Explicable (XAI)»**.

Cristhofer Arhon Alegre Cotrina · 20210577 · Pontificia Universidad Católica del Perú
Asesor: Dr. Edwin Rafael Villanueva Talavera

---

## Qué hay aquí

Un modelo multimodal que combina el contenido textual del correo, la estructura de su
HTML y los metadatos de red mediante atención cruzada, junto con el pipeline de datos,
los modelos de referencia contra los que se compara y la evidencia que sustenta cada
cifra del capítulo de resultados.

El repositorio está organizado para que **cualquier afirmación cuantitativa del documento
pueda comprobarse** sin volver a entrenar: las métricas, los registros de ejecución y las
predicciones necesarias están versionados en `medios_de_verificacion/`.

---

## Estructura

```
.
├── src/                          Código fuente (paquete instalable)
│   ├── phishing_pipeline/        Fase A · datos
│   │   ├── downloaders/          Descarga de las tres fuentes públicas
│   │   ├── cleaners/             Limpieza por fuente al esquema canónico
│   │   ├── dedup/                Deduplicación exacta y de casi-duplicados
│   │   ├── features/             Extracción estructural (DOM) y de red (URL, SPF)
│   │   ├── metrics/              Informes y figuras del corpus
│   │   ├── unifier.py            Unificación de fuentes
│   │   ├── splits.py             Particiones agrupadas por plantilla
│   │   └── pipeline.py           Orquestador con etapas seleccionables
│   │
│   ├── phishing_baseline/        Fase B · modelos de referencia
│   │   ├── b1_tfidf_lr.py        Texto clásico
│   │   ├── b2_distilbert.py      Texto profundo
│   │   ├── b3_rf_structural.py   Estructura HTML/DOM
│   │   ├── b4_network.py         Metadatos de red
│   │   ├── b5_classical_multimodal.py   Fusión temprana clásica
│   │   ├── group_cv.py           Validación por fuente no observada
│   │   └── evaluation.py         Métricas comunes a todos los modelos
│   │
│   └── phishing_model/           Fase C · modelo propuesto
│       ├── config.py             Configuración de arquitectura y entrenamiento
│       ├── dataset.py            Dataset multimodal y escalado acotado
│       ├── encoders/             Codificador de texto y tokenizadores tabulares
│       ├── fusion/               Etapa 1 (auto-atención) y etapa 2 (atención cruzada)
│       ├── model.py              Ensamblado de las cuatro variantes de fusión
│       ├── losses.py             Entropía cruzada ponderada y pérdida focal
│       ├── train.py              Bucle de entrenamiento
│       ├── train_loso.py         Validación por fuente no observada
│       ├── evaluate.py           Inferencia, umbrales y calibración
│       ├── sanity_check.py       Siete comprobaciones de integridad
│       ├── quantization.py       Exportación y cuantización
│       └── xai/                  Explicabilidad (atención, LIME, SHAP)
│
├── scripts/                      Análisis y verificación
│   └── entregable/               Armado del documento de tesis (docx)
├── notebooks/                    Cuadernos que documentan el flujo de datos
├── doc/                          Documentación que perdura
│   ├── plan_proyecto/            Figuras del Anexo A (EDT, Gantt, riesgos)
│   └── validacion_xai/           Instrumentos de evaluación con usuarios
├── medios_de_verificacion/       Evidencia de los IOV y MV (ver abajo)
└── data/                         Datos y reportes (mayormente ignorados)
```

### Qué NO está versionado, y por qué

| Ruta | Motivo |
|---|---|
| `data/Datasets_Originales/`, `Dataset_Unificado.parquet` | Datos crudos y corpus consolidado; regenerables con el pipeline |
| `data/model/` | Puntos de control y modelos exportados; binarios grandes |
| `data/predictions/` | Predicciones completas; en `medios_de_verificacion/` va el subconjunto que reproduce la prueba de significancia |
| `Entregables/` | Documentos Word y paquetes comprimidos; no son código |
| `scratch/` | Espacio temporal desechable |
| `.venv/` | Entorno virtual |

---

## Medios de verificación

`medios_de_verificacion/` contiene **los datos finales empleados para producir los
resultados y los medios de verificación que exige la matriz de objetivos**. Está
organizada por resultado, y cada apartado de su `README.md` enuncia el indicador (IOV) y
el medio (MV) declarados, seguidos del artefacto que los sustenta.

```
medios_de_verificacion/
├── README.md                     Índice IOV → MV → artefacto
├── R1.1_dataset/                 Cobertura modal, informe del corpus, prueba de humo
├── R1.2_pipeline/                Análisis por columna, agrupación de casi-duplicados
├── R1.3_arquitectura/            Siete comprobaciones funcionales, diagrama
├── R1.4_entrenamiento/
│   ├── historiales/              36 registros de ejecución con métricas por época
│   └── curvas_de_aprendizaje/    45 curvas de pérdida y exactitud
├── R2.1_modelos_referencia/      Resultados de B1 a B5, curvas ROC, matrices de confusión
└── R2.2_comparacion/
    ├── *_loso.json               15 corridas por fuente no observada
    ├── significancia_mcnemar.json
    ├── roc_auc_por_configuracion.json
    ├── matriz_confusion_*.png
    └── predicciones/             Predicciones pareadas del modelo propuesto, B1 y B2
```

**Criterio de actualización.** Cuando una corrida sustituya a otra, aquí van **los datos
vigentes**, no el histórico: la carpeta debe reflejar en todo momento las cifras que el
capítulo de resultados afirma. Al añadir o rehacer un resultado, actualizar también su
apartado en el `README.md` de la carpeta.

---

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.lock.txt
pip install -e .
```

`requirements.lock.txt` fija las versiones exactas, incluidas las transitivas.
`requirements.txt` declara solo las cotas mínimas.

## Reproducción

```bash
# Fase A · construir el corpus desde las fuentes públicas
python -m phishing_pipeline.pipeline --all

# Fase B · modelos de referencia
python -m phishing_baseline.run_baselines --b2-epochs 3 --b2-batch-size 32

# Fase C · integridad de la arquitectura (siete comprobaciones)
python -m phishing_model.sanity_check

# Fase C · una corrida por fuente no observada
python -m phishing_model.train_loso --fusion-type cross_attention_token_level \
    --seed 42 --scaler-kind quantile --modality-dropout-mode randomize \
    --epochs 3 --batch-size 32 --device cuda --no-resume

# Análisis de los resultados
python scripts/agregar_semillas.py     # agregado con desviación entre semillas
python scripts/veredicto_final.py      # comparación pareada y contraste de hipótesis
python scripts/generar_iov_mv.py       # significancia, ROC-AUC y matrices de confusión
```

El detalle de cada etapa del pipeline está en [doc/COMANDOS_PIPELINE.md](doc/COMANDOS_PIPELINE.md)
y [doc/PIPELINE_DATOS.md](doc/PIPELINE_DATOS.md).

---

## Documentación

| Documento | Contenido |
|---|---|
| [PREREGISTRO_HIPOTESIS](doc/PREREGISTRO_HIPOTESIS.md) | Hipótesis y umbrales congelados **antes** de entrenar, con criterio de falsación |
| [RESULTADOS_LOSO_TRES_SEMILLAS](doc/RESULTADOS_LOSO_TRES_SEMILLAS.md) | Resultados de las 12 corridas y contraste de la hipótesis |
| [ANALISIS_MODELO_FINAL](doc/ANALISIS_MODELO_FINAL.md) | Defectos verificados del modelo y decisiones de arquitectura |
| [REVISION_METODOLOGICA](doc/REVISION_METODOLOGICA.md) | Revisión del diseño experimental, métricas y validez de constructo |
| [GUIA_SUSTENTACION_CAPITULO4](doc/GUIA_SUSTENTACION_CAPITULO4.md) | Recorrido técnico del capítulo de resultados |
| [COMANDOS_PIPELINE](doc/COMANDOS_PIPELINE.md) · [PIPELINE_DATOS](doc/PIPELINE_DATOS.md) | Operación del pipeline de datos |
| [validacion_xai/](doc/validacion_xai/) | Instrumentos de evaluación con usuarios (R3.3) |

---

## Notas metodológicas

Tres condiciones gobiernan el trabajo experimental y conviene tenerlas presentes al leer
cualquier resultado:

1. **El protocolo primario es la validación por fuente no observada.** Una partición
   aleatoria no discrimina entre arquitecturas en este corpus: los cinco modelos de
   referencia obtienen F1 entre 0.8490 y 0.9986 y la comparación entre el multimodal
   clásico y el textual clásico no alcanza significancia.

2. **Las hipótesis están pre-registradas.** El umbral y el criterio de falsación se
   fijaron y fecharon antes de entrenar el modelo propuesto. La métrica primaria no puede
   cambiarse a posteriori; las métricas adicionales se reportan como secundarias
   declaradas.

3. **Las comparaciones exigen varianza entre semillas.** Cada configuración se ejecuta
   con tres semillas, porque las capas de fusión se inicializan al azar y sin esa
   dispersión no puede saberse si una diferencia entre arquitecturas es real.
