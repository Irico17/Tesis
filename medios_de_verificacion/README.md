# Medios de verificación y evidencia de los indicadores

Cristhofer Alegre · 20210577 · PUCP · 2026-08-28

## Dónde está la evidencia del entregable

> ### → [`corpus_real/`](corpus_real/README.md)

Esa carpeta contiene **toda** la evidencia del Capítulo 4 del entregable vigente
(`Entregables/20210577_CristhoferAlegre_EdwinVillanueva_E3_FINAL.docx`), organizada por
indicador: R1.1, R1.2, R1.3, R1.4, R2.1 y R2.2. Su índice enuncia, para cada uno, el
criterio comprometido y el artefacto que lo sustenta.

El corpus del trabajo es `Dataset_Real.parquet`: 35,723 correos de seis colecciones
públicas de correo electrónico, cinco de ellas ingeridas en su formato de mensaje original.

Para comprobar que las cifras del documento siguen correspondiendo a los artefactos:

```bash
python scripts/entregable/verificar_cifras.py
```

Recorre cada tabla del capítulo, recalcula sus valores desde los JSON y falla con detalle
si alguno dejó de coincidir.

---

## Las carpetas `R1.*` y `R2.*`

Contienen artefactos de una construcción de corpus anterior, sustituida durante el
desarrollo. **No sustentan ninguna afirmación del entregable vigente** y se conservan
únicamente como registro del trabajo: sus cifras no coinciden con las del documento y no
deben citarse.

Si lo que se busca es la evidencia de un indicador, está en `corpus_real/`.

---

## Relación con `data/reports/`

Los guiones escriben sus salidas bajo `data/reports/` durante la ejecución. La versión
curada y versionada de lo que sustenta el documento es la de `corpus_real/`:

| Ruta que produce la ejecución | Ubicación versionada |
|---|---|
| `data/reports/corpus_real_report.json` | `corpus_real/` |
| `data/reports/baselines_real.json` | `corpus_real/` |
| `data/reports/loso_real_slim/` | `corpus_real/loso/` |
| `data/reports/training_logs/` | `corpus_real/historiales/` |
| `data/reports/iov_mv/` | `corpus_real/iov_mv/` |
| `data/reports/model_sanity_check.json` | se cita en su ruta de origen (R1.3) |
| `data/reports/auditoria_caracteristicas_real.json` | se cita en su ruta de origen (R2.2) |

Las salidas de trabajo no se versionan por duplicado: su contenido vigente es exactamente
el de `corpus_real/`.
