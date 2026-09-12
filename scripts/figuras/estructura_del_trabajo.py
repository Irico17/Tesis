"""
Estructura de descomposición del trabajo, en un solo sitio.

La lámina de la EDT y la tabla de paquetes del Anexo A decían cosas distintas: la
tabla situaba las líneas base en el paquete de la Fase II mientras el apartado de
viabilidad temporal las asigna a la Fase I, y omitía por completo la tarea de
optimización que sustenta R2.3. Mantener dos copias de la misma estructura
garantiza que vuelvan a divergir, de modo que ambas se construyen desde aquí.

Este módulo no importa nada: lo consume tanto el guion de figuras, que necesita
matplotlib, como el de corrección del documento, que no debe necesitarlo.
"""

from __future__ import annotations

RAIZ = "Detección multimodal de phishing en correo electrónico con explicabilidad"

FASES = {"I": "Fase I, de marzo a julio de 2026",
         "II": "Fase II, de julio a octubre de 2026"}

# (título del paquete, tareas, entrega, fase)
PAQUETES = [
    ("1. Gestión y planificación",
     ["1.1 Formulación de objetivos, esquema canónico e indicadores",
      "1.2 Redacción del plan de proyecto",
      "1.3 Asesorías y control de cambios metodológicos"],
     "Plan de proyecto validado (Anexo A y Capítulo 1)", "I"),
    ("2. Fundamentación científica",
     ["2.1 Revisión sistemática en Scopus, ACM e IEEE con criterios PICOC",
      "2.2 Marco conceptual y normativo",
      "2.3 Conclusiones de la revisión de literatura"],
     "Capítulos 2 y 3", "I"),
    ("3. Ingeniería de datos y líneas base",
     ["3.1 Descarga e ingesta de ocho colecciones públicas",
      "3.2 Limpieza, esquema canónico y deduplicación",
      "3.3 Extracción de estructura y de metadatos de red",
      "3.4 Partición agrupada por campaña y prueba de humo",
      "3.5 Implementación de las líneas base unimodales y clásicas"],
     "R1.1, R1.2 y R2.1", "I"),
    ("4. Modelado predictivo",
     ["4.1 Arquitectura multimodal con atención cruzada",
      "4.2 Entrenamiento automatizado y ponderación de clases",
      "4.3 Evaluación comparativa y pruebas de significancia",
      "4.4 Optimización y exportación para el despliegue"],
     "R1.3, R1.4, R2.2 y R2.3", "II"),
    ("5. Explicabilidad",
     ["5.1 Selección del esquema interpretativo",
      "5.2 Integración del módulo de explicabilidad",
      "5.3 Validación de fidelidad de las explicaciones"],
     "R3.1, R3.2 y R3.3", "II"),
    ("6. Cierre documental",
     ["6.1 Redacción del capítulo de resultados",
      "6.2 Conclusiones, limitaciones y trabajo futuro",
      "6.3 Integración de la entrega y sustentación"],
     "Capítulos 4 y 5", "II"),
]

# Nombre canónico de cada resultado comprometido. Se emplea allí donde la tabla
# de paquetes los enumera, para que coincida con la matriz de objetivos y con el
# capítulo de resultados.
NOMBRE_DEL_RESULTADO = {
    "R1.1": "Corpus multimodal consolidado",
    "R1.2": "Pipeline de procesamiento estandarizado",
    "R1.3": "Arquitectura funcional de atención cruzada",
    "R1.4": "Pipeline de entrenamiento automatizado",
    "R2.1": "Modelos de referencia",
    "R2.2": "Cuadro comparativo de desempeño",
    "R2.3": "Modelo final optimizado",
    "R3.1": "Selección del esquema técnico de interpretabilidad",
    "R3.2": "Módulo de explicabilidad integrado",
    "R3.3": "Reporte de validación de transparencia operativa",
}


def entrega_detallada(resumen: str) -> str:
    """Convierte «R1.1, R1.2 y R2.1» en la enumeración con sus nombres."""
    codigos = [c for c in NOMBRE_DEL_RESULTADO if c in resumen]
    if not codigos:
        return resumen
    partes = [f"{c}: {NOMBRE_DEL_RESULTADO[c]}" for c in sorted(codigos)]
    if len(partes) == 1:
        return partes[0] + "."
    return "; ".join(partes[:-1]) + "; y " + partes[-1] + "."
