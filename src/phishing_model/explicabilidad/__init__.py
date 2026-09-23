"""Explicabilidad (R3): atribución, narrativa y validación sobre el modelo reportado.

Paquete NUEVO, deliberadamente separado de `phishing_model.xai`. Aquel se escribió
antes de la reconstrucción del corpus y ninguno de sus resultados llegó al
documento; se conserva intacto porque tres de sus piezas siguen siendo correctas y
se reutilizan desde aquí (`shap_explainer`, `lime_explainer`, `faithfulness` y
`narrative`). Lo que este paquete añade es lo que allí falta o está roto:

  - `atribucion_modal`: el reparto de atención entre las modalidades, corregido.
    `xai.attention_explainer` declara 18 nombres de característica y el modelo
    entrega 19 posiciones, porque `model._con_centinela` antepone el token de
    "sin modalidad" en la posición 0. Su comprobación de longitud aborta, de modo
    que el defecto nunca produjo atribuciones desalineadas en silencio, pero
    tampoco produjo ninguna. Aquí la anchura se resuelve en tiempo de ejecución.

  - `procedencia_del_modelo`: la negativa a explicar un punto de control que no es
    el que la tesis reporta. En el equipo local sobrevivía un punto de control de
    prueba de humo, de ocho pasos de optimización y F1 de 0.65 frente al 0.9927
    informado; explicar ese modelo habría producido un capítulo entero de
    afirmaciones sobre un objeto equivocado, sin que nada avisara.
"""
