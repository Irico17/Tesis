"""MultimodalPhishingClassifier — ensambla encoders + tokenizers + fusión (Stage 1+2) + cabeza.

Orquesta las 4 variantes de `FusionType` (switch de ablación de R2.2) bajo una
única interfaz `forward(batch) -> logits [batch, 2]`, y aplica el enmascaramiento
de modalidad durante entrenamiento (mecanismo central de manejo de modalidades
ausentes, ver plan Fase B / docstring de `_apply_modality_dropout`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from phishing_model.config import ModelConfig, FusionType, NETWORK_CATEGORICAL_VOCABS
from phishing_model.encoders.network_tokenizer import NetworkTokenizer
from phishing_model.encoders.structural_tokenizer import StructuralTokenizer
from phishing_model.encoders.text_encoder import TextEncoder
from phishing_model.fusion.cross_attention import CrossAttentionFusion, masked_mean_pool
from phishing_model.fusion.mixture_of_experts import MixtureOfModalityExperts
from phishing_model.fusion.modality_encoder import ModalityEncoder
from phishing_model.losses import CabezaAdversariaDeFuente


def sin_decaimiento(nombre: str) -> bool:
    """
    Indica si un parámetro debe quedar EXENTO del decaimiento de peso.

    Se excluyen los sesgos y los parámetros de normalización por capa. El
    decaimiento penaliza la norma del parámetro para limitar la capacidad efectiva
    del modelo, razonamiento que aplica a las matrices de pesos pero no a estos: la
    escala y el desplazamiento de una capa de normalización son parámetros de
    calibración cuyo valor de reposo es 1 y 0 respectivamente, de modo que empujarlos
    hacia cero altera la normalización en lugar de regularizarla. Es la práctica
    estándar en el ajuste fino de modelos tipo BERT desde su formulación original, y
    la que aplica por defecto el entrenador de la biblioteca `transformers`.

    Se comprobó sobre esta arquitectura que la omisión afectaba a 84 de los 147
    tensores entrenables, incluidas todas las normalizaciones del codificador
    preentrenado.
    """
    # El nombre llega relativo al submódulo que lo consulta, de modo que un sesgo
    # puede presentarse como "bias" a secas y no solo como "…​.bias".
    n = nombre.lower()
    return (
        n == "bias"
        or n.endswith(".bias")
        or "layernorm" in n
        or "layer_norm" in n
        or n.endswith(".norm.weight")
        or n.endswith("norm.weight")
    )


# Búferes cuya inicialización no depende del entrenamiento previo y que, por
# tanto, pueden reconstruirse si faltan en un punto de control anterior a su
# introducción. Cualquier OTRA clave faltante o sobrante sigue siendo un error:
# significaría que el punto de control no corresponde a esta arquitectura.
BUFERES_RECONSTRUIBLES = frozenset({"_gate_media"})


def cargar_pesos(modelo: nn.Module, state_dict: dict, origen: str = "punto de control") -> list[str]:
    """
    Carga pesos tolerando la ausencia de búferes añadidos después de guardarlos.

    Existe porque la arquitectura gana búferes con el tiempo y una carga estricta
    convierte esa evolución en un fallo por clave faltante, aunque el búfer en
    cuestión se reconstruya idéntico con su valor inicial. La alternativa perezosa
    —`strict=False` a secas— es peor: silenciaría también un punto de control que
    de verdad no corresponde a esta arquitectura, que es justo el error que la
    comprobación debe atrapar.

    Devuelve la lista de búferes reconstruidos, para que el llamador pueda
    registrarla.
    """
    informe = modelo.load_state_dict(state_dict, strict=False)
    inesperadas = set(informe.missing_keys) - BUFERES_RECONSTRUIBLES
    if inesperadas or informe.unexpected_keys:
        raise RuntimeError(
            f"El {origen} no corresponde a esta arquitectura. "
            f"Faltantes: {sorted(inesperadas)}. Sobrantes: {sorted(informe.unexpected_keys)}."
        )
    return sorted(informe.missing_keys)


class MultimodalPhishingClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.text_encoder = TextEncoder(
            config.text_model_name,
            d,
            freeze=config.freeze_text_encoder,
            grad_checkpointing=config.grad_checkpointing,
        )

        needs_modality_branches = config.fusion_type != FusionType.TEXT_ONLY
        if needs_modality_branches:
            self.structural_tokenizer = StructuralTokenizer(
                config.n_structural_features, d, dropout=config.dropout
            )
            self.network_tokenizer = NetworkTokenizer(
                config.n_network_continuous, NETWORK_CATEGORICAL_VOCABS, d
            )
            # Token centinela de "sin modalidad". Se antepone SIEMPRE a la memoria
            # de atención y nunca se enmascara, de modo que ninguna fila puede
            # quedar con el conjunto de claves vacío.
            #
            # Sustituye a la salvaguarda anterior, que ante una fila sin ninguna
            # modalidad desenmascaraba la posición 0 de la memoria. Esa posición
            # es, en la variante a nivel de token, el token de la primera
            # característica estructural: un vector con contenido aprendido —norma
            # L2 de 16.17 medida incluso con entrada nula—, no un relleno neutro.
            # El modelo atendía así a un token de contenido para filas que, por
            # definición, no tienen contenido que mostrar.
            #
            # La fuga no transportaba información en el corpus actual, porque la
            # primera característica estructural es `has_html` y la disponibilidad
            # estructural se define precisamente como `has_html == 1`, de modo que
            # su valor era constante en las filas afectadas. Pero la propiedad
            # dependía de ese detalle: reordenar las columnas estructurales o
            # cambiar la definición de disponibilidad la convertía en una fuga real
            # y silenciosa. Con el centinela, el aislamiento se sostiene por
            # construcción y no por coincidencia.
            self.no_modality_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        self.uses_stage1 = config.fusion_type == FusionType.CROSS_ATTENTION_TOKEN_LEVEL
        self.uses_cross_attention = config.fusion_type in (
            FusionType.CROSS_ATTENTION_TOKEN_LEVEL,
            FusionType.CROSS_ATTENTION_MODALITY_LEVEL,
        )
        if self.uses_stage1:
            self.modality_encoder = ModalityEncoder(
                d,
                config.n_heads,
                config.dim_feedforward,
                config.dropout,
                n_layers=config.n_fusion_layers,
                norm_first=config.norm_first,
                activation=config.fusion_activation,
            )
        if self.uses_cross_attention:
            self.cross_attention = CrossAttentionFusion(
                d,
                config.n_heads,
                config.dim_feedforward,
                config.dropout,
                n_layers=config.n_fusion_layers,
                norm_first=config.norm_first,
                activation=config.fusion_activation,
            )
            # Compuerta de contribución modal, inspirada en el gating de Alayrac
            # et al. (2022, "Flamingo"). La fusión se expresa como interpolación
            # entre la representación puramente textual y la fusionada:
            #
            #     fusionada = texto + tanh(g) · (atención_cruzada(texto, memoria) − texto)
            #
            # El valor de tanh(g) es una medida directa y comparable entre pliegues
            # de cuánto emplea el modelo las modalidades no textuales: evidencia
            # cuantitativa sobre la pregunta central de la investigación, en lugar
            # de una inferencia indirecta a partir de métricas agregadas.
            #
            # Sobre la inicialización. Flamingo inicializa la compuerta en cero, de
            # modo que el modelo arranca siendo exactamente el de solo texto. Aquí
            # esa elección NO funciona, y se comprobó antes de adoptarla: el
            # gradiente que llega a la atención cruzada es proporcional a tanh(g),
            # luego con g = 0 las capas de fusión reciben gradiente exactamente nulo
            # en el primer paso. La compuerta sí recibe gradiente (0.0184 medido),
            # pero solo puede evaluar una atención que sigue en su inicialización
            # aleatoria y que, por serlo, no aporta señal: tras 30 pasos la compuerta
            # se había movido a −0.000276, es decir, se cerraba en lugar de abrirse.
            # Flamingo tolera ese arranque frío porque entrena con órdenes de magnitud
            # más de pasos; aquí el presupuesto es de tres épocas.
            #
            # Se inicializa por tanto en un valor intermedio: la atención recibe
            # gradiente desde el primer paso y la compuerta queda libre de crecer o
            # decrecer. Partir del punto medio y no de un extremo hace además que el
            # valor final sea informativo, porque no está sesgado por el arranque.
            self.modality_gate = nn.Parameter(
                torch.atanh(torch.tensor([float(config.modality_gate_init)]))
            )

            # Compuerta CONDICIONADA a la entrada (`modality_gate_mode="conditional"`).
            #
            # La compuerta escalar de arriba es única para todo el corpus, de modo
            # que obliga al modelo a emplear las modalidades no textuales en la
            # misma proporción para todos los correos. Se verificó que la utilidad
            # de esas modalidades NO es homogénea: la información mutua condicional
            # entre las características tabulares y la etiqueta, medida dentro de
            # cada fuente, va de 0.0546 nats en Kaggle —donde no hay prácticamente
            # nada que aprovechar— a 0.6759 en Spam_Genuine_Mail. Un escalar único
            # solo puede alcanzar un compromiso entre regímenes incompatibles, y el
            # valor de 0.497 al que converge es exactamente eso.
            #
            # La variante condicionada calcula un valor por correo a partir de la
            # representación textual agregada y de las banderas de disponibilidad:
            #
            #     g_i = sigmoide(W · [texto_agregado_i ; hay_estructura_i ; hay_red_i] + b)
            #
            # Conserva la interpretabilidad y la amplía: en lugar de un escalar se
            # puede reportar la DISTRIBUCIÓN de la compuerta por fuente y por clase,
            # que responde a la pregunta de investigación con mucho más detalle que
            # un único número. El sesgo se inicializa en el logit de
            # `modality_gate_init` y los pesos en una escala pequeña, de modo que el
            # valor inicial medio coincide con el de la variante global y ambas
            # arrancan del mismo punto.
            self.modality_gate_mode = config.modality_gate_mode
            if self.modality_gate_mode == "conditional":
                self.modality_gate_proj = nn.Linear(d + 2, 1)
                nn.init.normal_(self.modality_gate_proj.weight, std=0.02)
                init = float(config.modality_gate_init)
                init = min(max(init, 1e-4), 1 - 1e-4)
                nn.init.constant_(
                    self.modality_gate_proj.bias, float(torch.logit(torch.tensor(init)))
                )
            # Media móvil del valor de la compuerta, para poder reportarla sin
            # necesidad de volver a pasar datos por el modelo. Se actualiza solo en
            # entrenamiento y acompaña al punto de control.
            self.register_buffer("_gate_media", torch.tensor(float(config.modality_gate_init)))

        self.usa_moe = config.fusion_type == FusionType.MIXTURE_OF_EXPERTS
        if self.usa_moe:
            self.moe = MixtureOfModalityExperts(
                d, n_expertos=config.n_expertos_moe,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout, activation=config.fusion_activation,
            )

        if config.fusion_type == FusionType.TEXT_ONLY:
            head_in = d
        elif config.fusion_type == FusionType.MIXTURE_OF_EXPERTS:
            # La mezcla devuelve una representación de la anchura del modelo.
            head_in = d
        elif config.fusion_type == FusionType.CONCAT_LATE_FUSION:
            head_in = d * 3  # texto + estructura + red, cada uno pooled independientemente
            # Representación aprendida de la ausencia, en lugar del vector nulo que
            # devuelve el promediado enmascarado. Un vector de ceros es
            # indistinguible de una rama cuyas características son legítimamente
            # nulas; un vector aprendido permite al modelo separar ambos casos.
            self.missing_structure = nn.Parameter(torch.randn(1, d) * 0.02)
            self.missing_network = nn.Parameter(torch.randn(1, d) * 0.02)
        else:  # ambas variantes de cross-attention devuelven texto fusionado pooled
            head_in = d

        # Cabeza lineal por defecto; perceptrón multicapa si se pide anchura
        # oculta. La diferencia no es cosmética: la variante `fusor_mlp` de E2
        # existe precisamente para contrastar la atención cruzada contra una
        # fusión aprendida pero simple, y con cabeza lineal sería indistinguible
        # de la concatenación tardía.
        if config.cabeza_oculta:
            self.classifier = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Linear(head_in, config.cabeza_oculta),
                nn.GELU() if config.fusion_activation == "gelu" else nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.cabeza_oculta, 2),
            )
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(head_in),
                nn.Dropout(config.dropout),
                nn.Linear(head_in, 2),
            )

        # Cabeza adversaria de fuente (ver `losses.CabezaAdversariaDeFuente`).
        # Opera sobre la MISMA representación que alimenta al clasificador, que es
        # donde importa que la fuente no sea recuperable: eliminarla antes de la
        # cabeza no sirve de nada si la cabeza puede reconstruirla.
        self.cabeza_adversaria = None
        if config.n_fuentes_adversario and config.n_fuentes_adversario > 1:
            self.cabeza_adversaria = CabezaAdversariaDeFuente(
                head_in, config.n_fuentes_adversario
            )

        self.modality_dropout_prob = config.modality_dropout_prob
        self.modality_dropout_mode = config.modality_dropout_mode
        # Distribución marginal de los cuatro patrones de disponibilidad, medida sobre el
        # corpus consolidado: ninguna 13.18%, solo red 46.97%, solo estructura 0.35%,
        # ambas 39.51%. Se registra como buffer para acompañar al modelo entre
        # dispositivos y quedar guardada en el punto de control, de modo que una
        # reanudación emplee la misma distribución con la que se entrenó.
        self.register_buffer(
            "_pattern_probs",
            torch.tensor(config.modality_pattern_probs, dtype=torch.float),
            persistent=True,
        )

    # ------------------------------------------------------------------ máscaras

    def _apply_modality_dropout(
        self, has_structure: torch.Tensor, has_network: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Enmascaramiento de modalidad durante el entrenamiento, en dos variantes.

        **Descarte independiente** (`modality_dropout_mode="independent"`). Cada rama
        disponible se suprime con probabilidad fija e independiente. Es la formulación
        original de este trabajo.

        **Aleatorización del patrón** (`"randomize"`, recomendada). Se sortea para cada
        fila un patrón de disponibilidad extraído de la distribución marginal empírica
        del corpus, con independencia de su etiqueta.

        Por qué se añadió la segunda. Se verificó que la disponibilidad de modalidad
        actúa como identificador de la fuente antes que como señal del fenómeno: informa
        0.2893 nats sobre el origen del dato frente a 0.0790 sobre la etiqueta, y un
        clasificador que solo la observa alcanza F1 de 0.7049 dentro de la distribución
        de entrenamiento. En PhishMMF, poseer estructura HTML distingue la clase con una
        brecha de 93 puntos porcentuales, relación que se invierte en Kaggle.

        El descarte independiente apenas atenúa ese atajo, y la razón es estructural más
        que de magnitud: suprime ramas, pero deja intacta la asociación entre patrón y
        etiqueta en las filas que no suprime. Con probabilidad 0.15, el 85% de las
        observaciones conserva su patrón original y con él su correlación con la clase.
        Se comprobó empíricamente: elevar la probabilidad a 0.30, 0.45 y 0.60 mantiene el
        F1 del pliegue de Kaggle entre 0.0000 y 0.0009 —igual que con 0.15— y degrada
        además el área bajo la curva de 0.9072 a 0.8453, porque destruye señal útil sin
        romper la asociación. Descartar no equivale a aleatorizar.

        La aleatorización sí ataca la asociación en su origen, al sortear el patrón de
        una distribución que no depende de la etiqueta. Medido sobre el corpus real, la
        información mutua entre patrón y etiqueta desciende de 0.0790 a **0.0235 nats**
        (media de seis repeticiones, desviación 0.0008), frente a 0.0602 nats que deja el
        descarte independiente. A diferencia de elevar la probabilidad de descarte, no
        sacrifica el contenido: cada modalidad sigue presentándose en la proporción
        marginal correcta.

        Conviene precisar el alcance, porque el mecanismo NO anula la información mutua.
        El patrón sorteado no puede AÑADIR una modalidad que la fila no posee —no existe
        contenido que mostrar—, de modo que la máscara resultante se intersecta con la
        disponibilidad natural. Esa intersección deja pasar la parte de la dependencia
        que proviene de la disponibilidad real, y es la razón de que quede un residuo de
        0.0235 nats en lugar de cero. La reducción es del 70%, no la eliminación.

        En evaluación, ambos modos quedan desactivados: las máscaras naturales se
        devuelven sin modificar y nunca se alteran in situ.
        """
        if not self.training:
            return has_structure, has_network

        if self.modality_dropout_mode == "randomize":
            # La aleatorización no consume `modality_dropout_prob`: su intensidad la
            # fija la distribución marginal, no una probabilidad de supresión. Se
            # comprueba aquí de forma explícita para que fijar la probabilidad en
            # cero no desactive el mecanismo en silencio, como ocurría antes al
            # compartir ambos modos una misma condición de guarda.
            probs = self._pattern_probs.to(has_structure.device)
            # 0 = ninguna, 1 = solo red, 2 = solo estructura, 3 = ambas.
            sorteado = torch.multinomial(probs, has_structure.shape[0], replacement=True)
            quiere_estructura = (sorteado >= 2).to(has_structure.dtype)
            quiere_red = (sorteado % 2 == 1).to(has_network.dtype)
            # Intersección con lo realmente disponible: no se puede mostrar lo que no hay.
            return has_structure * quiere_estructura, has_network * quiere_red

        if self.modality_dropout_prob <= 0:
            return has_structure, has_network

        drop_structure = (torch.rand_like(has_structure) < self.modality_dropout_prob) & (has_structure > 0.5)
        drop_network = (torch.rand_like(has_network) < self.modality_dropout_prob) & (has_network > 0.5)

        effective_structure = has_structure.clone()
        effective_network = has_network.clone()
        effective_structure[drop_structure] = 0.0
        effective_network[drop_network] = 0.0
        return effective_structure, effective_network

    def _con_centinela(
        self, memory: torch.Tensor, memory_pad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Antepone el token de "sin modalidad", siempre visible, a la memoria de atención."""
        batch = memory.shape[0]
        centinela = self.no_modality_token.expand(batch, -1, -1).to(memory.dtype)
        visible = torch.zeros(batch, 1, dtype=torch.bool, device=memory_pad.device)
        return torch.cat([centinela, memory], dim=1), torch.cat([visible, memory_pad], dim=1)

    def _valor_compuerta(
        self,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        has_structure: torch.Tensor | None,
        has_network: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Valor de la compuerta modal, con forma [batch, 1, 1] para poder difundirse
        sobre la secuencia textual. En modo global es el mismo para todas las filas.
        """
        if self.modality_gate_mode != "conditional" or has_structure is None:
            return torch.tanh(self.modality_gate).view(1, 1, 1)

        pooled = masked_mean_pool(text_tokens, text_key_padding_mask)  # [batch, d]
        entrada = torch.cat(
            [pooled, has_structure.unsqueeze(-1).to(pooled.dtype), has_network.unsqueeze(-1).to(pooled.dtype)],
            dim=-1,
        )
        return torch.sigmoid(self.modality_gate_proj(entrada)).unsqueeze(1)  # [batch, 1, 1]

    def _fusionar(
        self,
        text_tokens: torch.Tensor,
        text_key_padding_mask: torch.Tensor,
        memory: torch.Tensor,
        memory_pad: torch.Tensor,
        has_structure: torch.Tensor | None = None,
        has_network: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Atención cruzada regulada por la compuerta de contribución modal."""
        salida = self.cross_attention(text_tokens, text_key_padding_mask, memory, memory_pad)
        g = self._valor_compuerta(text_tokens, text_key_padding_mask, has_structure, has_network)
        self._ultima_compuerta = g.detach().reshape(-1)
        with torch.no_grad():
            self._gate_media.fill_(float(self._ultima_compuerta.mean()))
        return text_tokens + g * (salida - text_tokens)

    @torch.no_grad()
    def contribucion_modal(self) -> float:
        """
        Valor actual de la compuerta, en [0, 1). Cuantifica qué fracción de la
        representación fusionada procede de las modalidades no textuales: 0 significa
        que el modelo se comporta exactamente como la configuración de solo texto.

        En modo condicionado el valor varía por correo, de modo que lo devuelto es la
        media del último lote procesado. Para el análisis por fuente y por clase que
        habilita ese modo, emplear `compuerta_por_ejemplo()` tras un paso hacia
        delante, que devuelve el vector completo sin promediar.
        """
        if not self.uses_cross_attention:
            return 0.0
        if self.modality_gate_mode == "conditional":
            return float(self._gate_media.item())
        return float(torch.tanh(self.modality_gate).item())

    def perdida_adversaria(self, fuentes: torch.Tensor, escala: float) -> torch.Tensor | None:
        """
        Pérdida de la cabeza adversaria sobre la representación del último paso.

        Devuelve None si la cabeza no está activa, de modo que el llamador pueda
        sumarla incondicionalmente comprobando solo el nulo.
        """
        if self.cabeza_adversaria is None or getattr(self, "_ultima_representacion", None) is None:
            return None
        return self.cabeza_adversaria(self._ultima_representacion, fuentes, escala)

    @torch.no_grad()
    def compuerta_por_ejemplo(self) -> torch.Tensor | None:
        """Valor de la compuerta para cada fila del último lote, o None si aún no hubo ninguno."""
        return getattr(self, "_ultima_compuerta", None)

    # ------------------------------------------------------------------ forward

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Devuelve logits [batch, 2] (sin softmax -- CrossEntropyLoss/FocalLoss lo aplican internamente)."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        text_key_padding_mask = attention_mask == 0  # True = token de padding a excluir

        text_tokens = self.text_encoder(input_ids, attention_mask)  # [batch, seq_len, d_model]

        if self.config.fusion_type == FusionType.TEXT_ONLY:
            pooled_text = masked_mean_pool(text_tokens, text_key_padding_mask)
            self._ultima_representacion = pooled_text
            return self.classifier(pooled_text)

        has_structure, has_network = self._apply_modality_dropout(
            batch["has_structure"], batch["has_network"]
        )

        structural_tokens = self.structural_tokenizer(batch["structural_continuous"])  # [batch, 6, d]
        network_tokens = self.network_tokenizer(
            batch["network_continuous"], batch["network_categorical"]
        )  # [batch, 12, d]

        # Máscara de padding por rama: True = excluir. Se replica el flag de la fila
        # (escalar) a lo largo de todos los tokens de esa rama -- si la rama está
        # ausente, TODOS sus tokens se excluyen en bloque, no individualmente.
        struct_pad = (has_structure < 0.5).unsqueeze(1).expand(-1, structural_tokens.shape[1])
        net_pad = (has_network < 0.5).unsqueeze(1).expand(-1, network_tokens.shape[1])

        if self.usa_moe:
            # La mezcla opera sobre las tres ramas ya agregadas, no sobre la
            # secuencia: su enrutamiento decide CÓMO combinar, no dónde atender.
            # Las ramas ausentes entran como vector nulo y la compuerta lo sabe
            # por las banderas explícitas, de modo que no hace falta centinela:
            # es justamente la alternativa que E3 mide.
            pooled_text = masked_mean_pool(text_tokens, text_key_padding_mask)
            pooled_struct = masked_mean_pool(structural_tokens, struct_pad)
            pooled_net = masked_mean_pool(network_tokens, net_pad)
            pooled_struct = pooled_struct * (has_structure >= 0.5).unsqueeze(-1)
            pooled_net = pooled_net * (has_network >= 0.5).unsqueeze(-1)
            fusionada = self.moe(
                pooled_text, pooled_struct, pooled_net, has_structure, has_network
            )
            self._ultima_representacion = fusionada
            return self.classifier(fusionada)

        if self.config.fusion_type == FusionType.CONCAT_LATE_FUSION:
            pooled_text = masked_mean_pool(text_tokens, text_key_padding_mask)
            pooled_struct = masked_mean_pool(structural_tokens, struct_pad)
            pooled_net = masked_mean_pool(network_tokens, net_pad)
            # La rama ausente se representa con su vector aprendido de ausencia.
            presente_struct = (has_structure >= 0.5).unsqueeze(-1)
            presente_net = (has_network >= 0.5).unsqueeze(-1)
            pooled_struct = torch.where(presente_struct, pooled_struct, self.missing_structure.to(pooled_struct.dtype))
            pooled_net = torch.where(presente_net, pooled_net, self.missing_network.to(pooled_net.dtype))
            combined = torch.cat([pooled_text, pooled_struct, pooled_net], dim=-1)
            self._ultima_representacion = combined
            return self.classifier(combined)

        if self.config.fusion_type == FusionType.CROSS_ATTENTION_MODALITY_LEVEL:
            # Variante ligera: cada rama se colapsa a 1 token (mean-pool) antes de la
            # cross-attention -- comparación directa contra la fusión a nivel de token
            # (ablación de R2.2: ¿la riqueza de tokens individuales realmente aporta?).
            struct_token = masked_mean_pool(structural_tokens, struct_pad).unsqueeze(1)
            net_token = masked_mean_pool(network_tokens, net_pad).unsqueeze(1)
            memory = torch.cat([struct_token, net_token], dim=1)  # [batch, 2, d]
            memory_pad = torch.stack([has_structure < 0.5, has_network < 0.5], dim=1)  # [batch, 2]
            memory, memory_pad = self._con_centinela(memory, memory_pad)
            fused_text = self._fusionar(
                text_tokens, text_key_padding_mask, memory, memory_pad, has_structure, has_network
            )
            pooled = masked_mean_pool(fused_text, text_key_padding_mask)
            self._ultima_representacion = pooled
            return self.classifier(pooled)

        # CROSS_ATTENTION_TOKEN_LEVEL (variante principal, ver plan Fase B)
        modality_tokens = torch.cat([structural_tokens, network_tokens], dim=1)  # [batch, 18, d]
        modality_pad = torch.cat([struct_pad, net_pad], dim=1)  # [batch, 18]
        # El centinela se antepone ANTES de la etapa 1, de modo que tampoco allí
        # pueda existir una fila con todas las posiciones enmascaradas.
        modality_tokens, modality_pad = self._con_centinela(modality_tokens, modality_pad)
        modality_tokens = self.modality_encoder(modality_tokens, modality_pad)  # Stage 1
        fused_text = self._fusionar(
            text_tokens, text_key_padding_mask, modality_tokens, modality_pad,
            has_structure, has_network,
        )  # Stage 2
        pooled = masked_mean_pool(fused_text, text_key_padding_mask)
        self._ultima_representacion = pooled
        return self.classifier(pooled)

    def get_optimizer_param_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """
        Grupos de parámetros para AdamW con LR discriminativo: backbone de DistilBERT
        a `backbone_lr` (más bajo), resto de la arquitectura (tokenizers, fusión,
        cabeza de clasificación) a `head_lr` (más alto) -- ver justificación en el plan,
        Fase B, sección "Rama de texto".

        Los parámetros sueltos del módulo (la compuerta modal, el token centinela y los
        vectores de ausencia) no pertenecen a ningún submódulo y `named_children()` no
        los alcanza; se recogen aparte para que el optimizador no los ignore en silencio.
        """
        nombres_del_codificador = {
            id(p) for p in self.text_encoder.parameters() if p.requires_grad
        }
        grupos = self.text_encoder.get_param_groups(backbone_lr, head_lr)

        resto = [
            (nombre, p)
            for nombre, p in self.named_parameters()
            if p.requires_grad and id(p) not in nombres_del_codificador
        ]
        con_decaimiento = [p for nombre, p in resto if not sin_decaimiento(nombre)]
        sin_decaer = [p for nombre, p in resto if sin_decaimiento(nombre)]
        if con_decaimiento:
            grupos.append({"params": con_decaimiento, "lr": head_lr})
        if sin_decaer:
            grupos.append({"params": sin_decaer, "lr": head_lr, "weight_decay": 0.0})
        return grupos
