"""
Fusión por mezcla de expertos, con enrutamiento según la modalidad disponible.

Se incorpora como cuarta manera de tratar la ausencia de ramas, para decidirla por
medición y no por argumento. Las otras tres son el centinela con descarte
aleatorio, el entrenamiento restringido a filas completas, y la compuerta
condicionada por ejemplo.

**Qué problema ataca.** El centinela resuelve la ausencia de forma uniforme: un
mismo vector aprendido ocupa el hueco, sea cual sea la modalidad que falta y sea
cual sea el correo. Una mezcla de expertos permite que el modelo aprenda una
manera distinta de combinar según lo que tenga delante: un experto puede
especializarse en correo con marcado y otro en correo sin él, y la compuerta
decide por ejemplo.

**Por qué se espera poco de ella, y aun así se mide.** La auditoría de
características midió que ninguna de las dieciocho no textuales informa más sobre
la clase que sobre la colección de origen --el mayor cociente es 0.0933--. Si la
información no está en la entrada, ninguna arquitectura la extrae, y ese es el
cuello de botella medido. Se implementa porque cierra la objeción de que el
resultado nulo proceda de un mecanismo de fusión insuficiente, que es una
alternativa razonable mientras no se descarte.

**Formulación.** Con `T` la representación textual agregada, `E` la estructural y
`R` la de red, cada experto `k` produce una fusión propia y la compuerta las
mezcla:

    g = softmax( W_g · [T ; hay_estructura ; hay_red] )      (K pesos por correo)
    salida = Σ_k g_k · Experto_k([T ; E ; R])

La compuerta ve las banderas de disponibilidad de forma explícita, que es lo que
le permite enrutar por patrón de ausencia en lugar de tener que inferirlo.

Referencia de la familia: Shazeer et al. (2017) para la formulación de mezcla de
expertos con compuerta, y Bao et al. (2022) para su uso como expertos por
modalidad en modelos multimodales.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MixtureOfModalityExperts(nn.Module):
    """Mezcla de expertos sobre las tres representaciones ya agregadas."""

    def __init__(
        self,
        d_model: int,
        n_expertos: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """
        `n_expertos = 4` no es arbitrario: hay exactamente cuatro patrones de
        disponibilidad --ninguna modalidad, solo red, solo estructura, ambas-- y
        partir de un experto por patrón da a la compuerta la capacidad de
        reproducir el enrutamiento explícito si le resulta útil, sin obligarla a
        ello. Nada impide que aprenda otra partición.
        """
        super().__init__()
        self.n_expertos = n_expertos
        act = nn.GELU if activation == "gelu" else nn.ReLU

        # Cada experto ve las tres ramas concatenadas y devuelve una fusión.
        self.expertos = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(d_model * 3),
                nn.Linear(d_model * 3, dim_feedforward),
                act(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
            )
            for _ in range(n_expertos)
        )

        # La compuerta recibe el texto agregado y las dos banderas explícitas.
        self.compuerta = nn.Linear(d_model + 2, n_expertos)
        # Inicialización pequeña: se arranca cerca de la mezcla uniforme, de modo
        # que ningún experto quede excluido antes de recibir gradiente. Un
        # arranque asimétrico colapsa la mezcla sobre uno solo y el resto muere.
        nn.init.normal_(self.compuerta.weight, std=0.02)
        nn.init.zeros_(self.compuerta.bias)

        self._ultimos_pesos: torch.Tensor | None = None

    def forward(
        self,
        texto: torch.Tensor,
        estructura: torch.Tensor,
        red: torch.Tensor,
        hay_estructura: torch.Tensor,
        hay_red: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            texto, estructura, red: `[lote, d_model]`, ya agregadas.
            hay_estructura, hay_red: `[lote]`, 1.0 si la rama está disponible.

        Returns:
            `[lote, d_model]`, la representación fusionada.
        """
        entrada = torch.cat([texto, estructura, red], dim=-1)
        contexto = torch.cat(
            [texto, hay_estructura.unsqueeze(-1).to(texto.dtype),
             hay_red.unsqueeze(-1).to(texto.dtype)],
            dim=-1,
        )
        pesos = torch.softmax(self.compuerta(contexto), dim=-1)  # [lote, K]
        self._ultimos_pesos = pesos.detach()

        salidas = torch.stack([e(entrada) for e in self.expertos], dim=1)  # [lote,K,d]
        return torch.einsum("bk,bkd->bd", pesos, salidas)

    @torch.no_grad()
    def pesos_por_ejemplo(self) -> torch.Tensor | None:
        """Reparto de la compuerta en el último lote, para el análisis por fuente.

        Es la contrapartida interpretable de la mezcla: permite comprobar si los
        expertos se especializan por patrón de disponibilidad, que es la hipótesis
        que justifica la arquitectura, en lugar de repartirse de forma uniforme.
        """
        return self._ultimos_pesos
