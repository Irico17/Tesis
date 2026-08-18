"""Encoders/tokenizers de rama (Fase B): texto (DistilBERT), estructura y red.

Cada uno produce tokens `[batch, n_tokens, d_model]` a partir de datos crudos,
para ser consumidos por `fusion/` (Stage 1 modality_encoder + Stage 2 cross_attention).
"""
