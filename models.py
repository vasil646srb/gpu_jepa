"""
Архитектуры Text-JEPA: Encoder и Predictor
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Синусоидальное позиционное кодирование."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TextJEPAEncoder(nn.Module):
    """
    Context/Target Encoder для Text-JEPA.
    Добавлен CLS-токен для агрегирования семантики (лучше чем mean-pooling).
    Pooled output нормализуется L2 для контрастивного обучения.
    """

    def __init__(self, input_dim, hidden_dim, embed_dim, num_layers, nhead, max_seq_len, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len + 1, dropout)  # +1 для CLS

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: [batch, seq_len, input_dim] - входные эмбеддинги
            key_padding_mask: [batch, seq_len] - True для padding токенов

        Returns:
            sequence_output: [batch, seq_len, embed_dim] (без CLS)
            pooled_output: [batch, embed_dim] - нормализованный CLS
        """
        batch_size = x.size(0)

        # Проекция и позиционное кодирование
        h = self.input_proj(x)

        # Добавляем CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1)  # [batch, seq_len+1, hidden]

        # Расширяем key_padding_mask для CLS (CLS не является padding)
        if key_padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=h.device)
            key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        h = self.pos_encoding(h)

        # Transformer
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # Проекция в выходное пространство
        h = self.output_proj(h)
        h = self.layer_norm(h)

        # CLS token для pooled (нормализованный)
        pooled = F.normalize(h[:, 0], p=2, dim=-1)

        # Остальные токены для sequence output (без CLS)
        sequence_output = h[:, 1:]

        return sequence_output, pooled


class JEPAPredictor(nn.Module):
    """
    Predictor: предсказывает target-представления для замаскированных позиций.
    Принимает контекстные представления и индексы замаскированных позиций.
    """

    def __init__(self, embed_dim, hidden_dim, num_layers, nhead, max_seq_len=512, dropout=0.1):
        super().__init__()

        # Позиционные эмбеддинги для замаскированных позиций
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        # Transformer для предсказания
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, context_repr, masked_indices):
        """
        Args:
            context_repr: [batch, seq_len, embed_dim] - контекстные представления
            masked_indices: [batch, num_masked] - LongTensor индексов замаскированных позиций

        Returns:
            predicted_repr: [batch, num_masked, embed_dim] - предсказанные представления
        """
        batch_size, num_masked = masked_indices.shape
        embed_dim = context_repr.size(-1)

        # Извлекаем представления для замаскированных позиций через gather
        idx_expanded = masked_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        queries = context_repr.gather(1, idx_expanded)

        # Добавляем позиционные эмбеддинги
        pos_embeds = self.pos_embed(masked_indices)
        queries = queries + pos_embeds

        # Transformer processing
        predicted = self.transformer(queries)
        predicted = self.layer_norm(predicted)

        return predicted

