"""
VICReg Loss, InfoNCE Loss и функции маскирования для Text-JEPA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VICRegLoss(nn.Module):
    """
    VICReg: Variance-Invariance-Covariance Regularization.
    Предотвращает коллапс представлений в self-supervised learning.
    """

    def __init__(self, mse_weight=10.0, var_weight=1.0, cov_weight=0.02, gamma=1.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.gamma = gamma

    def forward(self, x, y):
        """
        Args:
            x: [N, D] - предсказанные представления
            y: [N, D] - целевые представления

        Returns:
            total_loss, mse_loss, var_loss, cov_loss
        """
        N, D = x.shape

        # 1. Invariance (MSE) - предсказание должно совпадать с целью
        mse_loss = F.mse_loss(x, y)

        # 2. Variance - каждое измерение должно иметь достаточную дисперсию
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        var_loss = (F.relu(self.gamma - std_x).mean() + 
                    F.relu(self.gamma - std_y).mean()) / 2

        # 3. Covariance - измерения должны быть независимыми
        x_centered = x - x.mean(dim=0)
        y_centered = y - y.mean(dim=0)
        cov_x = (x_centered.T @ x_centered) / (N - 1)
        cov_y = (y_centered.T @ y_centered) / (N - 1)

        # Обнуляем диагональ (дисперсию)
        diag_mask = torch.eye(D, device=x.device, dtype=torch.bool)
        cov_x = cov_x.masked_fill(diag_mask, 0.0)
        cov_y = cov_y.masked_fill(diag_mask, 0.0)

        cov_loss = (cov_x.pow(2).sum() + cov_y.pow(2).sum()) / D

        # Общий loss
        total_loss = (self.mse_weight * mse_loss + 
                      self.var_weight * var_loss + 
                      self.cov_weight * cov_loss)

        return total_loss, mse_loss, var_loss, cov_loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) для контрастивного обучения JEPA.
    Заставляет модель раздвигать разные тексты в embedding space.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, pooled_ctx, pooled_tgt):
        """
        Args:
            pooled_ctx: [N, D] - нормализованные pooled от context encoder
            pooled_tgt: [N, D] - нормализованные pooled от target encoder (EMA)

        Returns:
            contrastive_loss
        """
        N = pooled_ctx.size(0)

        # Матрица схожестей
        logits = torch.mm(pooled_ctx, pooled_tgt.t()) / self.temperature  # [N, N]
        labels = torch.arange(N, device=pooled_ctx.device)

        # Симметричный InfoNCE
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)

        return (loss_i + loss_t) / 2


def compute_mask_indices(shape, key_padding_mask, num_mask_blocks, block_size_range):
    """
    Генерирует индексы замаскированных токенов с улучшенным span-based маскированием.
    Блоки меньше, их больше — маскируются разные части текста, а не крупные куски.

    Args:
        shape: (batch_size, seq_len)
        key_padding_mask: [batch, seq_len] - True для padding токенов
        num_mask_blocks: количество блоков для маскирования
        block_size_range: (min_ratio, max_ratio) размера блока

    Returns:
        masked_indices: [batch, num_masked] - LongTensor индексов замаскированных позиций
        mask_bool: [batch, seq_len] - булева маска
    """
    batch_size, seq_len = shape
    device = key_padding_mask.device

    # Меньшие блоки, больше блоков — лучшее покрытие текста
    min_ratio, max_ratio = block_size_range
    block_size = int(np.random.uniform(min_ratio, max_ratio) * seq_len)
    block_size = max(1, min(block_size, seq_len // max(1, num_mask_blocks)))
    num_masked = num_mask_blocks * block_size
    num_masked = min(num_masked, seq_len - 1)

    masked_indices = torch.zeros(batch_size, num_masked, dtype=torch.long, device=device)
    mask_bool = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

    valid_lengths = (~key_padding_mask).sum(dim=1)  # [batch]

    # --- Быстрый векторизованный путь --------------------------------
    # Делим валидную область каждого примера на num_mask_blocks
    # непересекающихся сегментов и берём случайный блок block_size
    # внутри каждого сегмента. Это по построению исключает перекрытия
    # блоков и не требует Python-цикла по батчу.
    segment_len = torch.div(valid_lengths, num_mask_blocks, rounding_mode='floor')
    can_vectorize = segment_len >= block_size  # [batch] bool

    vec_idx = torch.where(can_vectorize)[0]
    if len(vec_idx) > 0:
        seg_len_v = segment_len[vec_idx]                      # [V]
        max_start_v = (seg_len_v - block_size).clamp(min=0)   # [V]

        rand = torch.rand(len(vec_idx), num_mask_blocks, device=device)
        starts_in_segment = (rand * (max_start_v + 1).unsqueeze(1).float()).long()  # [V, num_blocks]

        block_offsets = torch.arange(num_mask_blocks, device=device).unsqueeze(0) * seg_len_v.unsqueeze(1)
        block_starts = block_offsets + starts_in_segment       # [V, num_blocks]

        within_block = torch.arange(block_size, device=device).view(1, 1, -1)
        idx = block_starts.unsqueeze(-1) + within_block        # [V, num_blocks, block_size]
        idx = idx.reshape(len(vec_idx), -1).clamp(max=seq_len - 1)  # [V, num_masked]

        masked_indices[vec_idx] = idx
        mask_bool[vec_idx] = mask_bool[vec_idx].scatter(1, idx, True)

    # --- Медленный путь только для коротких примеров (обычно редкость) --
    fallback_idx = torch.where(~can_vectorize)[0]
    for b in fallback_idx.tolist():
        valid_positions = torch.where(~key_padding_mask[b])[0]

        if len(valid_positions) <= num_masked:
            if len(valid_positions) == 0:
                chosen = torch.zeros(num_masked, dtype=torch.long, device=device)
            else:
                repeats = (num_masked // len(valid_positions)) + 1
                chosen = valid_positions.repeat(repeats)[:num_masked]
        else:
            chosen = []
            available = valid_positions.tolist()

            for _ in range(num_mask_blocks):
                if len(available) < block_size:
                    chosen.extend(available)
                    break

                max_start = len(available) - block_size
                start_idx = np.random.randint(0, max_start + 1)
                block = available[start_idx:start_idx + block_size]
                chosen.extend(block)

                gap = 1
                remove_start = max(0, start_idx - gap)
                remove_end = min(len(available), start_idx + block_size + gap)
                available = available[:remove_start] + available[remove_end:]

            if len(chosen) < num_masked:
                chosen.extend([chosen[0]] * (num_masked - len(chosen)))

            chosen = torch.tensor(chosen[:num_masked], dtype=torch.long, device=device)

        masked_indices[b] = chosen
        mask_bool[b, chosen] = True

    return masked_indices, mask_bool


def update_ema(model, model_ema, tau):
    """Обновляет параметры EMA (Exponential Moving Average) модели."""
    with torch.no_grad():
        for p, p_ema in zip(model.parameters(), model_ema.parameters()):
            p_ema.data.mul_(tau).add_(p.data, alpha=1 - tau)
