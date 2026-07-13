"""
Датасет для streaming обучения Text-JEPA.
Использует memory-mapped numpy arrays для эффективной работы с большими шардами.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ShardedEmbeddingsDataset(Dataset):
    """
    Загружает один шард эмбеддингов и масок через mmap.
    Не загружает весь шард в RAM, читает напрямую с диска.
    """
    
    def __init__(self, shard_path, mask_path):
        """
        Args:
            shard_path: путь к .npy файлу с эмбеддингами [N, seq_len, dim]
            mask_path: путь к .npy файлу с масками [N, seq_len]
        """
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Файл шарда не найден: {shard_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Файл маски не найден: {mask_path}")
        
        # mmap_mode='r' - чтение без загрузки в RAM
        self.embeddings = np.load(shard_path, mmap_mode='r')
        self.masks = np.load(mask_path, mmap_mode='r')
        
        if len(self.embeddings) != len(self.masks):
            raise ValueError(
                f"Несоответствие размеров: {len(self.embeddings)} эмбеддингов "
                f"и {len(self.masks)} масок"
            )
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        emb = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.bool)
        return emb, mask
