import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config, DEVICE
from models import TextJEPAEncoder


# ==========================================
# ПРЕДВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ НА GPU
# ==========================================
class EmbeddingCache:
    """Кэширует эмбеддинги текстов в RAM (GPU)."""

    def __init__(self, model_path, device='cuda'):
        print(f"📦 Загрузка embedding модели: {model_path}")
        self.device = device
        self.st_model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
        self.tokenizer = self.st_model.tokenizer
        self.transformer = self.st_model[0].auto_model

        raw_max_len = getattr(self.st_model, 'max_seq_length', Config.max_seq_len)
        self.max_len = min(raw_max_len, Config.max_seq_len)
        print(f"   📐 max_seq_length (raw): {raw_max_len} → capped: {self.max_len}")
        print(f"   ✅ Модель загружена на {device}")

        self._cache = {}  # text -> (x_tensor, mask_tensor) on GPU

    def _compute(self, texts):
        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.transformer(**inputs)
            x = outputs.last_hidden_state.float()
        mask = (inputs["attention_mask"] == 0)
        return x, mask

    def get(self, texts):
        new_texts = [t for t in texts if t not in self._cache]
        if new_texts:
            batch_size = 32
            for i in range(0, len(new_texts), batch_size):
                batch = new_texts[i:i + batch_size]
                x, mask = self._compute(batch)
                for j, txt in enumerate(batch):
                    self._cache[txt] = (x[j:j+1], mask[j:j+1])
        xs, masks = [], []
        for t in texts:
            x, m = self._cache[t]
            xs.append(x)
            masks.append(m)
        return torch.cat(xs, dim=0), torch.cat(masks, dim=0)

    def get_jepa_embeddings(self, texts, encoder):
        x, mask = self.get(texts)
        with torch.no_grad():
            _, pooled = encoder(x, mask)
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled.cpu().numpy()

    def clear_cache(self):
        self._cache.clear()
        torch.cuda.empty_cache()
        print("🗑  Кэш очищен")


# ==========================================
# ЗАГРУЗКА МОДЕЛЕЙ
# ==========================================
def load_jepa():
    config = Config()
    encoder = TextJEPAEncoder(
        config.input_dim, config.hidden_dim, config.embed_dim,
        config.num_layers, config.nhead, config.max_seq_len,
        dropout=config.dropout
    ).to(DEVICE)

    ckpts = sorted(Path(Config.boss_checkpoint_dir).glob(Config.boss_checkpoint_pattern))
    if not ckpts:
        print("❌ Чекпоинты не найдены!"); sys.exit(1)

    ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt['context_encoder_state'])
    encoder.eval()
    print(f"✅ Загружен чекпоинт JEPA: {ckpts[-1].name}\n")

    cache = EmbeddingCache(config.model_path, device=DEVICE.type)
    return cache, encoder, config


def sim(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def main():
    cache, encoder, config = load_jepa()

    print("=" * 60)
    print("🥊 BOSS FIGHT: Тестирование глубинного понимания JEPA")
    print("=" * 60)

    # Тест 1
    t1a = "The company reported record profits this quarter, leading to a massive expansion of their global workforce."
    t1b = "The company reported record losses this quarter, leading to a massive reduction of their global workforce."
    cache.get([t1a, t1b])
    e1 = cache.get_jepa_embeddings([t1a, t1b], encoder)
    s1 = sim(e1[0], e1[1])
    thr1 = Config.boss_test1_threshold
    print(f"\n🧪 Тест 1: Лексическая ловушка (Антонимы)")
    print(f"   Схожесть: {s1:.3f} (Ожидается: < {thr1})")
    print(f"   Вердикт: {'✅ ПРОЙДЕН' if s1 < thr1 else '❌ ПРОВАЛ'}")

    # Тест 2
    t2a = "A virus injects its genetic material into a host cell, hijacking the replication machinery to produce more viruses."
    t2b = "A computer worm infiltrates a network, exploiting system vulnerabilities to execute payloads and replicate across nodes."
    cache.get([t2a, t2b])
    e2 = cache.get_jepa_embeddings([t2a, t2b], encoder)
    s2 = sim(e2[0], e2[1])
    thr2 = Config.boss_test2_threshold
    print(f"\n🧪 Тест 2: Кросс-доменная абстракция")
    print(f"   Схожесть: {s2:.3f} (Ожидается: > {thr2})")
    print(f"   Вердикт: {'✅ ПРОЙДЕН' if s2 > thr2 else '❌ ПРОВАЛ'}")

    # Тест 3
    q3 = "How to avoid flooding in an underground storage space?"
    d3_good = "Installing a sump pump and applying waterproof sealants to concrete walls are essential steps for maintaining a dry basement."
    d3_bad = "Flooding can easily destroy an underground storage space if water levels rise too quickly."
    cache.get([q3, d3_good, d3_bad])
    e3 = cache.get_jepa_embeddings([q3, d3_good, d3_bad], encoder)
    s3_good = sim(e3[0], e3[1])
    s3_bad = sim(e3[0], e3[2])
    print(f"\n🧪 Тест 3: Асимметричный интент")
    print(f"   Запрос -> Смысловой док: {s3_good:.3f}")
    print(f"   Запрос -> Лексический док: {s3_bad:.3f}")
    print(f"   Вердикт: {'✅ ПРОЙДЕН' if s3_good > s3_bad else '❌ ПРОВАЛ'}")

    # Тест 4
    t4a = "The new software update significantly improved the system's performance, making it highly stable."
    t4b = "The new software update failed to improve the system's performance, making it highly unstable."
    cache.get([t4a, t4b])
    e4 = cache.get_jepa_embeddings([t4a, t4b], encoder)
    s4 = sim(e4[0], e4[1])
    thr4 = Config.boss_test4_threshold
    print(f"\n🧪 Тест 4: Ловушка отрицания")
    print(f"   Схожесть: {s4:.3f} (Ожидается: < {thr4})")
    print(f"   Вердикт: {'✅ ПРОЙДЕН' if s4 < thr4 else '❌ ПРОВАЛ'}")

    # Тест 5
    t5a = "The engineer refactored the legacy codebase, removing redundant loops and fixing the race condition that caused intermittent crashes."
    t5b = "The engineer introduced a race condition into the codebase by adding redundant loops, causing the application to crash intermittently."
    cache.get([t5a, t5b])
    e5 = cache.get_jepa_embeddings([t5a, t5b], encoder)
    s5 = sim(e5[0], e5[1])
    thr5 = Config.boss_test5_threshold
    print(f"\n🧪 Тест 5: Программирование (Исправление бага vs внесение бага)")
    print(f"   Схожесть: {s5:.3f} (Ожидается: < {thr5})")
    print(f"   Вердикт: {'✅ ПРОЙДЕН' if s5 < thr5 else '❌ ПРОВАЛ'}")

    # Тест 6
    q6 = "Has the project plan been successfully implemented on schedule?"
    d6_good = "The team completed all milestones of the roadmap ahead of deadline and deployed the final release to production."
    d6_bad = "The project plan outlines the milestones, deliverables, and timeline for the upcoming quarter."
    cache.get([q6, d6_good, d6_bad])
    e6 = cache.get_jepa_embeddings([q6, d6_good, d6_bad], encoder)
    s6_good = sim(e6[0], e6[1])
    s6_bad = sim(e6[0], e6[2])
    print(f"\n🧪 Тест 6: Реализация плана проекта (План vs Факт исполнения)")
    print(f"   Запрос -> Отчёт о реализации: {s6_good:.3f}")
    print(f"   Запрос -> Просто описание плана: {s6_bad:.3f}")
    print(f"   Вердикт: {'✅ ПРОЙДЕН' if s6_good > s6_bad else '❌ ПРОВАЛ'}")

    print("\n" + "=" * 60)
    cache.clear_cache()


if __name__ == "__main__":
    main()

