import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config, DEVICE
from models import TextJEPAEncoder

def load_jepa():
    config = Config()
    encoder = TextJEPAEncoder(
        config.input_dim, config.hidden_dim, config.embed_dim,
        config.num_layers, config.nhead, config.max_seq_len
    ).to(DEVICE)
    
    # Ищем последний чекпоинт
    ckpts = sorted(Path("./checkpoints").glob("jepa_shard_*.pt"))
    if not ckpts:
        print("❌ Чекпоинты не найдены!"); sys.exit(1)
    
    ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt['context_encoder_state'])
    encoder.eval()
    print(f"✅ Загружен чекпоинт: {ckpts[-1].name}\n")
    
    # Загрузка ONNX
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    sess = ort.InferenceSession(
        os.path.join(config.model_path, "model.onnx"),
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if DEVICE.type == 'cuda' else ['CPUExecutionProvider']
    )
    return encoder, tokenizer, sess, config

def get_emb(texts, encoder, tokenizer, sess, config):
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=config.max_seq_len, return_tensors="np")
    ort_in = {"input_ids": inputs["input_ids"].astype(np.int64), "attention_mask": inputs["attention_mask"].astype(np.int64)}
    if "token_type_ids" in [i.name for i in sess.get_inputs()]:
        ort_in["token_type_ids"] = inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])).astype(np.int64)
    
    out = sess.run(None, ort_in)
    x = torch.tensor(out[0], dtype=torch.float32).to(DEVICE)
    mask = torch.tensor(inputs["attention_mask"] == 0, dtype=torch.bool).to(DEVICE)
    
    with torch.no_grad():
        _, pooled = encoder(x, mask)
    return F.normalize(pooled, p=2, dim=-1).cpu().numpy()

def sim(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def main():
    encoder, tokenizer, sess, config = load_jepa()
    
    print("="*60)
    print("🥊 BOSS FIGHT: Тестирование глубинного понимания JEPA")
    print("="*60)
    
    # Тест 1
    t1a = "The company reported record profits this quarter, leading to a massive expansion of their global workforce."
    t1b = "The company reported record losses this quarter, leading to a massive reduction of their global workforce."
    e1 = get_emb([t1a, t1b], encoder, tokenizer, sess, config)
    s1 = sim(e1[0], e1[1])
    print(f"\n🧪 Тест 1: Лексическая ловушка (Антонимы)")
    print(f"   Схожесть: {s1:.3f} (Ожидается: < 0.60)")
    print(f"   Вердикт: {'✅ ПРОЙДЕН (Модель видит смысл, а не слова)' if s1 < 0.60 else '❌ ПРОВАЛ (Модель ослеплена общими словами)'}")

    # Тест 2
    t2a = "A virus injects its genetic material into a host cell, hijacking the replication machinery to produce more viruses."
    t2b = "A computer worm infiltrates a network, exploiting system vulnerabilities to execute payloads and replicate across nodes."
    e2 = get_emb([t2a, t2b], encoder, tokenizer, sess, config)
    s2 = sim(e2[0], e2[1])
    print(f"\n🧪 Тест 2: Кросс-доменная абстракция")
    print(f"   Схожесть: {s2:.3f} (Ожидается: > 0.55)")
    print(f"   Вердикт: {'✅ ПРОЙДЕН (Модель поняла структурную аналогию)' if s2 > 0.55 else '❌ ПРОВАЛ (Модель не видит скрытых паттернов)'}")

    # Тест 3
    q3 = "How to avoid flooding in an underground storage space?"
    d3_good = "Installing a sump pump and applying waterproof sealants to concrete walls are essential steps for maintaining a dry basement."
    d3_bad = "Flooding can easily destroy an underground storage space if water levels rise too quickly."
    e3 = get_emb([q3, d3_good, d3_bad], encoder, tokenizer, sess, config)
    s3_good = sim(e3[0], e3[1])
    s3_bad = sim(e3[0], e3[2])
    print(f"\n🧪 Тест 3: Асимметричный интент")
    print(f"   Запрос -> Смысловой док: {s3_good:.3f}")
    print(f"   Запрос -> Лексический док: {s3_bad:.3f}")
    print(f"   Вердикт: {'✅ ПРОЙДЕН (Модель понимает цель запроса)' if s3_good > s3_bad else '❌ ПРОВАЛ (Модель работает как поиск по ключевым словам)'}")

    # Тест 4
    t4a = "The new software update significantly improved the system's performance, making it highly stable."
    t4b = "The new software update failed to improve the system's performance, making it highly unstable."
    e4 = get_emb([t4a, t4b], encoder, tokenizer, sess, config)
    s4 = sim(e4[0], e4[1])
    print(f"\n🧪 Тест 4: Ловушка отрицания")
    print(f"   Схожесть: {s4:.3f} (Ожидается: < 0.75)")
    print(f"   Вердикт: {'✅ ПРОЙДЕН (Модель чувствует инверсию смысла)' if s4 < 0.75 else '❌ ПРОВАЛ (Отрицание проигнорировано)'}")

    # Тест 5
    t5a = "The engineer refactored the legacy codebase, removing redundant loops and fixing the race condition that caused intermittent crashes."
    t5b = "The engineer introduced a race condition into the codebase by adding redundant loops, causing the application to crash intermittently."
    e5 = get_emb([t5a, t5b], encoder, tokenizer, sess, config)
    s5 = sim(e5[0], e5[1])
    print(f"\n🧪 Тест 5: Программирование (Исправление бага vs внесение бага)")
    print(f"   Схожесть: {s5:.3f} (Ожидается: < 0.65)")
    print(f"   Вердикт: {'✅ ПРОЙДЕН (Модель различает fix и bug при почти одинаковой лексике)' if s5 < 0.65 else '❌ ПРОВАЛ (Модель не различает исправление и внесение бага)'}")

    # Тест 6
    q6 = "Has the project plan been successfully implemented on schedule?"
    d6_good = "The team completed all milestones of the roadmap ahead of deadline and deployed the final release to production."
    d6_bad = "The project plan outlines the milestones, deliverables, and timeline for the upcoming quarter."
    e6 = get_emb([q6, d6_good, d6_bad], encoder, tokenizer, sess, config)
    s6_good = sim(e6[0], e6[1])
    s6_bad = sim(e6[0], e6[2])
    print(f"\n🧪 Тест 6: Реализация плана проекта (План vs Факт исполнения)")
    print(f"   Запрос -> Отчёт о реализации: {s6_good:.3f}")
    print(f"   Запрос -> Просто описание плана: {s6_bad:.3f}")
    print(f"   Вердикт: {'✅ ПРОЙДЕН (Модель отличает выполненный план от просто описания плана)' if s6_good > s6_bad else '❌ ПРОВАЛ (Модель не различает стадию планирования и стадию реализации)'}")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
