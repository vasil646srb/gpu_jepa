"""
🚀 Streaming обучение Text-JEPA (PyTorch/Sentence-Transformers)
- Поддержка любых embedding моделей (BGE-M3, Qwen3-Embedding, E5, GTE и др.)
- Автоопределение размерности и типа модели
- AMP (bfloat16) для ускорения на RTX 3090/A100
- Streaming pipeline: скачивание → инференс → обучение → удаление
"""
import os
import sys
import argparse
import time
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config, DEVICE
from models import TextJEPAEncoder, JEPAPredictor
from losses import VICRegLoss, compute_mask_indices, update_ema
from dataset import ShardedEmbeddingsDataset

# ==========================================
# КОНСТАНТЫ
# ==========================================
CHECKPOINTS_DIR = "./checkpoints"
SHARDS_DIR = "./shards"


# ==========================================
# УНИВЕРСАЛЬНЫЙ EMBEDDING ENGINE
# ==========================================
class EmbeddingEngine:
    """
    Универсальный движок для создания эмбеддингов.
    Автоматически определяет тип модели и выбирает оптимальный backend:
      - BGE-M3 → FlagEmbedding (самый быстрый для M3)
      - Остальные → Sentence-Transformers (универсальный)
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.type = None
        self.model = None
        
        print(f"📦 Загрузка embedding модели: {model_name}")
        
        # Определяем тип модели
        is_local = os.path.exists(model_name)
        name_lower = model_name.lower()
        
        # BGE-M3 → используем FlagEmbedding (быстрее)
        if "bge-m3" in name_lower:
            try:
                from FlagEmbedding import BGEM3FlagModel
                self.model = BGEM3FlagModel(
                    model_name,
                    use_fp16=True,
                    device=device
                )
                self.type = "flag_m3"
                print(f"   🎯 Backend: FlagEmbedding (BGE-M3 optimized)")
            except Exception as e:
                print(f"   ⚠️  FlagEmbedding недоступен: {e}")
                print(f"   🔄 Fallback на Sentence-Transformers")
                self._load_sentence_transformers(model_name)
        else:
            self._load_sentence_transformers(model_name)
        
        # Определяем размерность
        self.dim = self._detect_dim()
        print(f"   📐 Размерность: {self.dim}")
        print(f"   💻 Устройство: {self.device}")
    
    def _load_sentence_transformers(self, model_name: str):
        """Загружает модель через sentence-transformers."""
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            trust_remote_code=True
        )
        self.type = "sentence"
        print(f"   🎯 Backend: Sentence-Transformers")
    
    def _detect_dim(self) -> int:
        """Определяет размерность эмбеддингов."""
        test_emb = self.encode(["test sentence"], batch_size=1)
        return test_emb.shape[1]
    
    def encode(self, texts, batch_size=32, max_length=512):
        """
        Возвращает numpy array нормализованных эмбеддингов [N, dim]
        """
        if self.type == "flag_m3":
            output = self.model.encode(
                texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
                show_progress_bar=False
            )
            embeddings = output['dense_vecs']
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                truncate=True
            )
        
        # Убеждаемся что это numpy float32
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)
        elif embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        return embeddings


# ==========================================
# ЭТАП 1: ПРЕДВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ
# ==========================================
def precompute_shard(parquet_path, config, engine, shard_idx):
    """Предвычисляет эмбеддинги из parquet и сохраняет в шард."""
    import pyarrow.parquet as pq
    
    print(f"\n🔧 Предвычисление эмбеддингов ({engine.device})...")
    table = pq.read_table(parquet_path, columns=["text"])
    texts = table["text"].to_pylist()[:config.examples_per_file]
    
    # Фильтруем валидные тексты
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 10]
    
    if not texts:
        raise ValueError(f"Нет валидных текстов в {parquet_path}")
    
    os.makedirs(SHARDS_DIR, exist_ok=True)
    shard_path = os.path.join(SHARDS_DIR, f"shard_{shard_idx:03d}.npy")
    mask_path = os.path.join(SHARDS_DIR, f"mask_{shard_idx:03d}.npy")
    
    # Обрабатываем батчами для прогресса
    all_embeddings = []
    batch_size = config.parquet_batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        embeddings_np = engine.encode(
            batch_texts,
            batch_size=min(batch_size, 64),
            max_length=config.max_seq_len
        )
        all_embeddings.append(embeddings_np)
        
        if (i // batch_size) % 10 == 0:
            print(f"   📊 Обработано: {min(i+batch_size, len(texts))}/{len(texts)}")
    
    embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Создаём маску (всё валидно, так как модель сама обрабатывает padding)
    masks = np.zeros((len(embeddings), config.max_seq_len), dtype=bool)
    
    np.save(shard_path, embeddings)
    np.save(mask_path, masks)
    print(f"   💾 Шард создан: {len(embeddings)} примеров → {shard_path}")
    return shard_path, mask_path


# ==========================================
# ЭТАП 2: ОБУЧЕНИЕ НА ШАРДЕ
# ==========================================
def train_on_shard(shard_path, mask_path, encoder, predictor, target_encoder,
                   optimizer, scheduler, vicreg_loss, mask_token, config,
                   global_step, shard_idx):
    """Обучает модель на одном шарде (PyTorch с AMP)."""
    dataset = ShardedEmbeddingsDataset(shard_path, mask_path)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"), drop_last=True
    )
    
    encoder.train()
    predictor.train()
    target_encoder.eval()
    
    # === AMP (Mixed Precision) ===
    # bfloat16 - лучший выбор для RTX 3090 + VICReg
    use_amp = (DEVICE.type == "cuda")
    amp_dtype = torch.bfloat16
    autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp)
    if use_amp:
        print(f"   🔥 AMP активирован: {amp_dtype}")
    
    steps_per_shard = config.steps_per_shard
    step_time_acc = 0.0
    data_time_acc = 0.0
    
    print(f"\n🎯 Обучение на шарде {shard_idx}: {steps_per_shard} шагов (GPU: {DEVICE.type == 'cuda'})")
    print("-" * 60)
    
    step_in_shard = 0
    loader_iter = iter(loader)
    
    while step_in_shard < steps_per_shard:
        try:
            x, key_padding_mask = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, key_padding_mask = next(loader_iter)
        
        data_start = time.time()
        
        # Перенос на GPU/CPU
        x = x.to(DEVICE, non_blocking=True)
        key_padding_mask = key_padding_mask.to(DEVICE, non_blocking=True)
        
        # Генерация масок с ФИКСИРОВАННЫМ размером
        batch_size, seq_len, _ = x.shape
        masked_indices, mask_bool = compute_mask_indices(
            (batch_size, seq_len),
            key_padding_mask,
            config.num_mask_blocks,
            config.block_size_range
        )
        
        # Замаскированный вход для context encoder
        x_masked = x.clone()
        mask_token_expanded = mask_token.expand(batch_size, seq_len, -1)
        x_masked[mask_bool] = mask_token_expanded[mask_bool]
        
        step_start = time.time()
        data_time_acc += step_start - data_start
        
        # Forward pass (с AMP если доступно)
        with autocast_ctx:
            with torch.no_grad():
                target_repr, _ = target_encoder(x, key_padding_mask)
            
            context_repr, _ = encoder(x_masked, key_padding_mask)
            predicted_repr = predictor(context_repr, masked_indices)
            
            idx_expanded = masked_indices.unsqueeze(-1).expand(-1, -1, target_repr.size(-1))
            target_masked = target_repr.gather(1, idx_expanded)
            
            target_masked = target_masked.reshape(-1, target_repr.size(-1))
            predicted_masked = predicted_repr.reshape(-1, predicted_repr.size(-1))
            
            # Loss ВСЕГДА считаем в fp32 для стабильности VICReg!
            total_loss, mse_loss, var_loss, cov_loss = vicreg_loss(
                predicted_masked.float(), target_masked.float()
            )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()),
            config.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        
        # EMA обновление target encoder
        update_ema(encoder, target_encoder, config.ema_tau)
        
        step_time_acc += time.time() - step_start
        step_in_shard += 1
        global_step += 1
        
        # Логирование
        if step_in_shard % config.log_interval == 0:
            mask_pct = mask_bool.float().mean().item() * 100
            lr = optimizer.param_groups[0]['lr']
            
            gpu_info = ""
            if DEVICE.type == "cuda":
                try:
                    util = torch.cuda.utilization()
                    mem_used = torch.cuda.memory_allocated() / 1e9
                    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                    gpu_info = f" | GPU={util}% VRAM={mem_used:.1f}/{mem_total:.1f}GB"
                except Exception:
                    pass
            
            print(f"[Shard {shard_idx} | Step {global_step}] 📊 LR: {lr:.2e}")
            print(f"  ├─ 📉 Total={total_loss.item():.4f} | MSE={mse_loss.item():.4f} | "
                  f"Var={var_loss.item():.4f} | Cov={cov_loss.item():.4f}")
            print(f"  ├─ ⚙️  Grad Norm={grad_norm.item():.3f} | Масок={mask_pct:.1f}%")
            print(f"  └─ ⏱  Шаг={step_time_acc/config.log_interval:.3f}s "
                  f"(Data={data_time_acc/config.log_interval:.3f}s){gpu_info}")
            
            step_time_acc = 0.0
            data_time_acc = 0.0
    
    return global_step


# ==========================================
# ГЛАВНАЯ ФУНКЦИЯ STREAMING ОБУЧЕНИЯ
# ==========================================
def streaming_train(config, num_files, examples_per_file, steps_per_shard, resume_path=None):
    """Основной цикл streaming обучения."""
    config.examples_per_file = examples_per_file
    config.steps_per_shard = steps_per_shard
    config.num_files_to_process = num_files
    config.total_steps = num_files * steps_per_shard
    
    print(f"🎯 Конфигурация:")
    print(f"   Модель: {config.model_path}")
    print(f"   Файлов: {num_files}")
    print(f"   Примеров/файл: {examples_per_file}")
    print(f"   Шагов/шард: {steps_per_shard}")
    print(f"   Всего примеров: {num_files * examples_per_file}")
    print(f"   Всего шагов: {config.total_steps}")
    print("=" * 70)
    print("🚀 STREAMING ОБУЧЕНИЕ Text-JEPA")
    print("=" * 70)
    print(f"📁 Файлов для обработки: {num_files}")
    print(f"📊 Примеров с каждого файла: {examples_per_file}")
    print(f"🔄 Шагов обучения на шард: {steps_per_shard}")
    print(f"💻 Устройство: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Загрузка embedding модели
    engine = EmbeddingEngine(config.model_path, device=DEVICE.type)
    
    # Авто-обновление input_dim из модели
    if config.input_dim != engine.dim:
        print(f"⚠️  Автообновление input_dim: {config.input_dim} → {engine.dim}")
        config.input_dim = engine.dim
    
    # Создание моделей JEPA (все на GPU если доступно)
    encoder = TextJEPAEncoder(
        config.input_dim, config.hidden_dim, config.embed_dim,
        config.num_layers, config.nhead, config.max_seq_len
    ).to(DEVICE)
    
    predictor = JEPAPredictor(
        config.embed_dim, config.hidden_dim, config.num_layers, config.nhead
    ).to(DEVICE)
    
    target_encoder = TextJEPAEncoder(
        config.input_dim, config.hidden_dim, config.embed_dim,
        config.num_layers, config.nhead, config.max_seq_len
    ).to(DEVICE)
    
    # Инициализация target encoder как копия encoder
    with torch.no_grad():
        for p, p_target in zip(encoder.parameters(), target_encoder.parameters()):
            p_target.data.copy_(p.data)
    
    # Mask token (на том же устройстве)
    mask_token = nn.Parameter(torch.randn(1, 1, config.input_dim, device=DEVICE) * 0.02)
    
    # Оптимизатор
    params = list(encoder.parameters()) + list(predictor.parameters()) + [mask_token]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Scheduler (cosine annealing with warmup)
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Loss (с правильными весами для предотвращения коллапса)
    vicreg_loss = VICRegLoss(
        config.mse_weight, config.var_weight, config.cov_weight, config.vicreg_gamma
    )
    
    # Восстановление из чекпоинта
    global_step = 0
    start_file_idx = 0
    
    if resume_path and Path(resume_path).exists():
        print(f"\n🔄 Восстановление из {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        
        encoder.load_state_dict(ckpt['context_encoder_state'])
        predictor.load_state_dict(ckpt['predictor_state'])
        target_encoder.load_state_dict(ckpt['target_encoder_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        mask_token.data = ckpt['mask_token'].to(DEVICE)
        
        global_step = ckpt.get('global_step', 0)
        start_file_idx = ckpt.get('shard_idx', 0) + 1
        
        print(f"   Продолжаем с шарда {start_file_idx}, шаг {global_step}")
    else:
        print("\n🆕 Обучение с нуля")
    
    # ==========================================
    # ГЕНЕРАЦИЯ СПИСКА ФАЙЛОВ (БЕЗ СКАЧИВАНИЯ)
    # ==========================================
    print("\n🔍 Подготовка списка parquet-файлов HuggingFaceFW/fineweb-edu...")
    from huggingface_hub import hf_hub_download
    
    parquet_filenames = [f"sample/100BT/{i:03d}_00000.parquet" for i in range(num_files)]
    print(f"✅ Будет обработано файлов: {len(parquet_filenames)}")
    
    # Создание директорий
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(SHARDS_DIR, exist_ok=True)
    
    # ==========================================
    # ОСНОВНОЙ ЦИКЛ (СКАЧИВАНИЕ ПО ОДНОМУ)
    # ==========================================
    for file_idx in range(start_file_idx, len(parquet_filenames)):
        parquet_filename = parquet_filenames[file_idx]
        
        # ⬇️ СКАЧИВАЕМ ТОЛЬКО ОДИН ФАЙЛ ПЕРЕД ОБРАБОТКОЙ
        try:
            parquet_path = hf_hub_download(
                repo_id=config.dataset_name,
                filename=parquet_filename,
                repo_type="dataset",
                force_download=False
            )
        except Exception as e:
            print(f"⚠️  Не удалось скачать {parquet_filename}: {e}")
            continue
        
        print(f"\n{'='*70}")
        print(f"📥 ФАЙЛ {file_idx+1}/{len(parquet_filenames)}: {Path(parquet_path).name}")
        print(f"{'='*70}")
        
        # Предвычисление эмбеддингов
        shard_path, mask_path = precompute_shard(
            parquet_path, config, engine, file_idx
        )
        
        # Обучение на шарде
        print(f"\n📚 Загрузка шарда {file_idx}...")
        global_step = train_on_shard(
            shard_path, mask_path, encoder, predictor, target_encoder,
            optimizer, scheduler, vicreg_loss, mask_token, config,
            global_step, file_idx
        )
        
        # Сохранение чекпоинта
        ckpt_path = os.path.join(CHECKPOINTS_DIR, f"jepa_shard_{file_idx:03d}.pt")
        torch.save({
            'context_encoder_state': encoder.state_dict(),
            'predictor_state': predictor.state_dict(),
            'target_encoder_state': target_encoder.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'mask_token': mask_token.data,
            'global_step': global_step,
            'shard_idx': file_idx,
            'input_dim': config.input_dim,
            'embed_dim': config.embed_dim,
            'model_name': config.model_path,
        }, ckpt_path)
        print(f"\n💾 [ЧЕКПОИНТ] Шард {file_idx} → {ckpt_path}")
        
        # Очистка временных файлов
        try:
            os.remove(shard_path)
            os.remove(mask_path)
            print(f"   🗑  Шард удален (освобождено место)")
        except Exception:
            pass
        
        if config.delete_parquet_after and Path(parquet_path).exists():
            try:
                size_gb = Path(parquet_path).stat().st_size / 1e9
                os.remove(parquet_path)
                print(f"   🗑  Parquet удален (освобождено {size_gb:.2f} GB)")
            except Exception:
                pass
        
        print(f"\n✅ Файл {file_idx+1} полностью обработан")
    
    print(f"\n{'='*70}")
    print("🎉 STREAMING ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"{'='*70}")


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Streaming обучение Text-JEPA")
    parser.add_argument("--num-files", type=int, default=10)
    parser.add_argument("--examples-per-file", type=int, default=5000)
    parser.add_argument("--steps-per-shard", type=int, default=500)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    config = Config()
    streaming_train(
        config, args.num_files, args.examples_per_file,
        args.steps_per_shard, args.resume
    )


if __name__ == "__main__":
    main()
