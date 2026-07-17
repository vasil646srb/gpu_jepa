"""
🚀 Streaming обучение Text-JEPA (PyTorch/Sentence-Transformers)
- Поддержка любых embedding моделей
- AMP (bfloat16) для ускорения на GPU
- Streaming pipeline: скачивание → инференс → обучение → удаление
- Режимы: fixed (N примеров) или full (весь файл, multiple shards)
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
from config import Config, DEVICE, get_amp_dtype, get_parquet_filenames
from models import TextJEPAEncoder, JEPAPredictor
from losses import VICRegLoss, compute_mask_indices, update_ema
from dataset import ShardedEmbeddingsDataset


# ==========================================
# EMBEDDING ENGINE
# ==========================================
class EmbeddingEngine:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.type = None
        self.model = None

        print(f"📦 Загрузка embedding модели: {model_name}")

        is_local = os.path.exists(model_name)
        name_lower = model_name.lower()

        if Config.embedding_backend == "flag_m3" or (Config.embedding_backend == "auto" and "bge-m3" in name_lower):
            try:
                from FlagEmbedding import BGEM3FlagModel
                self.model = BGEM3FlagModel(model_name, use_fp16=True, device=device)
                self.type = "flag_m3"
                print(f"   🎯 Backend: FlagEmbedding (BGE-M3 optimized)")
            except Exception as e:
                print(f"   ⚠️  FlagEmbedding недоступен: {e}")
                print(f"   🔄 Fallback на Sentence-Transformers")
                self._load_sentence_transformers(model_name)
        else:
            self._load_sentence_transformers(model_name)

        self.dim = self._detect_dim()
        print(f"   📐 Размерность (hidden_dim): {self.dim}")
        print(f"   💻 Устройство: {self.device}")

    def _load_sentence_transformers(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=True)
        self.model.max_seq_length = Config.embedding_max_length
        self.type = "sentence"
        print(f"   🎯 Backend: Sentence-Transformers")

    def _detect_dim(self) -> int:
        test_emb = self.encode(["test sentence"], batch_size=1)
        return test_emb.shape[-1]  # hidden_dim, not seq_len!

    def encode(self, texts, batch_size=8, max_length=None, **kwargs):
        all_embs = []
        tokenizer = self.model.tokenizer
        transformer = self.model[0].auto_model
        max_len = max_length if max_length is not None else getattr(self.model, 'max_seq_length', 128)

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = transformer(**inputs)
                embs = outputs.last_hidden_state
                all_embs.append(embs.cpu())

        return torch.cat(all_embs, dim=0).float().numpy()


# ==========================================
# ПРЕДВЫЧИСЛЕНИЕ: FIXED MODE (как раньше)
# ==========================================
def precompute_shard_fixed(parquet_path, config, engine, shard_idx):
    """Предвычисляет эмбеддинги: только первые examples_per_file примеров."""
    import pyarrow.parquet as pq

    print(f"\n🔧 [FIXED MODE] Предвычисление эмбеддингов ({engine.device})...")
    table = pq.read_table(parquet_path, columns=[Config.text_column])
    texts = table[Config.text_column].to_pylist()[:config.examples_per_file]
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > Config.min_text_length]

    if not texts:
        raise ValueError(f"Нет валидных текстов в {parquet_path}")

    os.makedirs(Config.shards_dir, exist_ok=True)
    shard_path = os.path.join(Config.shards_dir, f"shard_{shard_idx:03d}.npy")
    mask_path = os.path.join(Config.shards_dir, f"mask_{shard_idx:03d}.npy")

    all_embeddings = []
    batch_size = Config.parquet_batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        embeddings_np = engine.encode(
            batch_texts,
            batch_size=min(batch_size, Config.embedding_encode_batch),
            max_length=config.max_seq_len
        )
        all_embeddings.append(embeddings_np)
        if (i // batch_size) % 10 == 0:
            print(f"   📊 Обработано: {min(i+batch_size, len(texts))}/{len(texts)}")

    embeddings = np.concatenate(all_embeddings, axis=0)
    masks = np.zeros((len(embeddings), config.max_seq_len), dtype=bool)

    np.save(shard_path, embeddings)
    np.save(mask_path, masks)
    print(f"   💾 Шард создан: {len(embeddings)} примеров → {shard_path}")
    return shard_path, mask_path


# ==========================================
# ПРЕДВЫЧИСЛЕНИЕ: FULL MODE (весь файл, multiple shards)
# ==========================================
def precompute_shard_full(parquet_path, config, engine, file_idx):
    """
    Предвычисляет эмбеддинги из ВСЕГО parquet-файла.
    Создаёт multiple shards по config.shard_size примеров каждый.
    Возвращает список (shard_path, mask_path).
    """
    import pyarrow.parquet as pq

    print(f"\n🔧 [FULL MODE] Обработка всего файла: {Path(parquet_path).name}")

    # Читаем метаданные без загрузки данных
    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"   📊 Всего строк в файле: {total_rows:,}")
    print(f"   📦 Размер шарда: {config.shard_size:,} примеров")
    estimated_shards = (total_rows + config.shard_size - 1) // config.shard_size
    print(f"   🎯 Оценка шардов: ~{estimated_shards}")

    os.makedirs(Config.shards_dir, exist_ok=True)

    shard_paths = []
    current_texts = []
    current_count = 0
    shard_idx = file_idx * 1000  # offset для уникальности

    # Читаем parquet батчами (потоково, без загрузки всего файла в RAM)
    batch_size = Config.parquet_batch_size
    rows_processed = 0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=[Config.text_column]
    ):
        texts = batch[Config.text_column].to_pylist()
        texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > Config.min_text_length]
        current_texts.extend(texts)
        rows_processed += len(texts)

        # Когда накопили достаточно для шарда — обрабатываем
        while len(current_texts) >= config.shard_size:
            shard_texts = current_texts[:config.shard_size]
            current_texts = current_texts[config.shard_size:]

            # Предвычисляем эмбеддинги для шарда
            all_embeddings = []
            for i in range(0, len(shard_texts), Config.parquet_batch_size):
                batch_texts = shard_texts[i:i+Config.parquet_batch_size]
                embeddings_np = engine.encode(
                    batch_texts,
                    batch_size=min(len(batch_texts), Config.embedding_encode_batch),
                    max_length=config.max_seq_len
                )
                all_embeddings.append(embeddings_np)

            embeddings = np.concatenate(all_embeddings, axis=0)
            masks = np.zeros((len(embeddings), config.max_seq_len), dtype=bool)

            shard_path = os.path.join(Config.shards_dir, f"shard_{shard_idx:05d}.npy")
            mask_path = os.path.join(Config.shards_dir, f"mask_{shard_idx:05d}.npy")
            np.save(shard_path, embeddings)
            np.save(mask_path, masks)
            shard_paths.append((shard_path, mask_path))

            print(f"   💾 Шард {shard_idx}: {len(embeddings):,} примеров → {Path(shard_path).name}")
            shard_idx += 1

        if rows_processed % 50000 == 0:
            print(f"   📊 Прогресс: {rows_processed:,}/{total_rows:,} строк обработано")

    # Обрабатываем остаток (если есть)
    if current_texts:
        all_embeddings = []
        for i in range(0, len(current_texts), Config.parquet_batch_size):
            batch_texts = current_texts[i:i+Config.parquet_batch_size]
            embeddings_np = engine.encode(
                batch_texts,
                batch_size=min(len(batch_texts), Config.embedding_encode_batch),
                max_length=config.max_seq_len
            )
            all_embeddings.append(embeddings_np)

        embeddings = np.concatenate(all_embeddings, axis=0)
        masks = np.zeros((len(embeddings), config.max_seq_len), dtype=bool)

        shard_path = os.path.join(Config.shards_dir, f"shard_{shard_idx:05d}.npy")
        mask_path = os.path.join(Config.shards_dir, f"mask_{shard_idx:05d}.npy")
        np.save(shard_path, embeddings)
        np.save(mask_path, masks)
        shard_paths.append((shard_path, mask_path))
        print(f"   💾 Финальный шард {shard_idx}: {len(embeddings):,} примеров")

    print(f"\n✅ Файл обработан: {len(shard_paths)} шардов создано, {rows_processed:,} примеров")
    return shard_paths


# ==========================================
# ОБУЧЕНИЕ НА ШАРДЕ
# ==========================================
def train_on_shard(shard_path, mask_path, encoder, predictor, target_encoder,
                   optimizer, scheduler, vicreg_loss, mask_token, config,
                   global_step, shard_idx):
    """Обучает модель на одном шарде."""
    dataset = ShardedEmbeddingsDataset(shard_path, mask_path)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=Config.pin_memory and (DEVICE.type == "cuda"),
        drop_last=Config.drop_last
    )

    encoder.train()
    predictor.train()
    target_encoder.eval()

    use_amp = Config.use_amp and (DEVICE.type == "cuda")
    amp_dtype = get_amp_dtype()
    autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp)
    if use_amp:
        print(f"   🔥 AMP активирован: {amp_dtype}")

    steps_per_shard = config.steps_per_shard
    step_time_acc = 0.0
    data_time_acc = 0.0

    print(f"\n🎯 Обучение на шарде {shard_idx}: {steps_per_shard} шагов")
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
        x = x.to(DEVICE, non_blocking=True)
        key_padding_mask = key_padding_mask.to(DEVICE, non_blocking=True)

        batch_size, seq_len, _ = x.shape
        masked_indices, mask_bool = compute_mask_indices(
            (batch_size, seq_len),
            key_padding_mask,
            config.num_mask_blocks,
            config.block_size_range
        )

        x_masked = x.clone()
        mask_token_expanded = mask_token.expand(batch_size, seq_len, -1)
        x_masked[mask_bool] = mask_token_expanded[mask_bool]

        step_start = time.time()
        data_time_acc += step_start - data_start

        with autocast_ctx:
            with torch.no_grad():
                target_repr, _ = target_encoder(x, key_padding_mask)

            context_repr, _ = encoder(x_masked, key_padding_mask)
            predicted_repr = predictor(context_repr, masked_indices)

            idx_expanded = masked_indices.unsqueeze(-1).expand(-1, -1, target_repr.size(-1))
            target_masked = target_repr.gather(1, idx_expanded)

            target_masked = target_masked.reshape(-1, target_repr.size(-1))
            predicted_masked = predicted_repr.reshape(-1, predicted_repr.size(-1))

            total_loss, mse_loss, var_loss, cov_loss = vicreg_loss(
                predicted_masked.float(), target_masked.float()
            )

        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()),
            config.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        update_ema(encoder, target_encoder, config.ema_tau)

        step_time_acc += time.time() - step_start
        step_in_shard += 1
        global_step += 1

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
    print(f"   Режим: {config.file_process_mode}")
    if config.file_process_mode == "fixed":
        print(f"   Примеров/файл: {examples_per_file}")
    else:
        print(f"   Размер шарда: {config.shard_size}")
    print(f"   Шагов/шард: {steps_per_shard}")
    print(f"   Всего шагов: {config.total_steps}")
    print("=" * 70)
    print("🚀 STREAMING ОБУЧЕНИЕ Text-JEPA")
    print("=" * 70)
    print(f"💻 Устройство: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    engine = EmbeddingEngine(config.model_path, device=DEVICE.type)

    if config.input_dim != engine.dim:
        print(f"⚠️  Автообновление input_dim: {config.input_dim} → {engine.dim}")
        config.input_dim = engine.dim

    encoder = TextJEPAEncoder(
        config.input_dim, config.hidden_dim, config.embed_dim,
        config.num_layers, config.nhead, config.max_seq_len,
        dropout=config.dropout
    ).to(DEVICE)

    predictor = JEPAPredictor(
        config.embed_dim, config.hidden_dim, config.num_layers, config.nhead,
        max_seq_len=config.max_seq_len, dropout=config.dropout
    ).to(DEVICE)

    target_encoder = TextJEPAEncoder(
        config.input_dim, config.hidden_dim, config.embed_dim,
        config.num_layers, config.nhead, config.max_seq_len,
        dropout=config.dropout
    ).to(DEVICE)

    with torch.no_grad():
        for p, p_target in zip(encoder.parameters(), target_encoder.parameters()):
            p_target.data.copy_(p.data)

    mask_token = nn.Parameter(
        torch.randn(1, 1, config.input_dim, device=DEVICE) * config.mask_token_init_std
    )

    params = list(encoder.parameters()) + list(predictor.parameters()) + [mask_token]
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
        return max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    vicreg_loss = VICRegLoss(
        config.mse_weight, config.var_weight, config.cov_weight, config.vicreg_gamma
    )

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

    from huggingface_hub import hf_hub_download

    print(f"\n🔍 Подготовка списка parquet-файлов {config.dataset_name}...")
    parquet_filenames = get_parquet_filenames(num_files)
    print(f"✅ Будет обработано файлов: {len(parquet_filenames)}")

    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    os.makedirs(Config.shards_dir, exist_ok=True)

    for file_idx in range(start_file_idx, len(parquet_filenames)):
        parquet_filename = parquet_filenames[file_idx]

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

        # Выбираем режим обработки
        if config.file_process_mode == "full":
            # FULL MODE: обрабатываем весь файл, создаём multiple shards
            shard_paths = precompute_shard_full(parquet_path, config, engine, file_idx)

            for shard_idx, (shard_path, mask_path) in enumerate(shard_paths):
                actual_shard_idx = file_idx * 1000 + shard_idx
                print(f"\n📚 Загрузка шарда {actual_shard_idx}...")
                global_step = train_on_shard(
                    shard_path, mask_path, encoder, predictor, target_encoder,
                    optimizer, scheduler, vicreg_loss, mask_token, config,
                    global_step, actual_shard_idx
                )

                # Сохранение чекпоинта после каждого шарда
                ckpt_path = os.path.join(Config.checkpoint_dir, f"jepa_shard_{actual_shard_idx:05d}.pt")
                torch.save({
                    'context_encoder_state': encoder.state_dict(),
                    'predictor_state': predictor.state_dict(),
                    'target_encoder_state': target_encoder.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'mask_token': mask_token.data,
                    'global_step': global_step,
                    'shard_idx': actual_shard_idx,
                    'input_dim': config.input_dim,
                    'embed_dim': config.embed_dim,
                    'model_name': config.model_path,
                }, ckpt_path)
                print(f"\n💾 [ЧЕКПОИНТ] Шард {actual_shard_idx} → {ckpt_path}")

                # Очистка шарда
                try:
                    os.remove(shard_path)
                    os.remove(mask_path)
                except Exception:
                    pass

        else:
            # FIXED MODE: как раньше, только первые N примеров
            shard_path, mask_path = precompute_shard_fixed(
                parquet_path, config, engine, file_idx
            )

            print(f"\n📚 Загрузка шарда {file_idx}...")
            global_step = train_on_shard(
                shard_path, mask_path, encoder, predictor, target_encoder,
                optimizer, scheduler, vicreg_loss, mask_token, config,
                global_step, file_idx
            )

            ckpt_path = os.path.join(Config.checkpoint_dir, f"jepa_shard_{file_idx:03d}.pt")
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

            try:
                os.remove(shard_path)
                os.remove(mask_path)
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
    parser.add_argument("--num-files", type=int, default=Config.num_files_to_process)
    parser.add_argument("--examples-per-file", type=int, default=Config.examples_per_file)
    parser.add_argument("--steps-per-shard", type=int, default=Config.steps_per_shard)
    parser.add_argument("--mode", type=str, choices=["fixed", "full"], default=Config.file_process_mode,
                        help="fixed: N примеров/файл | full: весь файл, multiple shards")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = Config()
    config.file_process_mode = args.mode
    config.examples_per_file = args.examples_per_file

    streaming_train(
        config, args.num_files, args.examples_per_file,
        args.steps_per_shard, args.resume
    )


if __name__ == "__main__":
    main()

