"""
🚀 Streaming обучение Text-JEPA
- Строгое пофайловое скачивание данных (не загружает весь датасет сразу)
- ONNX Runtime на GPU для предвычисления BGE-эмбеддингов
- PyTorch на GPU для обучения JEPA
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
import onnxruntime as ort
from huggingface_hub import hf_hub_download, snapshot_download

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config, DEVICE
from models import TextJEPAEncoder, JEPAPredictor
from losses import VICRegLoss, compute_mask_indices, update_ema
from dataset import ShardedEmbeddingsDataset

# ==========================================
# КОНСТАНТЫ
# ==========================================
HF_REPO_ID = "vasil646/jepa_text"
MODEL_PATH = "./bge-large-en-v1.5-onnx"
CHECKPOINTS_DIR = "./checkpoints"
SHARDS_DIR = "./shards"


# ==========================================
# АВТОЗАГРУЗКА БАЗОВОЙ МОДЕЛИ
# ==========================================
def ensure_base_model_available():
    """Скачивает базовую модель BGE, если она отсутствует локально."""
    if Path(MODEL_PATH).exists() and list(Path(MODEL_PATH).glob("*.onnx")):
        print(f"✅ Базовая модель найдена: {MODEL_PATH}")
        return
    
    print(f"📥 Базовая модель не найдена. Скачиваю с HF Hub...")
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir="./",
            allow_patterns=["./bge-large-en-v1.5-onnx/*"]
        )
        # Перемещаем model.onnx в корень (если он в подпапке onnx/)
        onnx_subdir = Path(MODEL_PATH) / "onnx" / "model.onnx"
        if onnx_subdir.exists() and not (Path(MODEL_PATH) / "model.onnx").exists():
            shutil.move(str(onnx_subdir), str(Path(MODEL_PATH) / "model.onnx"))
        print(f"✅ Базовая модель загружена в {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  Не удалось скачать с вашего репозитория: {e}")
        print(f"📥 Fallback: скачиваю Xenova/bge-small-en-v1.5...")
        snapshot_download(repo_id="Xenova/bge-large-en-v1.5", local_dir=MODEL_PATH)
        onnx_subdir = Path(MODEL_PATH) / "onnx" / "model.onnx"
        if onnx_subdir.exists() and not (Path(MODEL_PATH) / "model.onnx").exists():
            shutil.move(str(onnx_subdir), str(Path(MODEL_PATH) / "model.onnx"))


# ==========================================
# ONNX INFERENCE С ПОДДЕРЖКОЙ GPU
# ==========================================
def create_onnx_session(model_path):
    """Создает ONNX сессию с автоматическим выбором GPU/CPU."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    available_providers = ort.get_available_providers()
    
    if DEVICE.type == "cuda" and "CUDAExecutionProvider" in available_providers:
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': int(4 * 1024 * 1024 * 1024),  # 4 GB для ONNX
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider'
        ]
        sess_options.intra_op_num_threads = 1
        print(f"🎮 ONNX Runtime: CUDAExecutionProvider активирован (GPU для эмбеддингов)")
    else:
        providers = ['CPUExecutionProvider']
        import multiprocessing
        sess_options.intra_op_num_threads = multiprocessing.cpu_count()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        print(f"💻 ONNX Runtime: CPUExecutionProvider (CPU для эмбеддингов)")
    
    onnx_files = sorted([f for f in os.listdir(model_path) if f.endswith('.onnx')])
    if not onnx_files:
        raise FileNotFoundError(f"ONNX файлы не найдены в {model_path}")
    
    onnx_path = os.path.join(model_path, onnx_files[0])
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)
    
    print(f"📦 Активные провайдеры: {session.get_providers()}")
    expected_inputs = [inp.name for inp in session.get_inputs()]
    return session, expected_inputs


def compute_embeddings_batch(session, expected_inputs, tokenizer, texts, max_length=128):
    """Вычисляет BGE-эмбеддинги для батча текстов на том устройстве, где создана сессия."""
    inputs = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="np"
    )
    
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    if "token_type_ids" in expected_inputs:
        if "token_type_ids" in inputs:
            ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
        else:
            ort_inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"], dtype=np.int64)
    
    outputs = session.run(None, ort_inputs)
    
    embeddings = torch.tensor(outputs[0], dtype=torch.float32)
    embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-9)
    key_padding_mask = torch.tensor(inputs["attention_mask"] == 0, dtype=torch.bool)
    
    return embeddings, key_padding_mask


# ==========================================
# ЭТАП 1: ПРЕДВЫЧИСЛЕНИЕ ЭМБЕДДИНГОВ
# ==========================================
def precompute_shard(parquet_path, config, session, expected_inputs, tokenizer, shard_idx):
    """Предвычисляет эмбеддинги из parquet и сохраняет в шард."""
    import pyarrow.parquet as pq
    
    print(f"\n🔧 Предвычисление эмбеддингов ({'GPU' if DEVICE.type == 'cuda' else 'CPU'})...")
    table = pq.read_table(parquet_path, columns=["text"])
    texts = table["text"].to_pylist()[:config.examples_per_file]
    
    os.makedirs(SHARDS_DIR, exist_ok=True)
    shard_path = os.path.join(SHARDS_DIR, f"shard_{shard_idx:03d}.npy")
    mask_path = os.path.join(SHARDS_DIR, f"mask_{shard_idx:03d}.npy")
    
    all_embeddings = []
    all_masks = []
    
    batch_size = config.parquet_batch_size
    for i in range(0, len(texts), batch_size):
        batch_texts = [t for t in texts[i:i+batch_size] if isinstance(t, str) and len(t.strip()) > 10]
        if not batch_texts:
            continue
        
        embeddings, masks = compute_embeddings_batch(
            session, expected_inputs, tokenizer, batch_texts, config.max_seq_len
        )
        all_embeddings.append(embeddings)
        all_masks.append(masks)
        
        if (i // batch_size) % 10 == 0:
            print(f"   📊 Обработано: {min(i+batch_size, len(texts))}/{len(texts)}")
    
    if not all_embeddings:
        raise ValueError(f"Нет валидных текстов в {parquet_path}")
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    masks = torch.cat(all_masks, dim=0).numpy()
    
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
    """Обучает модель на одном шарде (PyTorch на GPU)."""
    dataset = ShardedEmbeddingsDataset(shard_path, mask_path)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"), drop_last=True
    )
    
    encoder.train()
    predictor.train()
    target_encoder.eval()
    
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
        
        # Forward pass
        with torch.no_grad():
            # ВАЖНО: target_repr идет первым, pooled вторым
            target_repr, _ = target_encoder(x, key_padding_mask)
        
        context_repr, _ = encoder(x_masked, key_padding_mask)
        
        # Predictor получает индексы замаскированных позиций
        predicted_repr = predictor(context_repr, masked_indices)
        
        # Собираем target для тех же индексов через gather
        idx_expanded = masked_indices.unsqueeze(-1).expand(-1, -1, target_repr.size(-1))
        target_masked = target_repr.gather(1, idx_expanded)
        
        # Приводим к виду [N, D] для VICReg
        target_masked = target_masked.reshape(-1, target_repr.size(-1))
        predicted_masked = predicted_repr.reshape(-1, predicted_repr.size(-1))
        
        # Loss
        total_loss, mse_loss, var_loss, cov_loss = vicreg_loss(
            predicted_masked, target_masked
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
    
    # Автозагрузка базовой модели
    ensure_base_model_available()
    
    # Загрузка ONNX
    print("\n📦 Загрузка ONNX модели...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    session, expected_inputs = create_onnx_session(MODEL_PATH)
    
    # Создание моделей (все на GPU если доступно)
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
    
    # Loss
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
            parquet_path, config, session, expected_inputs, tokenizer, file_idx
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

def find_latest_checkpoint():
    """Находит последний чекпоинт или скачивает с HF Hub."""
    ckpt_dir = Path(CHECKPOINTS_DIR)
    if ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("jepa_shard_*.pt"))
        if checkpoints:
            return str(checkpoints[-1])
    
    print(f"📥 Локальные чекпоинты не найдены. Скачиваю с HF Hub...")
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo_id=HF_REPO_ID, local_dir="./", allow_patterns=["checkpoints/*.pt"])
        checkpoints = sorted(Path(CHECKPOINTS_DIR).glob("jepa_shard_*.pt"))
        if checkpoints:
            return str(checkpoints[-1])
    except Exception as e:
        print(f"❌ Не удалось скачать чекпоинты: {e}")
    return None

# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Streaming обучение Text-JEPA")
    parser.add_argument("--num-files", type=int, default=10)
    parser.add_argument("--examples-per-file", type=int, default=5000)
    parser.add_argument("--steps-per-shard", type=int, default=500)
    parser.add_argument("--resume", type=str, default=find_latest_checkpoint())
    args = parser.parse_args()
    
    config = Config()
    streaming_train(
        config, args.num_files, args.examples_per_file,
        args.steps_per_shard, args.resume
    )


if __name__ == "__main__":
    main()

