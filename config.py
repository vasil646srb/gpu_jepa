"""
Конфигурация Text-JEPA с автоопределением устройства
"""
import os
import warnings
import torch
import multiprocessing

# ==========================================
# НАСТРОЙКИ ОКРУЖЕНИЯ
# ==========================================
os.environ["HF_HUB_ETAG_TIMEOUT"] = "500"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "500"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*nested tensors.*")
warnings.filterwarnings("ignore", message=".*TracerWarning.*")

# ==========================================
# АВТООПРЕДЕЛЕНИЕ УСТРОЙСТВА
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CORES = multiprocessing.cpu_count()

if DEVICE.type == "cuda":
    print(f"🔥 Обнаружена GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.set_float32_matmul_precision('high')
    print(f"🎯 Режим: GPU (CUDA)")
else:
    print(f"💻 GPU не найдена. Используем CPU ({NUM_CORES} потоков)")
    torch.set_num_threads(NUM_CORES)
    torch.set_float32_matmul_precision('high')
    print(f"🎯 Режим: CPU (AVX2/FMA)")


# ==========================================
# КОНФИГУРАЦИЯ МОДЕЛИ
# ==========================================
class Config:
    # Архитектура
    input_dim = 384          # BGE-Small embedding dim
    hidden_dim = 256         # Внутренняя размерность
    embed_dim = 128          # Выходная размерность JEPA
    num_layers = 2           # Слои Transformer Encoder
    nhead = 4                # Количество голов внимания
    max_seq_len = 128        # Максимальная длина последовательности
    
    # Обучение
    batch_size = 512         # Увеличено для GPU (12GB VRAM)
    learning_rate = 3e-4
    total_steps = 50000
    warmup_steps = 1000
    weight_decay = 1e-4
    max_grad_norm = 1.0
    
    # JEPA специфика
    ema_tau = 0.996          # EMA для target encoder
    mse_weight = 10.0
    var_weight = 1.0
    cov_weight = 0.1
    vicreg_gamma = 1.0
    
    # Маскирование (фиксированный размер)
    num_mask_blocks = 2
    block_size_range = (0.15, 0.25)
    
    # Логирование
    save_interval = 1000
    eval_interval = 500
    log_interval = 50
    
    # Пути
    model_path = "./bge-small-en-v1.5-onnx-Q"
    checkpoint_dir = "./checkpoints"
    shards_dir = "./shards"
    
    # Параметры загрузки
    parquet_batch_size = 200
    shard_size = 10000
    num_files_to_process = 10
    examples_per_file = 10000
    delete_parquet_after = True
    total_examples = 0
    
    # Датасет
    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_config = "sample-10BT"


if __name__ == "__main__":
    c = Config()
    print(f"\n📋 Конфигурация:")
    print(f"   batch_size: {c.batch_size}")
    print(f"   embed_dim: {c.embed_dim}")
    print(f"   hidden_dim: {c.hidden_dim}")
    print(f"   num_layers: {c.num_layers}")
