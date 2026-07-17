"""
Конфигурация Text-JEPA с автоопределением устройства
Все настройки проекта собраны в одном месте.
"""
import os
import warnings
import torch
import multiprocessing

# ==========================================
# НАСТРОЙКИ ОКРУЖЕНИЯ
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
class ModelConfig:
    """Архитектурные параметры JEPA."""
    input_dim = 1024
    hidden_dim = 512
    embed_dim = 512
    num_layers = 8
    nhead = 8
    max_seq_len = 128
    dropout = 0.1


# ==========================================
# КОНФИГУРАЦИЯ ОБУЧЕНИЯ
# ==========================================
class TrainConfig:
    """Параметры обучения."""
    batch_size = 128
    learning_rate = 3e-4
    total_steps = 50000
    warmup_steps = 1000
    weight_decay = 1e-4
    max_grad_norm = 1.0

    # Шарды и потоковая загрузка
    steps_per_shard = 1000
    parquet_batch_size = 2000
    shard_size = 10000
    num_files_to_process = 10
    examples_per_file = 10000
    delete_parquet_after = True
    file_process_mode = "fixed"  # "fixed" или "full"

    # DataLoader
    num_workers = 0
    pin_memory = True
    drop_last = True

    # Embedding engine
    embedding_backend = "auto"
    embedding_max_length = 256
    embedding_encode_batch = 64

    # Пути
    checkpoint_dir = "./checkpoints"
    shards_dir = "./shards"

    # Логирование и сохранение
    save_interval = 1000
    eval_interval = 500
    log_interval = 50

    # AMP
    use_amp = True
    amp_dtype = "bfloat16"


# ==========================================
# КОНФИГУРАЦИЯ JEPA
# ==========================================
class JEPAConfig:
    """Специфические параметры JEPA."""
    ema_tau = 0.996
    mse_weight = 10.0
    var_weight = 1.0
    cov_weight = 0.1
    vicreg_gamma = 1.0
    num_mask_blocks = 2
    block_size_range = (0.15, 0.25)
    mask_token_init_std = 0.02


# ==========================================
# КОНФИГУРАЦИЯ ДАТАСЕТА
# ==========================================
class DatasetConfig:
    """Параметры данных."""
    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_config = "sample-10BT"
    dataset_split = "train"
    model_path = "Qwen/Qwen3-Embedding-0.6B"
    min_text_length = 10
    max_text_length = 1024
    text_column = "text"
    total_examples = 0
    parquet_files_pattern = "sample/100BT/{i:03d}_00000.parquet"


# ==========================================
# КОНФИГУРАЦИЯ ТЕСТИРОВАНИЯ
# ==========================================
class TestConfig:
    """Параметры тестирования."""
    checkpoint_dir = "./checkpoints"
    checkpoint_pattern = "jepa_shard_*.pt"
    output_dir = "./test_results"
    paraphrase_threshold = 0.70
    noise_robustness_threshold = 0.75
    fine_grained_close_threshold = 0.65
    fine_grained_far_threshold = 0.55
    clustering_accuracy_threshold = 0.80


# ==========================================
# КОНФИГУРАЦИЯ BOSS FIGHT
# ==========================================
class BossFightConfig:
    """Параметры бенчмарка."""
    checkpoint_dir = "./checkpoints"
    checkpoint_pattern = "jepa_shard_*.pt"
    test1_antonym_threshold = 0.60
    test2_analogy_threshold = 0.55
    test4_negation_threshold = 0.75
    test5_bug_threshold = 0.65


# ==========================================
# ОБЩИЙ КОНФИГ (legacy)
# ==========================================
class Config:
    """Единый конфигурационный класс."""
    input_dim = ModelConfig.input_dim
    hidden_dim = ModelConfig.hidden_dim
    embed_dim = ModelConfig.embed_dim
    num_layers = ModelConfig.num_layers
    nhead = ModelConfig.nhead
    max_seq_len = ModelConfig.max_seq_len
    dropout = ModelConfig.dropout

    batch_size = TrainConfig.batch_size
    learning_rate = TrainConfig.learning_rate
    total_steps = TrainConfig.total_steps
    warmup_steps = TrainConfig.warmup_steps
    weight_decay = TrainConfig.weight_decay
    max_grad_norm = TrainConfig.max_grad_norm
    save_interval = TrainConfig.save_interval
    eval_interval = TrainConfig.eval_interval
    log_interval = TrainConfig.log_interval
    checkpoint_dir = TrainConfig.checkpoint_dir
    shards_dir = TrainConfig.shards_dir
    steps_per_shard = TrainConfig.steps_per_shard
    parquet_batch_size = TrainConfig.parquet_batch_size
    shard_size = TrainConfig.shard_size
    num_files_to_process = TrainConfig.num_files_to_process
    examples_per_file = TrainConfig.examples_per_file
    delete_parquet_after = TrainConfig.delete_parquet_after
    file_process_mode = TrainConfig.file_process_mode
    use_amp = TrainConfig.use_amp
    amp_dtype = TrainConfig.amp_dtype
    num_workers = TrainConfig.num_workers
    pin_memory = TrainConfig.pin_memory
    drop_last = TrainConfig.drop_last
    embedding_backend = TrainConfig.embedding_backend
    embedding_max_length = TrainConfig.embedding_max_length
    embedding_encode_batch = TrainConfig.embedding_encode_batch

    ema_tau = JEPAConfig.ema_tau
    mse_weight = JEPAConfig.mse_weight
    var_weight = JEPAConfig.var_weight
    cov_weight = JEPAConfig.cov_weight
    vicreg_gamma = JEPAConfig.vicreg_gamma
    num_mask_blocks = JEPAConfig.num_mask_blocks
    block_size_range = JEPAConfig.block_size_range
    mask_token_init_std = JEPAConfig.mask_token_init_std

    model_path = DatasetConfig.model_path
    dataset_name = DatasetConfig.dataset_name
    dataset_config = DatasetConfig.dataset_config
    min_text_length = DatasetConfig.min_text_length
    max_text_length = DatasetConfig.max_text_length
    text_column = DatasetConfig.text_column
    total_examples = DatasetConfig.total_examples
    parquet_files_pattern = DatasetConfig.parquet_files_pattern

    paraphrase_threshold = TestConfig.paraphrase_threshold
    noise_robustness_threshold = TestConfig.noise_robustness_threshold
    fine_grained_close_threshold = TestConfig.fine_grained_close_threshold
    fine_grained_far_threshold = TestConfig.fine_grained_far_threshold
    clustering_accuracy_threshold = TestConfig.clustering_accuracy_threshold

    boss_test1_threshold = BossFightConfig.test1_antonym_threshold
    boss_test2_threshold = BossFightConfig.test2_analogy_threshold
    boss_test4_threshold = BossFightConfig.test4_negation_threshold
    boss_test5_threshold = BossFightConfig.test5_bug_threshold


def get_amp_dtype():
    if Config.amp_dtype == "bfloat16":
        return torch.bfloat16
    elif Config.amp_dtype == "float16":
        return torch.float16
    return torch.float32


def get_parquet_filenames(num_files):
    return [Config.parquet_files_pattern.format(i=i) for i in range(num_files)]


def print_config():
    print("\n" + "=" * 60)
    print("📋 КОНФИГУРАЦИЯ TEXT-JEPA")
    print("=" * 60)
    configs = [
        ("🧠 МОДЕЛЬ", ModelConfig),
        ("🚀 ОБУЧЕНИЕ", TrainConfig),
        ("🎭 JEPA", JEPAConfig),
        ("📊 ДАТАСЕТ", DatasetConfig),
        ("🧪 ТЕСТ", TestConfig),
        ("⚔️  BOSS FIGHT", BossFightConfig),
    ]
    for name, cfg_class in configs:
        print(f"\n{name}:")
        for key, value in vars(cfg_class).items():
            if not key.startswith('_'):
                print(f"   {key}: {value}")
    print(f"\n💻 УСТРОЙСТВО:")
    print(f"   device: {DEVICE}")
    print(f"   num_cores: {NUM_CORES}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_config()

