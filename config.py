"""
Конфигурация Text-JEPA с автоопределением устройства.
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
# 1. АРХИТЕКТУРА МОДЕЛИ (models.py)
# ==========================================
class ModelConfig:
    """Архитектурные параметры JEPA."""
    input_dim = 1024          # BGE-Small embedding dim (или автоопределение)
    hidden_dim = 512            # Внутренняя размерность
    embed_dim = 512             # Выходная размерность JEPA
    num_layers = 8              # Слои Transformer Encoder
    nhead = 8                   # Количество голов внимания
    max_seq_len = 32            # Максимальная длина последовательности
    dropout = 0.1               # Dropout для регуляризации

    # Predictor специфичные
    predictor_num_layers = 4    # Слои в predictor (если отличается от encoder)
    predictor_max_seq_len = 512 # Макс длина для позиционных эмбеддингов predictor'а


# ==========================================
# 2. ОБУЧЕНИЕ (train_streaming.py)
# ==========================================
class TrainConfig:
    """Параметры обучения."""
    # Основные гиперпараметры
    batch_size = 256
    learning_rate = 3e-4
    total_steps = 50000
    warmup_steps = 1000
    weight_decay = 1e-4
    max_grad_norm = 1.0

    # Шарды и потоковая загрузка
    steps_per_shard = 1000      # Шагов обучения на одном шарде
    parquet_batch_size = 2000   # Размер батча для чтения parquet
    shard_size = 10000          # Размер шарда в примерах
    num_files_to_process = 10   # Количество файлов для обработки
    examples_per_file = 10000   # Примеров из одного parquet-файла
    delete_parquet_after = True # Удалять parquet после обработки

    # Пути
    checkpoint_dir = "./checkpoints"
    shards_dir = "./shards"

    # Логирование и сохранение
    save_interval = 1000        # Сохранять чекпоинт каждые N шагов
    eval_interval = 500         # Оценивать каждые N шагов
    log_interval = 50           # Логировать каждые N шагов

    # AMP (Automatic Mixed Precision)
    use_amp = True              # Использовать mixed precision
    amp_dtype = "bfloat16"      # "bfloat16" или "float16"

    # Оптимизатор
    optimizer_type = "adamw"    # adamw, adam, sgd

    # Scheduler
    scheduler_type = "cosine_with_warmup"  # cosine_with_warmup, linear, constant

    # DataLoader
    num_workers = 0             # 0 для совместимости, 4+ для CPU-only
    pin_memory = True           # pin_memory в DataLoader (True для GPU)
    drop_last = True            # drop_last в DataLoader

    # Embedding engine
    embedding_backend = "auto"  # auto, flag_m3, sentence_transformers
    embedding_max_length = 256  # Максимальная длина для токенизатора embedding модели
    embedding_encode_batch = 64 # Батч для encode в EmbeddingEngine


# ==========================================
# 3. JEPA СПЕЦИФИКА (losses.py + train_streaming.py)
# ==========================================
class JEPAConfig:
    """Специфические параметры JEPA."""
    # EMA для target encoder
    ema_tau = 0.996             # Базовое значение tau для EMA
    ema_tau_final = 1.0         # Финальное значение tau (для schedule)

    # Маскирование
    num_mask_blocks = 2         # Количество маскируемых блоков
    block_size_range = (0.15, 0.25)  # (min, max) доля от seq_len для блока

    # VICReg loss веса
    mse_weight = 10.0
    var_weight = 1.0
    cov_weight = 0.1
    vicreg_gamma = 1.0          # Множитель для variance loss

    # Маск-токен
    mask_token_init_std = 0.02  # Стандартное отклонение для инициализации маск-токена


# ==========================================
# 4. ДАТАСЕТ (dataset.py + train_streaming.py)
# ==========================================
class DatasetConfig:
    """Параметры данных."""
    # Источник данных
    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_config = "sample-10BT"   # Конфигурация датасета
    dataset_split = "train"            # Сплит для загрузки

    # Текстовые параметры
    min_text_length = 10             # Минимальная длина текста в символах
    max_text_length = 1024           # Максимальная длина текста для токенизации
    text_column = "text"               # Название колонки с текстом

    # Embedding модель
    model_path = "Qwen/Qwen3-Embedding-0.6B"

    # Локальные данные
    local_data_path = None           # Путь к локальным данным (если не HF dataset)

    # Фильтрация
    total_examples = 0               # 0 = без ограничений

    # Параметры parquet
    parquet_column = "text"          # Колонка для чтения из parquet
    parquet_files_pattern = "sample/100BT/{i:03d}_00000.parquet"


# ==========================================
# 5. ТЕСТИРОВАНИЕ (test_jepa.py)
# ==========================================
class TestConfig:
    """Параметры тестирования и оценки."""
    # Пути
    checkpoint_dir = "./checkpoints"           # Директория с чекпоинтами
    checkpoint_pattern = "jepa_shard_*.pt"       # Паттерн для поиска чекпоинтов
    output_dir = "./test_results"

    # Базовая модель (ONNX)
    onnx_model_path = "./bge-large-en-v1.5-onnx"
    onnx_repo_id = "vasil646/jepa_text"        # HF репо для скачивания
    onnx_fallback_repo = "Xenova/bge-large-en-v1.5"

    # Параметры ONNX Runtime
    onnx_graph_opt_level = "all"               # all, extended, basic, none
    onnx_cuda_threads = 1
    onnx_cpu_threads = NUM_CORES
    onnx_execution_mode = "sequential"         # sequential, parallel

    # Параметры токенизации
    tokenizer_max_length = 64                  # max_length для токенизатора в тесте
    tokenizer_padding = True
    tokenizer_truncation = True

    # Параметры оценки
    test_batch_size = 128
    num_test_batches = 100                     # 0 = весь датасет

    # Пороги тестов
    paraphrase_threshold = 0.70
    noise_robustness_threshold = 0.75
    fine_grained_close_threshold = 0.65
    fine_grained_far_threshold = 0.55
    clustering_accuracy_threshold = 0.80

    # Визуализация
    num_visualization_samples = 10
    save_attention_maps = False

    # Метрики
    compute_perplexity = True
    compute_similarity = True
    compute_reconstruction_error = True


# ==========================================
# 6. BOSS FIGHT (boss_fight.py)
# ==========================================
class BossFightConfig:
    """Параметры для бенчмарка/соревнования."""
    # Пути
    checkpoint_dir = "./checkpoints"
    checkpoint_pattern = "jepa_shard_*.pt"

    # Пороги тестов
    test1_antonym_threshold = 0.60
    test2_analogy_threshold = 0.55
    test4_negation_threshold = 0.75
    test5_bug_threshold = 0.65

    # Embedding модель
    model_path = DatasetConfig.model_path


# ==========================================
# ОБЩИЙ КОНФИГ (legacy, для обратной совместимости)
# ==========================================
class Config:
    """
    Единый конфигурационный класс (legacy).
    Собирает все подконфиги для удобства импорта.
    Используется во всём проекте как единый источник правды.
    """
    # --- ModelConfig ---
    input_dim = ModelConfig.input_dim
    hidden_dim = ModelConfig.hidden_dim
    embed_dim = ModelConfig.embed_dim
    num_layers = ModelConfig.num_layers
    nhead = ModelConfig.nhead
    max_seq_len = ModelConfig.max_seq_len
    dropout = ModelConfig.dropout

    # --- TrainConfig ---
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
    use_amp = TrainConfig.use_amp
    amp_dtype = TrainConfig.amp_dtype
    num_workers = TrainConfig.num_workers
    pin_memory = TrainConfig.pin_memory
    drop_last = TrainConfig.drop_last
    embedding_backend = TrainConfig.embedding_backend
    embedding_max_length = TrainConfig.embedding_max_length
    embedding_encode_batch = TrainConfig.embedding_encode_batch

    # --- JEPAConfig ---
    ema_tau = JEPAConfig.ema_tau
    mse_weight = JEPAConfig.mse_weight
    var_weight = JEPAConfig.var_weight
    cov_weight = JEPAConfig.cov_weight
    vicreg_gamma = JEPAConfig.vicreg_gamma
    num_mask_blocks = JEPAConfig.num_mask_blocks
    block_size_range = JEPAConfig.block_size_range
    mask_token_init_std = JEPAConfig.mask_token_init_std

    # --- DatasetConfig ---
    model_path = DatasetConfig.model_path
    dataset_name = DatasetConfig.dataset_name
    dataset_config = DatasetConfig.dataset_config
    dataset_split = DatasetConfig.dataset_split
    min_text_length = DatasetConfig.min_text_length
    max_text_length = DatasetConfig.max_text_length
    text_column = DatasetConfig.text_column
    local_data_path = DatasetConfig.local_data_path
    total_examples = DatasetConfig.total_examples
    parquet_column = DatasetConfig.parquet_column
    parquet_files_pattern = DatasetConfig.parquet_files_pattern

    # --- TestConfig ---
    test_checkpoint_dir = TestConfig.checkpoint_dir
    test_checkpoint_pattern = TestConfig.checkpoint_pattern
    test_output_dir = TestConfig.output_dir
    onnx_model_path = TestConfig.onnx_model_path
    onnx_repo_id = TestConfig.onnx_repo_id
    onnx_fallback_repo = TestConfig.onnx_fallback_repo
    tokenizer_max_length = TestConfig.tokenizer_max_length
    paraphrase_threshold = TestConfig.paraphrase_threshold
    noise_robustness_threshold = TestConfig.noise_robustness_threshold
    fine_grained_close_threshold = TestConfig.fine_grained_close_threshold
    fine_grained_far_threshold = TestConfig.fine_grained_far_threshold
    clustering_accuracy_threshold = TestConfig.clustering_accuracy_threshold

    # --- BossFightConfig ---
    boss_checkpoint_dir = BossFightConfig.checkpoint_dir
    boss_checkpoint_pattern = BossFightConfig.checkpoint_pattern
    boss_test1_threshold = BossFightConfig.test1_antonym_threshold
    boss_test2_threshold = BossFightConfig.test2_analogy_threshold
    boss_test4_threshold = BossFightConfig.test4_negation_threshold
    boss_test5_threshold = BossFightConfig.test5_bug_threshold


# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def get_amp_dtype():
    """Возвращает torch dtype для AMP на основе конфига."""
    if Config.amp_dtype == "bfloat16":
        return torch.bfloat16
    elif Config.amp_dtype == "float16":
        return torch.float16
    return torch.float32


def get_onnx_graph_opt_level():
    """Возвращает уровень оптимизации ONNX Runtime."""
    import onnxruntime as ort
    level_map = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }
    return level_map.get(TestConfig.onnx_graph_opt_level, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)


def get_onnx_execution_mode():
    """Возвращает режим выполнения ONNX Runtime."""
    import onnxruntime as ort
    if TestConfig.onnx_execution_mode == "parallel":
        return ort.ExecutionMode.ORT_PARALLEL
    return ort.ExecutionMode.ORT_SEQUENTIAL


def get_parquet_filenames(num_files):
    """Генерирует список parquet-файлов для скачивания."""
    return [Config.parquet_files_pattern.format(i=i) for i in range(num_files)]


def print_config():
    """Красивый вывод всей конфигурации."""
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

