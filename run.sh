#!/bin/bash
set -e

echo "=========================================="
echo "🚀 GPU-JEPA Deployment Script (RTX 3090)"
echo "Container: nvcr.io/nvidia/pytorch:24.08-py3"
echo "=========================================="

# ==========================================
# 0. ПРОВЕРКА ОКРУЖЕНИЯ
# ==========================================
echo ""
echo "🔍 Проверка окружения..."

python --version

# ==========================================
# 0.5. ОБНОВЛЕНИЕ PYTORCH (требуется для Qwen3 и transformers>=4.51)
# ==========================================
echo ""
echo "🔧 Обновление PyTorch до 2.6+ (требуется для Qwen3 / transformers>=4.51)..."

pip install --no-cache-dir --upgrade "torch>=2.5.0" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu124

# Убираем конфликтующий torch-tensorrt от NGC (не используется в проекте)
pip uninstall -y torch-tensorrt 2>/dev/null || true

echo "   ✅ PyTorch обновлён"

# ==========================================
# 1. ОБНОВЛЕНИЕ PIP
# ==========================================
echo ""
echo "📦 Обновление pip..."
pip install --no-cache-dir --upgrade pip setuptools wheel

# ==========================================
# 2. УСТАНОВКА ЗАВИСИМОСТЕЙ
# ==========================================
echo ""
echo "📦 Установка зависимостей проекта..."

# Основные пакеты
pip install --no-cache-dir --upgrade     transformers     sentence-transformers     accelerate     huggingface-hub     scipy     pyarrow

# FlagEmbedding — опционально, нужен только если Config.embedding_backend
# выставлен в "flag_m3" (или "auto" + имя модели содержит "bge-m3").
# Код в train_streaming.py сам делает fallback на sentence-transformers,
# если пакет отсутствует, поэтому падение установки не критично.
pip install --no-cache-dir FlagEmbedding || echo "⚠️  FlagEmbedding не удалось установить, будет fallback на sentence-transformers"

# optree — убирает FutureWarning от PyTorch 2.6
pip install --no-cache-dir --upgrade "optree>=0.13.0" || echo "⚠️  optree не обновился, warning останется (не критично)"

# ==========================================
# 3. ПРОВЕРКА GPU И АРХИТЕКТУРЫ (Ampere/3090)
# ==========================================
echo ""
echo "🔍 Проверка GPU..."

python -c "
import torch

if not torch.cuda.is_available():
    print('❌ CUDA недоступна! Проверьте драйвер NVIDIA и что контейнер запущен с --gpus all')
    raise SystemExit(1)

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f'✅ GPU: {name}')
print(f'✅ Compute Capability: {major}.{minor}')
print(f'✅ VRAM: {vram_gb:.1f} GB')

if '3090' not in name:
    print(f'⚠️  Обнаружена не RTX 3090 ({name}) — параметры в config.py могут потребовать ручной корректировки')

# Ampere (SM 8.x) поддерживает bf16 нативно — важно для AMP в train_streaming.py
if major >= 8:
    print('✅ Архитектура поддерживает bfloat16 AMP (Ampere+)')
else:
    print('⚠️  GPU старше Ampere — bfloat16 может быть эмулирован или недоступен, используйте float16')
"

# ==========================================
# 4. ПРОВЕРКА ВЕРСИЙ ПАКЕТОВ
# ==========================================
echo ""
echo "✅ Проверка установленных пакетов..."

python -c "
import torch
import transformers
import sentence_transformers
import pyarrow
import numpy

print(f'✅ torch: {torch.__version__}')
print(f'✅ transformers: {transformers.__version__}')
print(f'✅ sentence-transformers: {sentence_transformers.__version__}')
print(f'✅ pyarrow: {pyarrow.__version__}')
print(f'✅ numpy: {numpy.__version__}')

try:
    import FlagEmbedding
    print(f'✅ FlagEmbedding: установлен (backend BGE-M3 доступен)')
except ImportError:
    print('ℹ️  FlagEmbedding не установлен — будет использоваться sentence-transformers backend')

# Проверка numpy<->torch совместимости
t = torch.tensor([1.0, 2.0, 3.0])
arr = t.numpy()
print(f'✅ torch->numpy OK: {arr}')

# Проверка загрузки Qwen3
print('')
print('🧪 Тестовая загрузка Qwen/Qwen3-Embedding-0.6B...')
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device='cuda', trust_remote_code=True)
print(f'✅ Qwen3 загружен, dim={m.get_sentence_embedding_dimension()}')
"

# ==========================================
# 5. ОПРЕДЕЛЕНИЕ РАБОЧЕЙ ДИРЕКТОРИИ
# ==========================================
echo ""
echo "📁 Определение рабочей директории..."

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "   Директория скрипта: $SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/config.py" ]; then
    WORKDIR="$SCRIPT_DIR"
    echo "   ✅ Найдены исходники рядом со скриптом"
else
    WORKDIR="/workspace/gpu_jepa"
    echo "   ⚠️  Исходники не найдены рядом, используем $WORKDIR"
fi

mkdir -p "$WORKDIR"
cd "$WORKDIR"

mkdir -p "$WORKDIR/checkpoints"
mkdir -p "$WORKDIR/shards"
mkdir -p "$WORKDIR/test_results"

# ==========================================
# 6. ПРОВЕРКА ИСХОДНИКОВ
# ==========================================
echo ""
echo "🔍 Проверка исходников в $WORKDIR..."

for file in config.py models.py dataset.py losses.py train_streaming.py test_jepa.py boss_fight.py; do
    if [ -f "$WORKDIR/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file — отсутствует!"
    fi
done

# ==========================================
# 7. ПРОВЕРКА СИНТАКСИСА
# ==========================================
echo ""
echo "🔍 Проверка синтаксиса Python..."

python -m py_compile "$WORKDIR/config.py"
python -m py_compile "$WORKDIR/models.py"
python -m py_compile "$WORKDIR/dataset.py"
python -m py_compile "$WORKDIR/losses.py"
python -m py_compile "$WORKDIR/train_streaming.py"
python -m py_compile "$WORKDIR/test_jepa.py"
python -m py_compile "$WORKDIR/boss_fight.py"

echo "✅ Синтаксис OK"

# ==========================================
# 8. ТЕСТОВЫЙ ЗАПУСК КОНФИГА
# ==========================================
echo ""
echo "🧪 Тестовый запуск config.py..."

cd "$WORKDIR"
python config.py

# ==========================================
# 9. ИНФОРМАЦИЯ О ЗАПУСКЕ
# ==========================================
echo ""
echo "=========================================="
echo "🎉 Развёртывание завершено!"
echo "=========================================="
echo ""
echo "📂 Рабочая директория: $WORKDIR"
echo ""
echo "💡 Рекомендации под RTX 3090 (24 GB VRAM):"
echo "  - batch_size можно поднять до 512-1024 (в config.py)"
echo "  - embedding_encode_batch можно увеличить до 256"
echo "  - hidden_dim/embed_dim можно поднять до 768-1024"
echo "  - num_layers можно увеличить до 10-12"
echo "  - amp_dtype='bfloat16' полностью поддерживается на Ampere"
echo ""
echo "🚀 Команды для запуска:"
echo ""
echo "  cd $WORKDIR"
echo ""
echo "  # Обучение (по умолчанию: 10 файлов, 10000 примеров/шард, 1000 шагов/шард)"
echo "  python train_streaming.py"
echo ""
echo "  # Обучение с кастомными параметрами"
echo "  python train_streaming.py --num-files 20 --steps-per-shard 1000 --mode full"
echo ""
echo "  # Возобновление обучения"
echo "  python train_streaming.py --resume ./checkpoints/jepa_shard_009.pt"
echo ""
echo "  # Тестирование"
echo "  python test_jepa.py"
echo ""
echo "  # Boss Fight (глубинное тестирование)"
echo "  python boss_fight.py"
echo ""
echo "📊 Мониторинг GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "💡 Советы:"
echo "  - Для первого теста используйте --num-files 1 --examples-per-file 100"
echo "  - Проверьте config.py перед запуском для настройки параметров"
echo "  - Логи сохраняются в stdout (можно перенаправить: python train_streaming.py | tee train.log)"
echo "=========================================="

