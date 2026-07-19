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
# 0.5. ФИКС NUMPY/PYARROW (контейнер 24.08 скомпилирован с numpy 1.x)
# ==========================================
echo ""
echo "🔧 Фикс NumPy (даунгрейд до <2.0 для совместимости с PyTorch)..."

CURRENT_NUMPY=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "none")
echo "   Текущий NumPy: $CURRENT_NUMPY"

pip install --no-cache-dir "numpy<2.0" "pyarrow<16.2.0"

echo "   ✅ NumPy/PyArrow зафиксированы на совместимых версиях"

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

# Основные пакеты (PyTorch/CUDA уже в контейнере nvcr.io)
# ONNX Runtime больше НЕ ставится: текущий пайплайн (train_streaming.py, test_jepa.py)
# использует чистый PyTorch-инференс через sentence-transformers, onnxruntime
# нигде в коде не импортируется.
pip install --no-cache-dir \
    sentence-transformers \
    transformers \
    accelerate \
    huggingface-hub \
    scipy

# FlagEmbedding — опционально, нужен только если Config.embedding_backend
# выставлен в "flag_m3" (или "auto" + имя модели содержит "bge-m3").
# Код в train_streaming.py сам делает fallback на sentence-transformers,
# если пакет отсутствует, поэтому падение установки не критично.
pip install --no-cache-dir FlagEmbedding || echo "⚠️  FlagEmbedding не удалось установить, будет fallback на sentence-transformers"

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
echo "  - В config.py можно поднять TrainConfig.batch_size (напр. 128 → 256-384)"
echo "  - TrainConfig.embedding_encode_batch тоже можно увеличить (напр. 64 → 128)"
echo "  - amp_dtype='bfloat16' полностью поддерживается на Ampere, менять не нужно"
echo "  - При увеличении batch_size соразмерно поднимите warmup_steps"
echo ""
echo "🚀 Команды для запуска:"
echo ""
echo "  cd $WORKDIR"
echo ""
echo "  # Обучение (по умолчанию: 10 файлов, 5000 примеров, 500 шагов/шард)"
echo "  python train_streaming.py"
echo ""
echo "  # Обучение с кастомными параметрами"
echo "  python train_streaming.py --num-files 20 --examples-per-file 10000 --steps-per-shard 1000"
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
