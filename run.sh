#!/bin/bash
set -e

echo "=========================================="
echo "🚀 GPU-JEPA Deployment Script"
echo "Container: nvcr.io/nvidia/pytorch:24.08-py3"
echo "=========================================="

# ==========================================
# 0. ПРОВЕРКА ОКРУЖЕНИЯ
# ==========================================
echo ""
echo "🔍 Проверка окружения..."

python --version

# ==========================================
# 0.5. ФИКС NUMPY (контейнер 24.08 скомпилирован с numpy 1.x)
# ==========================================
echo ""
echo "🔧 Фикс NumPy (даунгрейд до <2.0 для совместимости с PyTorch)..."

# Сначала проверим текущую версию numpy
CURRENT_NUMPY=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "none")
echo "   Текущий NumPy: $CURRENT_NUMPY"

# Даунгрейд numpy до совместимой версии
pip install --no-cache-dir "numpy<2.0" "pyarrow<16.2.0"

echo "   ✅ NumPy обновлён до совместимой версии"

# ==========================================
# 1. ОБНОВЛЕНИЕ PIP И БАЗОВЫЕ ЗАВИСИМОСТИ
# ==========================================
echo ""
echo "📦 Обновление pip..."
pip install --no-cache-dir --upgrade pip setuptools wheel

# ==========================================
# 2. УСТАНОВКА ЗАВИСИМОСТЕЙ
# ==========================================
echo ""
echo "📦 Установка зависимостей..."

# Основные пакеты (PyTorch уже в контейнере)
pip install --no-cache-dir \
    sentence-transformers \
    transformers \
    accelerate \
    huggingface-hub \
    scipy

# ONNX Runtime с GPU поддержкой
pip install --no-cache-dir onnxruntime-gpu

# FlagEmbedding (опционально, для BGE-M3)
pip install --no-cache-dir FlagEmbedding || echo "⚠️ FlagEmbedding не удалось установить, будет fallback на sentence-transformers"

# ==========================================
# 3. ПРОВЕРКА УСТАНОВКИ
# ==========================================
echo ""
echo "✅ Проверка установленных пакетов..."

python -c "
import torch
import transformers
import sentence_transformers
import onnxruntime
import pyarrow
import numpy

print(f'✅ torch: {torch.__version__}')
print(f'✅ transformers: {transformers.__version__}')
print(f'✅ sentence-transformers: {sentence_transformers.__version__}')
print(f'✅ onnxruntime: {onnxruntime.__version__}')
print(f'✅ pyarrow: {pyarrow.__version__}')
print(f'✅ numpy: {numpy.__version__}')

# Проверка CUDA для ONNX
providers = onnxruntime.get_available_providers()
print(f'✅ ONNX providers: {providers}')

# Проверка GPU
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️ GPU не доступна, будет использоваться CPU')

# Проверка numpy-тензоров
t = torch.tensor([1.0, 2.0, 3.0])
arr = t.numpy()
print(f'✅ torch->numpy OK: {arr}')
"

# ==========================================
# 4. ОПРЕДЕЛЕНИЕ РАБОЧЕЙ ДИРЕКТОРИИ
# ==========================================
echo ""
echo "📁 Определение рабочей директории..."

# Определяем директорию, где лежит run.sh
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
echo "   Директория скрипта: $SCRIPT_DIR"

# Если мы в ~/gpu_jepa — используем её, иначе /workspace/gpu_jepa
if [ -f "$SCRIPT_DIR/config.py" ]; then
    WORKDIR="$SCRIPT_DIR"
    echo "   ✅ Найдены исходники рядом со скриптом"
else
    WORKDIR="/workspace/gpu_jepa"
    echo "   ⚠️  Исходники не найдены рядом, используем $WORKDIR"
fi

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Создаём поддиректории
mkdir -p "$WORKDIR/checkpoints"
mkdir -p "$WORKDIR/shards"
mkdir -p "$WORKDIR/test_results"

# ==========================================
# 5. ПРОВЕРКА ИСХОДНИКОВ
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
# 6. ПРОВЕРКА СИНТАКСИСА
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
# 7. ТЕСТОВЫЙ ЗАПУСК КОНФИГА
# ==========================================
echo ""
echo "🧪 Тестовый запуск config.py..."

cd "$WORKDIR"
python config.py

# ==========================================
# 8. ИНФОРМАЦИЯ О ЗАПУСКЕ
# ==========================================
echo ""
echo "=========================================="
echo "🎉 Развёртывание завершено!"
echo "=========================================="
echo ""
echo "📂 Рабочая директория: $WORKDIR"
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

