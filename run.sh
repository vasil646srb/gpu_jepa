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
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

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

# Проверяем наличие constraint файла (может быть в более новых контейнерах)
if [ -f "/etc/pip/constraint.txt" ]; then
    echo "⚠️  Обнаружен pip constraint файл. Удаляем ограничения для наших пакетов..."
    # Сохраняем оригинал
    cp /etc/pip/constraint.txt /etc/pip/constraint.txt.bak
    # Удаляем ограничения для пакетов, которые мы хотим обновить
    sed -i '/sentence-transformers/d' /etc/pip/constraint.txt || true
    sed -i '/transformers/d' /etc/pip/constraint.txt || true
    sed -i '/onnxruntime/d' /etc/pip/constraint.txt || true
    sed -i '/huggingface-hub/d' /etc/pip/constraint.txt || true
    sed -i '/numpy/d' /etc/pip/constraint.txt || true
    sed -i '/pyarrow/d' /etc/pip/constraint.txt || true
fi

# Основные пакеты (PyTorch уже в контейнере)
pip install --no-cache-dir \
    sentence-transformers \
    transformers \
    accelerate \
    huggingface-hub \
    pyarrow \
    numpy \
    scipy

# ONNX Runtime с GPU поддержкой (CUDA 12.x)
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
"

# ==========================================
# 4. СОЗДАНИЕ СТРУКТУРЫ ПРОЕКТА
# ==========================================
echo ""
echo "📁 Создание структуры проекта..."

mkdir -p /workspace/gpu_jepa
mkdir -p /workspace/gpu_jepa/checkpoints
mkdir -p /workspace/gpu_jepa/shards
mkdir -p /workspace/gpu_jepa/test_results

cd /workspace/gpu_jepa

# ==========================================
# 5. КОПИРОВАНИЕ ИСХОДНИКОВ (если есть рядом)
# ==========================================
echo ""
echo "📂 Копирование исходников..."

SCRIPT_DIR=$(dirname "$(realpath "$0")")

if [ -d "$SCRIPT_DIR/src" ]; then
    cp -r "$SCRIPT_DIR/src"/* /workspace/gpu_jepa/
    echo "✅ Исходники скопированы из $SCRIPT_DIR/src/"
elif [ -f "$SCRIPT_DIR/config.py" ]; then
    cp "$SCRIPT_DIR"/*.py /workspace/gpu_jepa/
    echo "✅ Исходники скопированы из $SCRIPT_DIR/"
else
    echo "⚠️  Исходники не найдены рядом со скриптом."
    echo "   Ожидается структура:"
    echo "   run.sh"
    echo "   config.py"
    echo "   train_streaming.py"
    echo "   test_jepa.py"
    echo "   boss_fight.py"
    echo "   models.py"
    echo "   dataset.py"
    echo "   losses.py"
fi

# ==========================================
# 6. ПРОВЕРКА ИСХОДНИКОВ
# ==========================================
echo ""
echo "🔍 Проверка исходников..."

for file in config.py models.py dataset.py losses.py train_streaming.py test_jepa.py boss_fight.py; do
    if [ -f "/workspace/gpu_jepa/$file" ]; then
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

python -m py_compile /workspace/gpu_jepa/config.py
python -m py_compile /workspace/gpu_jepa/models.py
python -m py_compile /workspace/gpu_jepa/dataset.py
python -m py_compile /workspace/gpu_jepa/losses.py
python -m py_compile /workspace/gpu_jepa/train_streaming.py
python -m py_compile /workspace/gpu_jepa/test_jepa.py
python -m py_compile /workspace/gpu_jepa/boss_fight.py

echo "✅ Синтаксис OK"

# ==========================================
# 8. ТЕСТОВЫЙ ЗАПУСК КОНФИГА
# ==========================================
echo ""
echo "🧪 Тестовый запуск config.py..."

cd /workspace/gpu_jepa
python config.py

# ==========================================
# 9. ИНФОРМАЦИЯ О ЗАПУСКЕ
# ==========================================
echo ""
echo "=========================================="
echo "🎉 Развёртывание завершено!"
echo "=========================================="
echo ""
echo "📂 Рабочая директория: /workspace/gpu_jepa"
echo ""
echo "🚀 Команды для запуска:"
echo ""
echo "  # Обучение (по умолчанию: 10 файлов, 5000 примеров, 500 шагов/шард)"
echo "  cd /workspace/gpu_jepa"
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

