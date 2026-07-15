#!/bin/bash
set -e

echo "============================================================"
echo "🚀 Text-JEPA (gpu_jepa): Автоматическая настройка окружения"
echo "============================================================"

REPO_URL="https://github.com/vasil646srb/gpu_jepa.git"
REPO_DIR="gpu_jepa"

# ==========================================
# Определение GPU
# ==========================================
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        GPU_AVAILABLE=true
        echo "🎮 GPU: $GPU_INFO"
    fi
fi

# ==========================================
# Системные пакеты (sudo, только если не root)
# ==========================================
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

$SUDO apt-get update -qq
$SUDO apt-get install -y -qq git python3-pip python3-venv build-essential screen nvtop 2>/dev/null || true

# ==========================================
# Клонирование ПРАВИЛЬНОГО репозитория
# ==========================================
if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

# ==========================================
# Venv
# ==========================================
[ ! -d ".venv" ] && python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

# ==========================================
# PyTorch
# RTX 50xx (Blackwell, sm_120) требует CUDA 12.8+.
# cu121/cu124 колёса не содержат нужных kernel'ов для этой архитектуры.
# ==========================================
if [ "$GPU_AVAILABLE" = true ]; then
    echo "📦 Установка PyTorch (CUDA 12.8, для Blackwell/RTX 50xx)..."
    pip install torch --index-url https://download.pytorch.org/whl/cu128 -q
else
    echo "📦 Установка PyTorch (CPU)..."
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q
fi

# Проверка, что CUDA реально видна PyTorch
python3 - <<'PYEOF'
import torch
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Compute capability: {torch.cuda.get_device_capability(0)}")
PYEOF

# ==========================================
# ML-библиотеки (только реально используемые в коде)
# ==========================================
echo "📦 Установка ML-библиотек..."
pip install -q numpy transformers huggingface_hub pyarrow

if [ "$GPU_AVAILABLE" = true ]; then
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
    pip install -q onnxruntime-gpu
else
    pip install -q onnxruntime
fi

# ==========================================
# Базовая модель BGE-large скачивается автоматически
# самим train_streaming.py / test_jepa.py при первом запуске
# (ensure_base_model_available), в правильный путь
# ./bge-large-en-v1.5-onnx — отдельно скачивать в run.sh не нужно.
# ==========================================

mkdir -p checkpoints shards

echo ""
echo "============================================================"
echo "✅ ГОТОВО! Окружение настроено в $(pwd)"
echo "============================================================"
echo ""
echo "Запуск обучения (в отдельной screen-сессии, чтобы пережить обрыв связи):"
echo "   screen -S jepa"
echo "   source .venv/bin/activate"
echo "   python train_streaming.py --num-files 50 --examples-per-file 10000 --steps-per-shard 1000"
echo ""
echo "Отсоединиться от screen: Ctrl+A, затем D"
echo "Вернуться обратно:       screen -r jepa"
echo ""
echo "Проверка качества модели после обучения:"
echo "   python test_jepa.py"
echo "   python boss_fight.py"
