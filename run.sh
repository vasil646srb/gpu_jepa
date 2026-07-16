#!/bin/bash
set -e

echo "============================================================"
echo "🚀 Text-JEPA (gpu_jepa): Автоматическая настройка окружения"
echo "============================================================"

# run.sh лежит внутри самого репозитория — просто переходим в его
# директорию, никуда клонироваться не нужно.
cd "$(dirname "$(readlink -f "$0")")"

MODEL_PATH="./bge-large-en-v1.5-onnx"

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
$SUDO apt-get install -y git python3-pip python3-venv build-essential screen nvtop 2>/dev/null || true

# ==========================================
# Venv
# ==========================================
[ ! -d ".venv" ] && python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# ==========================================
# PyTorch
# RTX 50xx (Blackwell, sm_120) требует CUDA 12.8+.
# cu121/cu124 колёса не содержат нужных kernel'ов для этой архитектуры.
# ==========================================
if [ "$GPU_AVAILABLE" = true ]; then
    echo "📦 Установка PyTorch (CUDA 12.8, для Blackwell/RTX 50xx)..."
    pip install torch
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
pip install numpy transformers huggingface_hub pyarrow

if [ "$GPU_AVAILABLE" = true ]; then
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
    pip install -q onnxruntime-gpu
else
    pip install -q onnxruntime
fi

# ==========================================
# Базовая модель BGE-large (ONNX)
# Та же логика, что и в ensure_base_model_available() из train_streaming.py:
# качаем, только если model.onnx ещё не лежит в MODEL_PATH.
# ==========================================
if [ -d "$MODEL_PATH" ] && ls "$MODEL_PATH"/*.onnx >/dev/null 2>&1; then
    echo "✅ Базовая модель уже есть: $MODEL_PATH"
else
    echo "📥 Базовая модель не найдена, скачиваю Xenova/bge-large-en-v1.5..."
    python3 - "$MODEL_PATH" <<'PYEOF'
import sys, shutil
from pathlib import Path
from huggingface_hub import snapshot_download

model_path = sys.argv[1]
snapshot_download(repo_id="Xenova/bge-large-en-v1.5", local_dir=model_path)

# ONNX-веса у Xenova лежат в подпапке onnx/ — переносим model.onnx в корень,
# т.к. AutoTokenizer/ort.InferenceSession в проекте ждут его именно там.
onnx_subdir = Path(model_path) / "onnx" / "model.onnx"
target = Path(model_path) / "model.onnx"
if onnx_subdir.exists() and not target.exists():
    shutil.move(str(onnx_subdir), str(target))

print(f"✅ Модель скачана в {model_path}")
PYEOF
fi

mkdir -p checkpoints shards

echo ""
echo "============================================================"
echo "✅ ГОТОВО! Окружение настроено в $(pwd)"
echo "============================================================"
echo ""
echo "Запуск обучения (в отдельной screen-сессии, чтобы пережить обрыв связи):"
echo "   screen -S jepa"
echo "   source .venv/bin/activate"
echo "   python train_streaming.py --num-files 50 --examples-per-file 10000 --steps-per-shard 500"
echo ""
echo "Отсоединиться от screen: Ctrl+A, затем D"
echo "Вернуться обратно:       screen -r jepa"
echo ""
echo "Проверка качества модели после обучения:"
echo "   python test_jepa.py"
echo "   python boss_fight.py"
