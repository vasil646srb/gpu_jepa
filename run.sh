#!/bin/bash
set -e

echo "============================================================"
echo "🚀 Text-JEPA: Автоматическая настройка окружения"
echo "============================================================"

# Определение GPU
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    if [ -n "$GPU_INFO" ]; then
        GPU_AVAILABLE=true
        echo "🎮 GPU: $GPU_INFO"
    fi
fi

# Системные пакеты
apt-get update -qq && apt-get install -y -qq git python3-pip python3-venv screen nvtop 2>/dev/null || true

# Клонирование
[ ! -d "jepa_text" ] && git clone https://github.com/vasil646srb/jepa_text.git
cd jepa_text

# Venv
[ ! -d ".venv" ] && python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

# PyTorch
if [ "$GPU_AVAILABLE" = true ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
else
    pip install torch torchvision torchaudio -q
fi

# ML библиотеки
pip install -q numpy transformers pyarrow huggingface_hub sentence-transformers

if [ "$GPU_AVAILABLE" = true ]; then
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
    pip install onnxruntime-gpu -q
else
    pip install onnxruntime -q
fi

# Скачивание модели BGE
if [ ! -f "./bge-small-en-v1.5-onnx-Q/model.onnx" ]; then
    echo "📥 Скачивание BGE..."
    huggingface-cli download Xenova/bge-small-en-v1.5 --local-dir ./bge-small-en-v1.5-onnx-Q
    [ -f "./bge-small-en-v1.5-onnx-Q/onnx/model.onnx" ] && \
        mv ./bge-small-en-v1.5-onnx-Q/onnx/model.onnx ./bge-small-en-v1.5-onnx-Q/model.onnx
fi

echo ""
echo "✅ ГОТОВО! Запуск:"
echo "   screen -S jepa"
echo "   python train_streaming.py --num-files 50 --examples-per-file 10000 --steps-per-shard 1000"
