#!/bin/bash

# Check if model URL is provided
if [ -z "$1" ]; then
    echo "Error: Model URL is required"
    echo "Usage: $0 <model_url>"
    echo "Example: sh $0 https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r-gguf/resolve/main/Llama-xLAM-2-8B-fc-r-Q4_K_M.gguf"
    exit 1
fi

# Activate development environment
source ./.venv/bin/activate

# Start llama.cpp server
echo "Starting llama.cpp server..."

echo "Processing model from: $1"
MODEL_DIR="./linalg_zero/distillation/llama-cpp/models"
# Ensure models directory exists
mkdir -p "${MODEL_DIR}"

# Extract filename from URL
MODEL_NAME=$(basename "$1")
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

# Download model if it doesn't exist locally
if [ ! -f "${MODEL_PATH}" ]; then
    echo "Model not found locally. Downloading..."
    if curl -L -o "${MODEL_PATH}" "$1"; then
        echo "Model successfully downloaded to: ${MODEL_PATH}"
    else
        echo "Error: Failed to download model from $1"
        exit 1
    fi
else
    echo "Model already exists locally: ${MODEL_PATH}"
fi

# Set the model path and alias
ARGS="--model ${MODEL_PATH}"

# Offload all layers to GPU where possible
# The GPU-offload layers must be tuned otherwise we get errors.
echo "Setting GPU-offload layers to: ${2}"
ARGS="${ARGS} --n_gpu_layers ${2}"

# Set the split mode to row
# LLAMA_SPLIT_MODE_NONE = 0
# LLAMA_SPLIT_MODE_LAYER = 1
# LLAMA_SPLIT_MODE_ROW = 2
ARGS="${ARGS} --split_mode 2"

# Set the context limit: 0 = model default, any other value = custom
ARGS="${ARGS} --n_ctx 2048"

# Set the host and port
ARGS="${ARGS} --host 0.0.0.0 --port 8000"

# Start the server
uv run python3 -m llama_cpp.server $ARGS
sleep 10

echo "Keeping the container running..."
tail -f /dev/null
