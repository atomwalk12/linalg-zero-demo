#!/bin/bash

export HF_HOME=/workspace/linalg-zero/hf-cache
export HUGGINGFACE_HUB_CACHE=/workspace/linalg-zero/hf-cache
export VLLM_WORKDIR=/workspace/linalg-zero/vllm-cache
export UV_CACHE_DIR=/workspace/linalg-zero/uv-cache
export XDG_CACHE_HOME=/workspace/linalg-zero/.cache
export PIP_CACHE_DIR=/workspace/linalg-zero/pip-cache
export USING_VLLM=true
export INFERENCE_BACKEND=vllm
export ARGILLA_API_URL=https://atomwalk12-linalgzero-distilled.hf.space
export ARGILLA_API_KEY=<token>
export HF_TOKEN=<token>
