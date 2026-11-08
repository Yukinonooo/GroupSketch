#!/usr/bin/env bash
# 本机 GroupSketch 环境；只设置本进程，不改共享 conda 环境。
export GROUPSKETCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GROUPSKETCH_PYTHON_ENV="${GROUPSKETCH_PYTHON_ENV:-/data/anaconda3/envs/video_svg}"
export PATH="$GROUPSKETCH_PYTHON_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$GROUPSKETCH_ROOT/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export WANDB_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export TMPDIR="$GROUPSKETCH_ROOT/tmp/runtime"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$WANDB_DIR"
