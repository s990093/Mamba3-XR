#!/bin/bash

# --- Mamba-3-XR Inference Helper Script ---

# 1. Configuration
VENV_PYTHON="/Users/hungwei/Desktop/Proj/Mamba3-XR/.venv/bin/python3"
INFERENCE_SCRIPT="inference.py"
DEFAULT_CKPT="/Users/hungwei/Downloads/mamba3_colab_checkpoint.pt"
DEFAULT_SPM="data/spm_tokenizer.model"
DEFAULT_PROMPT="There are several spaces inside and around the brain filled with fluid, called cerebrospinal fluid (CSF). CSF is a clear fluid that surrounds and cushions the brain and spinal cord. CSF is constantly being produced"
DEFAULT_STEPS=512
DEFAULT_TOP_K=40
DEFAULT_TOP_P=0.9
DEFAULT_TEMP=0.7
DEFAULT_REP_PENALTY=1.1
DEFAULT_PRES_PENALTY=0.0
DEFAULT_FREQ_PENALTY=0.0

# 2. Get arguments or use defaults
CKPT=${1:-$DEFAULT_CKPT}
PROMPT=${2:-$DEFAULT_PROMPT}
STEPS=${3:-$DEFAULT_STEPS}
TOP_K=${4:-$DEFAULT_TOP_K}
TOP_P=${5:-$DEFAULT_TOP_P}
TEMP=${6:-$DEFAULT_TEMP}

# 3. Execute
echo "🚀 Starting Mamba-3-XR Inference..."
echo "📍 Checkpoint: $CKPT"
echo "📝 Prompt: '$PROMPT'"
echo "🔢 Steps: $STEPS | Top-K: $TOP_K | Top-P: $TOP_P | Temp: $TEMP"
echo "⚖️  Penalties: Rep: $DEFAULT_REP_PENALTY | Pres: $DEFAULT_PRES_PENALTY | Freq: $DEFAULT_FREQ_PENALTY"
echo "--------------------------------------------------"

$VENV_PYTHON $INFERENCE_SCRIPT \
    --ckpt "$CKPT" \
    --spm "$DEFAULT_SPM" \
    --prompt "$PROMPT" \
    --steps "$STEPS" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --temp "$TEMP" \
    --rep_penalty "$DEFAULT_REP_PENALTY" \
    --pres_penalty "$DEFAULT_PRES_PENALTY" \
    --freq_penalty "$DEFAULT_FREQ_PENALTY"
