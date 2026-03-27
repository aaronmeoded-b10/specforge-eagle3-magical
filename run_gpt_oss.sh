#!/bin/bash

pip install -q uv
apt update -y
apt install -y git curl

#git clone -b orpheus_head https://REDACTED@github.com/basetenlabs/model-training-SpecForge.git
#mv train_configs.sh model-training-SpecForge/examples/train_configs.sh

#SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#ROOT_DIR=$(dirname $SCRIPT_DIR)
#export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

cd model-training-SpecForge

uv venv -p 3.11
source .venv/bin/activate
uv pip install -v .
uv pip install torch-c-dlpack-ext
uv pip install vllm
uv pip install datasets huggingface_hub
uv pip install ninja  # Speed up compilation

# Set up CUDA environment
export CUDA_HOME=/opt/conda
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:$CPATH

# Install flash-attn from pre-built wheel (much faster and more reliable)
# For PyTorch 2.8+ and CUDA 12.x
echo "Installing flash-attn..."

# Verify CUDA extensions are available

echo "⚠️  Pre-built wheel missing CUDA extensions. Building from source..."

# Set build environment variables
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
# export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="8.0;9.0"  # Limit architectures to speed up build

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention

# Build and install into uv venv
uv pip install -v . --no-build-isolation

cd ..
#rm -rf flash-attention

set -e

############################################
# Configuration
############################################
# Base model (HF repo)
MODEL_NAME="openai/gpt-oss-20b"

# Eagle head checkpoint (HF repo) - set to empty if training from scratch
EAGLE_HEAD_CHECKPOINT=""

# Dataset (HF dataset repo)
CUSTOM_DATASET="baseten-admin/orpheus-v1-english-preview-1712-high-quality-english-sentences:train"

# Optional eval dataset (update/remove if your dataset doesn't have this split)
EVAL_DATASET="baseten-admin/orpheus-v1-english-preview-1712-high-quality-english-sentences:test"

# Training hyperparameters
BATCH_SIZE=1
LEARNING_RATE=1e-4
NUM_EPOCHS=3
MAX_LENGTH=4096
TTT_LENGTH=7
SAVE_INTERVAL=50
EVAL_INTERVAL=500
LOG_INTERVAL=50


# Output directory
OUTPUT_DIR=./output/gpt-oss-20b-eagle3-custom

# HuggingFace Hub upload settings
HF_REPO_ID=baseten-admin/gpt-oss-20b-eagle3  # Change to your desired repo
UPLOAD_TO_HF=true  # Set to false to disable upload

# Other settings
SEED=0
TP_SIZE=2  # GPT-OSS-20B requires tensor parallelism
ATTENTION_BACKEND="fa"  # Use flash attention (fa) for draft model
NUM_GPUS=8

############################################
# Train with extended script
############################################

# Ensure venv is active
source .venv/bin/activate

echo "============================================"
echo "🏋️  Starting GPT-OSS-20B Eagle3 Training"
echo "Model: $MODEL_NAME"
echo "Eagle Head: $EAGLE_HEAD_CHECKPOINT"
echo "Dataset: $CUSTOM_DATASET"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Verify we're in the venv and flash-attn is available
echo "Python: $(which python)"
python -c "from flash_attn import flash_attn_func; print('✅ flash-attn verified')"

# Use torchrun for multi-GPU training
TRAIN_ARGS="--target-model-path $MODEL_NAME \
    --train-data-path $CUSTOM_DATASET \
    --eval-data-path $EVAL_DATASET \
    --is-prompt-output \
    --draft-model-config /workspace/model-training-SpecForge/configs/gpt-oss-20B-eagle3.json \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --num-epochs $NUM_EPOCHS \
    --max-length $MAX_LENGTH \
    --ttt-length $TTT_LENGTH \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --log-interval $LOG_INTERVAL \
    --seed $SEED \
    --tp-size $TP_SIZE \
    --attention-backend $ATTENTION_BACKEND \
    --report-to wandb \
    --cache-dir ./cache \
    --hf-repo-id $HF_REPO_ID \
    --target-model-backend sglang \
    --sglang-attention-backend fa3 \
    --dist-timeout 60"

# Add Eagle head checkpoint if specified
if [ -n "$EAGLE_HEAD_CHECKPOINT" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --eagle-head-hf-checkpoint \"$EAGLE_HEAD_CHECKPOINT\""
fi

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 scripts/train_eagle3_extended.py $TRAIN_ARGS

echo "✅ Training completed!"
echo "Model saved to: $OUTPUT_DIR"
if [ "$UPLOAD_TO_HF" = true ] && [ -n "$HF_REPO_ID" ]; then
    echo "Model will be uploaded to: https://huggingface.co/$HF_REPO_ID"
fi

