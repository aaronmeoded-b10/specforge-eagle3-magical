# SpecForge Extensions

This repository extends the upstream [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge) with additional features.

## Custom Files (Safe from upstream conflicts)

These files add new functionality without modifying core SpecForge files:

- **`scripts/train_eagle3_extended.py`** - Extended training script with:
  - `--is-prompt-output`: Train on prompt/output datasets without chat templates
  - `--eagle-head-hf-checkpoint`: Load pre-trained Eagle heads from HuggingFace
  - `--init-backbone-from-layer`: Initialize backbone from target model layer
  - Support for HuggingFace Hub datasets

- **`specforge/data/prompt_output.py`** - Preprocessing for raw prompt/output format

- **`specforge/checkpoint_utils.py`** - Utilities for loading custom checkpoints

- **`specforge/data/__init__.py`** - (if modified) Export new preprocessing functions

- **`benchmarks/benchmarker/raw_completion.py`** - Benchmarker without chat templates

## Syncing with Upstream

Since this repo has unrelated history with upstream, use this workflow:

### Option 1: Manual Sync (Recommended)
```bash
# 1. Check what changed upstream
git fetch upstream
git log upstream/main --oneline -10

# 2. If important changes in core files, manually review and apply:
git diff upstream/main:scripts/train_eagle3.py main:scripts/train_eagle3.py

# 3. Your custom files are isolated, so they won't conflict
```

### Option 2: Replace Core Files
```bash
# Replace specific upstream files while keeping your extensions
git fetch upstream
git checkout upstream/main -- scripts/train_eagle3.py
git checkout upstream/main -- specforge/core/
git checkout upstream/main -- specforge/modeling/

# Your custom files remain untouched:
# - scripts/train_eagle3_extended.py ✓
# - specforge/data/prompt_output.py ✓
# - specforge/checkpoint_utils.py ✓
```

### Option 3: Fresh Start (Nuclear option)
```bash
# 1. Clone fresh from upstream
git clone https://github.com/sgl-project/SpecForge.git SpecForge-fresh
cd SpecForge-fresh

# 2. Copy your custom files
cp /path/to/old/scripts/train_eagle3_extended.py scripts/
cp /path/to/old/specforge/data/prompt_output.py specforge/data/
cp /path/to/old/specforge/checkpoint_utils.py specforge/

# 3. Push to your fork
git remote add origin https://github.com/your-org/SpecForge.git
git add .
git commit -m "Add custom extensions"
git push origin main
```

## Why This Approach Works

✅ **No merge conflicts** - Custom files don't overlap with core files  
✅ **Easy updates** - Pull upstream changes to core files anytime  
✅ **Can contribute back** - Submit PRs for specific features  
✅ **Maintainable** - Clear separation of concerns  

## Using the Extensions

```bash
# Use the extended training script instead of the original
python scripts/train_eagle3_extended.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --train-data-path username/dataset:train \
    --is-prompt-output \
    --eagle-head-hf-checkpoint user/checkpoint \
    --init-backbone-from-layer 15 \
    --output-dir output/
```

All original `train_eagle3.py` arguments still work!
