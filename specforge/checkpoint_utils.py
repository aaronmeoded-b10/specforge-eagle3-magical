"""
Utilities for loading Eagle head checkpoints from HuggingFace.
"""

import glob
import os
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


def load_eagle_head_from_hf(
    draft_model,
    hf_checkpoint: str,
    cache_dir: str = None,
):
    """
    Load Eagle head weights from HuggingFace Hub or local path.
    
    Args:
        draft_model: The draft model to load weights into.
        hf_checkpoint: HuggingFace Hub repo ID or local path.
        cache_dir: Directory to cache downloaded models.
    """
    print(f"Loading Eagle head from HuggingFace hub or local path: {hf_checkpoint}")
    
    # Download from HF Hub
    repo_dir = snapshot_download(
        repo_id=hf_checkpoint,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors"],
    )
    
    # Find all safetensors files (HF models may be sharded)
    weight_files = sorted(glob.glob(os.path.join(repo_dir, "*.safetensors")))
    if not weight_files:
        raise RuntimeError(f"No .safetensors files found in {repo_dir}")
    
    print(f"Found {len(weight_files)} safetensors file(s). Loading...")
    
    # Merge all shards into one state dict
    hf_state = {}
    for wf in weight_files:
        print(f"Loading: {wf}")
        hf_state.update(load_file(wf))
    
    # Detect whether keys are prefixed with "draft_model."
    has_prefix = any(k.startswith("draft_model.") for k in hf_state.keys())
    
    filtered_state = {}
    for k, v in hf_state.items():
        # Ignore embeddings
        if "embed" in k.lower():
            continue
        
        if has_prefix:
            # Expect format: draft_model.midlayer.self_attn.q_proj.weight
            if k.startswith("draft_model."):
                filtered_state[k[len("draft_model."):]] = v  # strip prefix
        else:
            # Expect format: midlayer.self_attn.q_proj.weight
            filtered_state[k] = v
    
    if not filtered_state:
        raise RuntimeError(
            "Filtered state dict is empty. "
            "Likely mismatch between checkpoint key format and loader assumptions."
        )
    
    missing, unexpected = draft_model.load_state_dict(filtered_state, strict=False)
    
    print("✅ Eagle head loaded.")
    print(f"Missing: {missing}")
    print(f"Unexpected: {unexpected}")


def initialize_backbone_from_target_layer(
    draft_model,
    target_model_path: str,
    layer_index: int,
    cache_dir: str = None,
):
    """
    Initialize draft model backbone from a specific layer of the target model.
    
    Args:
        draft_model: The draft model to initialize.
        target_model_path: Path to the target model.
        layer_index: Which layer to copy from the target model.
        cache_dir: Directory to cache downloaded models.
    """
    print("=" * 80)
    print(f"🔧 BACKBONE INITIALIZATION STARTING")
    print(f"   Target layer: {layer_index}")
    print(f"   Target model: {target_model_path}")
    print("=" * 80)
    
    try:
        from safetensors import safe_open
        
        # Download model files without loading the full model
        print(f"📥 Downloading model files from {target_model_path}...")
        model_dir = snapshot_download(
            repo_id=target_model_path,
            cache_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.json"],
        )
        
        # Find safetensors files
        weight_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not weight_files:
            raise RuntimeError(f"No safetensors files found in {model_dir}")
        
        print(f"📂 Found {len(weight_files)} weight file(s): {[os.path.basename(f) for f in weight_files]}")
        
        # Load only the specific layer weights
        target_layer_state = {}
        layer_prefix = f"model.layers.{layer_index}."
        
        print(f"🔍 Looking for keys with prefix: {layer_prefix}")
        for weight_file in weight_files:
            with safe_open(weight_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith(layer_prefix):
                        # Remove the layer prefix to get relative key
                        relative_key = key[len(layer_prefix):]
                        target_layer_state[relative_key] = f.get_tensor(key)
                        print(f"   Found: {key} -> {relative_key}")
        
        if not target_layer_state:
            print(f"❌ No weights found for layer {layer_index}")
            print("Continuing with random initialization...")
            return
        
        print(f"✅ Loaded {len(target_layer_state)} tensors from layer {layer_index}")
        print(f"   Keys: {list(target_layer_state.keys())}")
        
        # Load into draft model's midlayer
        draft_midlayer_state = draft_model.midlayer.state_dict()
        
        print(f"Draft model keys: {list(draft_midlayer_state.keys())[:5]}...")
        print(f"Target layer keys: {list(target_layer_state.keys())[:5]}...")
        
        # Try to match keys intelligently
        matched_keys = 0
        mismatched_keys = []
        weights_to_load = {}
        
        for draft_key in draft_midlayer_state.keys():
            # Skip attention projection layers - they have different input dimensions
            # Eagle3 uses 2*hidden_size input (concat of embeddings + hidden states)
            # while normal transformers use hidden_size
            if any(proj in draft_key for proj in ['q_proj', 'k_proj', 'v_proj']):
                mismatched_keys.append(f"{draft_key} (skipped - dimension mismatch)")
                continue
            
            # Try direct match first
            if draft_key in target_layer_state:
                target_param = target_layer_state[draft_key]
                draft_param = draft_midlayer_state[draft_key]
                
                # Check shape compatibility
                if target_param.shape == draft_param.shape:
                    weights_to_load[draft_key] = target_param.to(torch.bfloat16).cuda()
                    matched_keys += 1
                    print(f"  ✓ Matched {draft_key}: {target_param.shape}")
                else:
                    mismatched_keys.append(f"{draft_key} (shape: {target_param.shape} vs {draft_param.shape})")
            else:
                # Try to find similar key (handle naming differences)
                found = False
                for target_key in target_layer_state.keys():
                    if (draft_key.replace('self_attn', 'attention') == target_key or
                        draft_key.replace('attention', 'self_attn') == target_key or
                        draft_key.split('.')[-1] == target_key.split('.')[-1]):
                        target_param = target_layer_state[target_key]
                        draft_param = draft_midlayer_state[draft_key]
                        
                        if target_param.shape == draft_param.shape:
                            weights_to_load[draft_key] = target_param.to(torch.bfloat16).cuda()
                            matched_keys += 1
                            found = True
                            print(f"  ✓ Matched {draft_key} → {target_key}: {target_param.shape}")
                            break
                        else:
                            mismatched_keys.append(f"{draft_key} → {target_key} (shape: {target_param.shape} vs {draft_param.shape})")
                            found = True
                            break
                if not found:
                    mismatched_keys.append(f"{draft_key} (no match found)")
        
        # Load the matched weights into the model
        if weights_to_load:
            draft_model.midlayer.load_state_dict(weights_to_load, strict=False)
        
        print(f"✅ Initialized backbone from target layer {layer_index}")
        print(f"   Matched {matched_keys}/{len(draft_midlayer_state)} parameters")
        if mismatched_keys:
            print(f"   ⚠️  Could not match: {mismatched_keys[:5]}{'...' if len(mismatched_keys) > 5 else ''}")
        
        # Clean up
        del target_layer_state
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"❌ Error initializing backbone from target layer: {e}")
        print("Continuing with random initialization...")
        import traceback
        traceback.print_exc()
