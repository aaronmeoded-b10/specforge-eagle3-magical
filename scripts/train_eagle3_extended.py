#!/usr/bin/env python3
"""
Extended training script that supports prompt-output datasets and custom checkpoints.

This script wraps the original train_eagle3.py functionality and adds support for:
- --is-prompt-output: Use prompt/output format without chat template
- --eagle-head-hf-checkpoint: Load pre-trained Eagle head from HuggingFace
- --init-backbone-from-layer: Initialize backbone from target model layer
- HuggingFace Hub datasets: Use "username/dataset" or "username/dataset:split"

Usage:
    # Local JSONL file
    python scripts/train_eagle3_extended.py \
        --target-model-path meta-llama/Llama-3.1-8B-Instruct \
        --train-data-path data.jsonl \
        --is-prompt-output \
        --output-dir output/
    
    # HuggingFace Hub dataset
    python scripts/train_eagle3_extended.py \
        --target-model-path meta-llama/Llama-3.1-8B-Instruct \
        --train-data-path username/my-dataset \
        --is-prompt-output \
        --output-dir output/
    
    # With split specification
    python scripts/train_eagle3_extended.py \
        --target-model-path meta-llama/Llama-3.1-8B-Instruct \
        --train-data-path username/my-dataset:train \
        --eval-data-path username/my-dataset:validation \
        --is-prompt-output \
        --output-dir output/
    
    # All features combined
    python scripts/train_eagle3_extended.py \
        --target-model-path meta-llama/Llama-3.1-8B-Instruct \
        --train-data-path username/my-dataset:train \
        --is-prompt-output \
        --eagle-head-hf-checkpoint user/eagle-checkpoint \
        --init-backbone-from-layer 15 \
        --output-dir output/
"""

import sys
import os

# Add parent directory to path to import from specforge
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the prompt-output preprocessing
from specforge.data.prompt_output import build_prompt_output_dataset
from specforge.checkpoint_utils import (
    load_eagle_head_from_hf,
    initialize_backbone_from_target_layer,
)

# Import original training script components
from scripts.train_eagle3 import (
    parse_args,
    build_tracker,
    build_target_model,
    sanity_check,
    run_forward,
    run_backward_and_update,
    record_metrcs,
    save_checkpoints,
    get_dp_data_shard_from_tp,
    main as original_main,
)

# Import other dependencies
import argparse
import hashlib
import math
import time
from typing import Tuple, Optional
from argparse import ArgumentParser, Namespace

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoProcessor
from tqdm import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from specforge import OnlineEagle3Model, QwenVLOnlineEagle3Model, AutoEagle3DraftModel, AutoDraftModelConfig
from specforge.data import generate_vocab_mapping_file, prepare_dp_dataloaders
from specforge.distributed import get_dp_group
from specforge.optimizer import BF16Optimizer
from specforge.utils import rank_0_priority, print_with_rank, create_draft_config_from_target, get_last_checkpoint

def print_masked_example(batch, tokenizer):
    RED = "\033[91m"
    BLUE = "\033[94m"
    RESET = "\033[0m"

    input_ids = batch["input_ids"][0]
    loss_mask = batch["loss_mask"][0]

    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)

    # Split by mask
    prompt_ids = input_ids[loss_mask == 0]
    output_ids = input_ids[loss_mask == 1]

    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    output_text = tokenizer.decode(output_ids, skip_special_tokens=False)

    print("\n" + "=" * 80)
    print("🧪 RAW TRAINING EXAMPLE (PROMPT → OUTPUT)")
    print("=" * 80)

    print("\n🔵 PROMPT (masked, no loss):\n")
    print(f"{BLUE}{prompt_text}{RESET}")

    print("\n🔴 OUTPUT (trained, contributes to loss):\n")
    print(f"{RED}{output_text}{RESET}")

    print("\n" + "=" * 80 + "\n")



def parse_args_extended() -> Tuple[ArgumentParser, Namespace]:
    """Extended argument parser that adds custom flags.
    
    We modify the original parse_args to add our custom arguments before parsing.
    """
    import argparse
    from scripts.train_eagle3 import parse_args as _original_parse_args
    
    # Monkey-patch the original parse_args to add our arguments
    # We'll temporarily replace ArgumentParser.parse_args to inject our args
    
    original_parse_args_method = argparse.ArgumentParser.parse_args
    our_parser_ref = [None]  # Use list to capture parser reference
    
    def patched_parse_args(self, args=None, namespace=None):
        # Add our custom arguments before parsing
        try:
            self.add_argument(
                "--is-prompt-output",
                action="store_true",
                help="Whether the input data has 'prompt' and 'output' columns.",
            )
        except argparse.ArgumentError:
            pass  # Already added
            
        try:
            self.add_argument(
                "--eagle-head-hf-checkpoint",
                type=str,
                default=None,
                help="HuggingFace Hub repo ID containing a pre-trained Eagle head.",
            )
        except argparse.ArgumentError:
            pass
            
        try:
            self.add_argument(
                "--init-backbone-from-layer",
                type=int,
                default=None,
                help="Initialize draft backbone from this target model layer.",
            )
        except argparse.ArgumentError:
            pass
            
        try:
            self.add_argument(
                "--unfreeze-embeddings",
                action="store_true",
                help="Keep embeddings trainable.",
            )
        except argparse.ArgumentError:
            pass
        
        try:
            self.add_argument(
                "--hf-repo-id",
                type=str,
                default=None,
                help="HuggingFace Hub repository ID to upload trained model (e.g., 'username/model-name')",
            )
        except argparse.ArgumentError:
            pass
        
        our_parser_ref[0] = self
        return original_parse_args_method(self, args, namespace)
    
    # Temporarily patch
    argparse.ArgumentParser.parse_args = patched_parse_args
    
    try:
        parser, args = _original_parse_args()
        return our_parser_ref[0] if our_parser_ref[0] else parser, args
    finally:
        # Restore original
        argparse.ArgumentParser.parse_args = original_parse_args_method


def build_dataloaders_extended(
    args: Namespace,
    draft_model_config,
    processor: Optional[AutoProcessor] = None,
) -> Tuple[DataLoader, str, Optional[DataLoader]]:
    """
    Extended dataloader builder that supports prompt-output format.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    
    # Load dataset - support both local files and HuggingFace datasets
    if os.path.exists(args.train_data_path):
        # Local JSONL file
        print_with_rank(f"Loading local dataset from: {args.train_data_path}")
        train_dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    else:
        # Try loading from HuggingFace Hub
        # Format: "username/dataset" or "username/dataset:split"
        print_with_rank(f"Loading dataset from HuggingFace Hub: {args.train_data_path}")
        if ":" in args.train_data_path:
            dataset_name, split = args.train_data_path.split(":", 1)
            train_dataset = load_dataset(dataset_name, split=split)
        else:
            # Default to "train" split
            train_dataset = load_dataset(args.train_data_path, split="train")
    
    # Remove 'status' column if present (added by regeneration scripts)
    if "status" in train_dataset.column_names:
        print_with_rank("Removing 'status' column from dataset")
        train_dataset = train_dataset.remove_columns(["status"])
    
    # For chatgpt/harmony chat template, convert "assistant" role to "assistant_final"
    if args.chat_template == "chatgpt":
        def convert_assistant_roles(example):
            if "conversations" in example:
                for msg in example["conversations"]:
                    if msg.get("role") == "assistant":
                        msg["role"] = "assistant_final"
            return example
        print_with_rank("Converting 'assistant' roles to 'assistant_final' for chatgpt template")
        train_dataset = train_dataset.map(convert_assistant_roles)
    # Build cache key
    # v2: Added role conversion for chatgpt template
    cache_version = "v2"
    if args.is_prompt_output:
        cache_params_string = (
            f"{cache_version}-"
            f"{args.train_data_path}-"
            f"{args.max_length}-"
            f"{args.target_model_path}-"
            f"prompt_output=True"
        )
    else:
        cache_params_string = (
            f"{cache_version}-"
            f"{args.train_data_path}-"
            f"{args.max_length}-"
            f"{args.chat_template}-"
            f"{args.target_model_path}-"
            f"preformatted={args.is_preformatted}"
        )
    
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()
    
    with rank_0_priority():
        if args.is_prompt_output:
            # Use our custom prompt-output preprocessing
            print_with_rank("Using prompt-output dataset format (no chat template)")
            train_eagle3_dataset = build_prompt_output_dataset(
                dataset=train_dataset,
                tokenizer=tokenizer,
                max_length=args.max_length,
                cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
                cache_key=cache_key,
                num_proc=args.build_dataset_num_proc,
            )
        else:
            # Use original preprocessing
            from specforge.data import build_eagle3_dataset
            train_eagle3_dataset = build_eagle3_dataset(
                dataset=train_dataset,
                tokenizer=tokenizer,
                chat_template=args.chat_template,
                max_length=args.max_length,
                cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
                cache_key=cache_key,
                is_vlm=args.is_vlm,
                is_preformatted=args.is_preformatted,
                processor=processor,
                num_proc=args.build_dataset_num_proc,
            )

        vocab_mapping_path = generate_vocab_mapping_file(
            dataset=train_eagle3_dataset,
            target_vocab_size=draft_model_config.vocab_size,
            draft_vocab_size=draft_model_config.draft_vocab_size,
            cache_dir=os.path.join(args.cache_dir, "vocab_mapping"),
            cache_key=cache_key,
        )
    
    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.target_batch_size,
        num_workers=4,
        shuffle=True,
        process_group=get_dp_group(),
        is_vlm=args.is_vlm,
    )
    
    # Handle eval dataset
    eval_dataloader = None
    if args.eval_data_path is not None:
        # Load eval dataset - support both local files and HuggingFace datasets
        if os.path.exists(args.eval_data_path):
            print_with_rank(f"Loading local eval dataset from: {args.eval_data_path}")
            eval_dataset = load_dataset("json", data_files=args.eval_data_path)["train"]
        else:
            print_with_rank(f"Loading eval dataset from HuggingFace Hub: {args.eval_data_path}")
            if ":" in args.eval_data_path:
                dataset_name, split = args.eval_data_path.split(":", 1)
                eval_dataset = load_dataset(dataset_name, split=split)
            else:
                # Default to "validation" or "test" split for eval
                try:
                    eval_dataset = load_dataset(args.eval_data_path, split="validation")
                except:
                    eval_dataset = load_dataset(args.eval_data_path, split="test")
        
        if args.is_prompt_output:
            eval_eagle3_dataset = build_prompt_output_dataset(
                dataset=eval_dataset,
                tokenizer=tokenizer,
                max_length=args.max_length,
                num_proc=args.build_dataset_num_proc,
            )
        else:
            from specforge.data import build_eagle3_dataset
            eval_eagle3_dataset = build_eagle3_dataset(
                eval_dataset,
                tokenizer,
                args.chat_template,
                args.max_length,
                is_vlm=args.is_vlm,
                processor=processor,
                num_proc=args.build_dataset_num_proc,
                is_preformatted=args.is_preformatted,
            )
        
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.target_batch_size,
            num_workers=4,
            shuffle=False,
            process_group=get_dp_group(),
            is_vlm=args.is_vlm,
        )
        print_with_rank("Initialized eval dataloader")
    
    return train_dataloader, vocab_mapping_path, eval_dataloader


def build_draft_model_extended(args: Namespace):
    """
    Extended draft model builder with custom checkpoint loading support.
    """
    import torch.nn as nn
    
    # Handle draft model config
    if args.draft_model_config is None:
        # Auto-generate and save config file
        auto_config_path = create_draft_config_from_target(
            target_model_path=args.target_model_path, 
            cache_dir=args.model_download_dir
        )
        draft_model_config = AutoDraftModelConfig.from_file(auto_config_path)
    else:
        # Use provided config file
        draft_model_config = AutoDraftModelConfig.from_file(args.draft_model_config)
    
    # Handle base ckpt, config file
    draft_model_last_checkpoint = None
    if args.ckpt_dir is not None:
        if os.path.isdir(args.ckpt_dir):
            draft_model_config = os.path.join(args.ckpt_dir, "config.json")
            draft_model_last_checkpoint = args.ckpt_dir
            print_with_rank(f"Finetuning from base model: {draft_model_last_checkpoint}")
        else:
            raise ValueError(
                f"Provided base model dir {args.ckpt_dir} is not a valid directory."
            )
    
    # Detecting last ckpt for draft model
    if args.resume and os.path.isdir(args.output_dir):
        print_with_rank(args.output_dir)
        draft_model_last_checkpoint = get_last_checkpoint(args.output_dir)
        print_with_rank(f"Last checkpoint detected: {draft_model_last_checkpoint}")
    
    if draft_model_last_checkpoint:
        draft_model = AutoEagle3DraftModel.from_pretrained(
            draft_model_last_checkpoint,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        draft_model = AutoEagle3DraftModel.from_config(
            draft_model_config,
            attention_backend=args.attention_backend,
            torch_dtype=torch.bfloat16,
        ).cuda()
    
    print_with_rank(f"DEBUG: Draft model initialized. args.eagle_head_hf_checkpoint = {args.eagle_head_hf_checkpoint}")
    
    # Load Eagle head checkpoint if specified
    if args.eagle_head_hf_checkpoint is not None:
        load_eagle_head_from_hf(
            draft_model,
            args.eagle_head_hf_checkpoint,
            cache_dir=args.model_download_dir,
        )
    
    # Initialize backbone from target model layer if specified
    print_with_rank(f"DEBUG: Checking backbone init - args.init_backbone_from_layer = {args.init_backbone_from_layer}")
    if args.init_backbone_from_layer is not None:
        initialize_backbone_from_target_layer(
            draft_model,
            args.target_model_path,
            args.init_backbone_from_layer,
            cache_dir=args.model_download_dir,
        )
    
    # Load embeddings from target model
    draft_model.load_embedding(args.target_model_path, embedding_key=args.embedding_key)
    
    if not args.unfreeze_embeddings:
        draft_model.freeze_embedding()
        print_with_rank("✅ Embeddings frozen (using target model embeddings)")
    else:
        print_with_rank("⚠️  Embeddings kept trainable (will adapt during training)")
    
    return draft_model_config, draft_model


def main():
    """Main training function with extended functionality."""
    from accelerate.utils import set_seed
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
    from specforge.distributed import init_distributed, destroy_distributed
    
    # Parse arguments with extension
    parser, args = parse_args_extended()
    set_seed(args.seed)
    
    from specforge.distributed import init_distributed
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    
    is_online = (
        args.train_data_path is not None and args.train_hidden_states_path is None
    )
    
    sanity_check(args)
    print_with_rank("Initialized distributed environment")
    
    # Debug: Check if HF upload is configured
    if hasattr(args, 'hf_repo_id') and args.hf_repo_id:
        print(f"[Config] HuggingFace upload enabled to: {args.hf_repo_id}")
    else:
        print(f"[Config] HuggingFace upload disabled (hf_repo_id: {getattr(args, 'hf_repo_id', 'not set')})")
    
    # Build models with extended functionality
    draft_model_config, draft_model = build_draft_model_extended(args)
    print_with_rank("Initialized draft model")
    target_model, processor = build_target_model(args, draft_model_config, is_online)
    
    # Build dataloaders with extended support
    train_dataloader, vocab_mapping_path, eval_dataloader = build_dataloaders_extended(
        args, draft_model_config, processor
    )
    
    # Load vocab mapping
    draft_model.load_vocab_mapping(vocab_mapping_path)
    print_with_rank("Loaded vocab mapping")

    if dist.get_rank() == 0:
        print_with_rank("📊 Printing one masked training example before training...")
        sample_batch = next(iter(train_dataloader))
        print_masked_example(
            sample_batch,
            tokenizer=AutoTokenizer.from_pretrained(args.target_model_path),
        )
    
    dist.barrier()
    
    # Calculate total steps
    if args.total_steps is None:
        steps_per_epoch = math.ceil(
            len(train_dataloader) / args.draft_accumulation_steps
        )
        args.total_steps = args.num_epochs * steps_per_epoch
        print_with_rank(
            f"Auto-calculated total_steps: {args.total_steps} "
            f"(num_epochs={args.num_epochs} * steps_per_epoch={steps_per_epoch})"
        )
    
    # Build Eagle3 model
    if (
        args.is_vlm
        and getattr(draft_model_config, "target_model_type", None) == "qwen2_5_vl"
    ):
        eagle3_model = QwenVLOnlineEagle3Model(
            target_model=target_model,
            draft_model=draft_model,
            processor=processor,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    else:
        eagle3_model = OnlineEagle3Model(
            draft_model=draft_model,
            length=args.ttt_length,
            attention_backend=args.attention_backend,
        )
    
    eagle3_model = FSDP(
        eagle3_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        process_group=dist.group.WORLD,
    )
    print_with_rank("Initialized Eagle3 FSDP model")
    
    # Build optimizer
    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=args.total_steps,
    )
    print_with_rank("Initialized optimizer and scheduler")
    
    # Build tracker
    tracker = build_tracker(args, parser)
    global_step = 0
    start_epoch = 0
    dist.barrier()
    
    last_time = time.time()
    
    # Training loop
    print_with_rank(f"Starting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch + 1)
        draft_model.train()
        
        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader
        
        for data in progress_bar:
            global_step += 1
            
            # Forward pass
            plosses, acces = run_forward(
                args, eagle3_model, data, target_model, is_online
            )
            run_backward_and_update(args, plosses, optimizer, global_step)
            
            # Logging
            if global_step % args.log_interval == 0:
                record_metrcs(
                    args, acces, plosses, global_step, tracker, optimizer, mode="train"
                )
            
            if dist.get_rank() == 0:
                time_per_step = time.time() - last_time
                last_time = time.time()
                avg_loss = sum(pl for pl in plosses) / len(plosses)
                avg_acc = sum(acces) / len(acces)
                progress_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.2f}",
                        "acc": f"{avg_acc:.2f}",
                        "time": f"{time_per_step:.2f}s",
                    }
                )
            
            # Evaluation
            if eval_dataloader is not None and global_step % args.eval_interval == 0:
                draft_model.eval()
                eval_acces = [[] for _ in range(eagle3_model.length)]
                eval_plosses = [[] for _ in range(eagle3_model.length)]
                
                for data in tqdm(eval_dataloader, desc=f"Evaluating Epoch {epoch}"):
                    with torch.no_grad():
                        plosses, acces = run_forward(
                            args, eagle3_model, data, target_model, is_online
                        )
                        eval_acces = [
                            eval_acces[i] + [acces[i]] for i in range(len(acces))
                        ]
                        eval_plosses = [
                            eval_plosses[i] + [plosses[i]] for i in range(len(plosses))
                        ]
                
                eval_acces = [torch.stack(acc).mean() for acc in eval_acces]
                eval_plosses = [torch.stack(pl).mean() for pl in eval_plosses]
                
                record_metrcs(
                    args,
                    eval_acces,
                    eval_plosses,
                    global_step,
                    tracker,
                    mode="eval",
                )
            
            # Save checkpoints
            if global_step % args.save_interval == 0:
                print_with_rank(f"[Checkpoint] Saving checkpoint at epoch {epoch} step {global_step}")
                save_checkpoints(args, epoch, global_step, eagle3_model, optimizer)
                
                # Upload to HuggingFace Hub after each checkpoint save
                if args.hf_repo_id:
                    if dist.get_rank() == 0:
                        print_with_rank(f"[Upload] Starting HuggingFace upload for epoch {epoch} step {global_step}")
                        try:
                            upload_to_huggingface(args, eagle3_model, epoch=epoch, checkpoint_step=global_step)
                            print_with_rank(f"[Upload] Completed HuggingFace upload for epoch {epoch} step {global_step}")
                        except Exception as e:
                            print_with_rank(f"[Upload] ERROR during upload: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print_with_rank(f"[Upload] Skipping upload on rank {dist.get_rank()}")
                else:
                    print_with_rank(f"[Upload] No HF repo ID configured, skipping upload")
            
            if args.max_num_steps is not None and global_step >= args.max_num_steps:
                break
        
        if args.max_num_steps is not None and global_step >= args.max_num_steps:
            break
    
    # Final upload to HuggingFace Hub if requested
    if args.hf_repo_id and dist.get_rank() == 0:
        upload_to_huggingface(args, eagle3_model, epoch=epoch, checkpoint_step=global_step, final=True)
    
    tracker.close()
    destroy_distributed()


def upload_to_huggingface(args, model, epoch=None, checkpoint_step=None, final=False):
    """Upload the trained model to HuggingFace Hub."""
    from huggingface_hub import HfApi, create_repo
    import glob
    
    step_info = f"epoch-{epoch}-step-{checkpoint_step}" if checkpoint_step else "final"
    print_with_rank("=" * 80)
    print_with_rank(f"📤 Uploading model checkpoint ({step_info}) to HuggingFace Hub: {args.hf_repo_id}")
    print_with_rank("=" * 80)
    
    try:
        # Create repo if it doesn't exist
        create_repo(args.hf_repo_id, exist_ok=True, private=False, repo_type="model")
        print_with_rank(f"✅ Repository {args.hf_repo_id} ready")
        
        # Find the checkpoint to upload
        if checkpoint_step and epoch is not None:
            # Upload specific checkpoint with format: epoch_i_step_j
            checkpoint_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{checkpoint_step}")
            if os.path.exists(checkpoint_dir):
                upload_path = checkpoint_dir
                print_with_rank(f"Uploading checkpoint: {checkpoint_dir}")
            else:
                print_with_rank(f"⚠️  Checkpoint directory not found: {checkpoint_dir}")
                print_with_rank(f"Looking for alternative checkpoint formats...")
                # Try alternative formats
                alt_formats = [
                    os.path.join(args.output_dir, f"checkpoint-{checkpoint_step}"),
                    os.path.join(args.output_dir, f"step_{checkpoint_step}"),
                ]
                for alt_path in alt_formats:
                    if os.path.exists(alt_path):
                        upload_path = alt_path
                        print_with_rank(f"Found checkpoint at: {alt_path}")
                        break
                else:
                    print_with_rank(f"❌ Could not find checkpoint directory")
                    return
        else:
            # Find the latest checkpoint for final upload
            # Look for pattern: epoch_*_step_*
            checkpoints = glob.glob(os.path.join(args.output_dir, "epoch_*_step_*"))
            if checkpoints:
                # Sort by step number
                def extract_step(path):
                    try:
                        return int(path.split("_step_")[-1])
                    except:
                        return 0
                latest_checkpoint = max(checkpoints, key=extract_step)
                upload_path = latest_checkpoint
                print_with_rank(f"Uploading latest checkpoint: {latest_checkpoint}")
            else:
                upload_path = args.output_dir
                print_with_rank(f"Uploading from output directory: {args.output_dir}")
        
        # List files to be uploaded
        files_to_upload = []
        for root, dirs, files in os.walk(upload_path):
            for file in files:
                files_to_upload.append(os.path.join(root, file))
        print_with_rank(f"Found {len(files_to_upload)} files to upload:")
        for f in files_to_upload[:10]:  # Show first 10 files
            print_with_rank(f"  - {os.path.relpath(f, upload_path)}")
        if len(files_to_upload) > 10:
            print_with_rank(f"  ... and {len(files_to_upload) - 10} more files")
        
        # Upload all files - delete existing files and upload fresh
        api = HfApi()
        commit_msg = f"Upload Eagle3 checkpoint ({step_info})" if not final else "Upload final Eagle3 model"
        
        # Delete all existing files in the repo first to ensure clean upload
        try:
            existing_files = api.list_repo_files(repo_id=args.hf_repo_id, repo_type="model")
            if existing_files:
                print_with_rank(f"Deleting {len(existing_files)} existing files from repo...")
                api.delete_files(
                    repo_id=args.hf_repo_id,
                    repo_type="model",
                    paths=existing_files,
                    commit_message="Clear repo before uploading new checkpoint"
                )
        except Exception as e:
            print_with_rank(f"Note: Could not delete existing files (repo might be empty): {e}")
        
        # Upload all files from checkpoint - no patterns, upload everything
        print_with_rank(f"Uploading all files from {upload_path}...")
        api.upload_folder(
            folder_path=upload_path,
            repo_id=args.hf_repo_id,
            repo_type="model",
            commit_message=commit_msg,
            delete_patterns=None,  # Don't auto-delete anything
            ignore_patterns=None,  # Don't ignore anything
            allow_patterns=None,   # Allow everything
        )
        
        print_with_rank(f"✅ Successfully uploaded to https://huggingface.co/{args.hf_repo_id}")
        
    except Exception as e:
        print_with_rank(f"❌ Failed to upload to HuggingFace Hub: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import os
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.cuda.init()
    main()
