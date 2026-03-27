"""
Preprocessing for prompt-output format datasets.

This module provides functionality to process datasets that have 'prompt' and 'output' 
columns without applying any chat template. It trains only on output tokens.
"""

import torch
from typing import Dict, List
from transformers import PreTrainedTokenizer


def preprocess_prompt_output(
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    outputs: List[str],
    max_length: int = 2048,
) -> Dict[str, List[torch.Tensor]]:
    """
    Preprocess prompt-output pairs without applying chat template.
    
    Args:
        tokenizer: The tokenizer to use for tokenization.
        prompts: List of prompt strings.
        outputs: List of output strings.
        max_length: The maximum length of the tokenized input.
    
    Returns:
        A dictionary containing:
            - input_ids: List of tokenized input IDs.
            - loss_mask: List of loss masks (1 for output tokens, 0 for prompt tokens).
            - attention_mask: List of attention masks.
    """
    results = {"input_ids": [], "loss_mask": [], "attention_mask": []}
    
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    
    for prompt, output in zip(prompts, outputs):
        if not prompt or not output:
            continue
            
        # Tokenize prompt and output separately to know boundaries
        prompt_encoding = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        output_encoding = tokenizer(
            output,
            add_special_tokens=False,
            return_tensors="pt",
        )
        
        prompt_ids = prompt_encoding.input_ids[0]
        output_ids = output_encoding.input_ids[0]
        
        # Concatenate prompt + output
        input_ids = torch.cat([prompt_ids, output_ids])
        
        # Truncate if needed
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        # Create loss mask: 0 for prompt tokens, 1 for output tokens
        prompt_length = min(len(prompt_ids), max_length)
        total_length = len(input_ids)
        
        loss_mask = torch.zeros(total_length, dtype=torch.long)
        loss_mask[prompt_length:] = 1  # Only train on output tokens
        
        # Attention mask (all 1s)
        attention_mask = torch.ones(total_length, dtype=torch.long)
        
        results["input_ids"].append(input_ids[None, :])
        results["loss_mask"].append(loss_mask[None, :])
        results["attention_mask"].append(attention_mask[None, :])
    
    return results


def build_prompt_output_dataset(
    dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    shuffle_seed: int = 42,
    num_proc: int = 8,
    cache_dir: str = None,
    cache_key: str = None,
):
    """
    Build dataset from prompt-output format without chat template.
    
    Args:
        dataset: HF dataset with 'prompt' and 'output' columns.
        tokenizer: The tokenizer to use.
        max_length: Maximum sequence length.
        shuffle_seed: Random seed for shuffling.
        num_proc: Number of processes for mapping.
        cache_dir: Directory for caching.
        cache_key: Cache key for processed dataset.
    
    Returns:
        Processed HF dataset.
    """
    import os
    import warnings
    
    dataset = dataset.shuffle(seed=shuffle_seed)
    original_cols = dataset.column_names
    
    # Validate columns
    if "prompt" not in original_cols or "output" not in original_cols:
        raise ValueError(
            f"Dataset must have 'prompt' and 'output' columns. Found: {original_cols}"
        )
    
    def preprocess_function(examples):
        return preprocess_prompt_output(
            tokenizer,
            examples["prompt"],
            examples["output"],
            max_length,
        )
    
    # Setup caching
    if cache_dir and cache_key:
        load_from_cache_file = True
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = os.path.join(cache_dir, f"{cache_key}_prompt_output.pkl")
        print(f"Dataset is cached at {cache_file_name}")
    elif cache_dir is None and cache_key is None:
        load_from_cache_file = False
        cache_file_name = None
        print(f"Dataset is not cached")
    else:
        warnings.warn(
            "cache_dir and cache_key must be provided together to make caching work"
        )
        load_from_cache_file = False
        cache_file_name = None
    
    # Process dataset
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        batch_size=1000,
        remove_columns=original_cols,
        load_from_cache_file=load_from_cache_file,
        cache_file_name=cache_file_name,
    )
    
    dataset.set_format(type="torch")
    return dataset
