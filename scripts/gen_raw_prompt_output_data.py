import argparse
import os
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import re


def comma_separated_ints(s):
    if not s:
        return []
    try:
        return [int(p) for p in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Inference ports must be comma-separated integers, e.g. 30000,30001,30002"
        )


# -----------------------------
# Conversion Function
# -----------------------------
def convert_example(prompt, output, tokenizer):
    user_message = [{"role": "user", "content": prompt}]
    user_message = tokenizer.apply_chat_template(user_message, tokenize=False, add_generation_prompt=True)
    return {
        "prompt": user_message,
        "output": output
    }

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(
        description="Generate raw prompt output data using a specified model."
    )

    # Define the arguments based on your shell command
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--custom-dataset", type=str, required=True, help="Path to the custom dataset")
    parser.add_argument("--prompt-column-name", type=str, default="prompt", help="Name of the prompt column (default: prompt)")
    parser.add_argument("--output-column-name", type=str, default="output", help="Name of the output column (default: output)")
    parser.add_argument("--subset-size", type=int, default=200000, help="Subset size for streaming dataset (default: 200000)")
    parser.add_argument("--output-name", type=str, required=True, help="Name of saved data")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature (default: 0.1)")
    parser.add_argument("--regen-data", action="store_true", help="Whether to regenerate data")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing (default: 8)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    parser.add_argument(
        "--inference-ports",
        type=comma_separated_ints,
        default=[],
        help="Comma-separated list of inference ports (e.g. 30000,30001,30002)",
    )
    # Parse the arguments
    args = parser.parse_args()

    # --- Your Logic Starts Here ---
    
    print(f"Starting generation with model: {args.model_name}")
    print(f"Dataset: {args.custom_dataset}")
    print(f"Output file location: /workspace/model-training-SpecForge/cache/dataset/{args.output_name}")
    
    # Check if output directory exists, create if not
    if not os.path.exists("/workspace/model-training-SpecForge/cache/dataset"):
        os.makedirs("/workspace/model-training-SpecForge/cache/dataset")
        print(f"Created directory: /workspace/model-training-SpecForge/cache/dataset")

    if args.regen_data:
        print("Flag 'regen-data' is active. Overwriting existing data if necessary.")

    # -----------------------------
    # Load Tokenizer
    # -----------------------------
    REPO_ID = args.model_name

    print(f"Loading tokenizer/template from {REPO_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Could not load tokenizer: {e}")
        exit(1)

    # -----------------------------
    # Stream Dataset
    # -----------------------------
    SUBSET_SIZE = args.subset_size
    DATASET_ID = args.custom_dataset

    print(f"Streaming first {SUBSET_SIZE} examples...")
    ds_stream = load_dataset(
        DATASET_ID, 
        args.split,
        streaming=True
    )['train'].take(SUBSET_SIZE)

    # -----------------------------
    # Process and Save to JSONL
    # -----------------------------
    OUTPUT_FILE = "/workspace/model-training-SpecForge/cache/dataset/" + args.output_name + ".jsonl"
    valid_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in tqdm(ds_stream, total=SUBSET_SIZE):
            result = convert_example(example[args.prompt_column_name], example[args.output_column_name], tokenizer)
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                valid_count += 1



    print(f"\n✅ Done! Saved {valid_count} valid examples to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
