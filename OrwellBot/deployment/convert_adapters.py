#!/usr/bin/env python3
"""
Convert MLX LoRA adapters to HuggingFace PEFT format.

MLX and PEFT use different naming conventions for LoRA weights:
- MLX: layers.{n}.self_attn.{q,k,v,o}_proj.lora_{a,b}
- PEFT: base_model.model.model.layers.{n}.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight

This script converts the MLX safetensors format to PEFT-compatible format.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# Paths
SCRIPT_DIR = Path(__file__).parent
ADAPTER_DIR = SCRIPT_DIR.parent / "adapters"
OUTPUT_DIR = SCRIPT_DIR / "peft_adapters"

# LoRA config from adapter_config.json
LORA_RANK = 8
LORA_ALPHA = 20.0  # scale in MLX config
LORA_DROPOUT = 0.0
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
# Include both attention and MLP projections
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_mlx_adapters(adapter_path: Path) -> Dict[str, torch.Tensor]:
    """Load MLX LoRA adapters from safetensors file."""
    tensors = {}
    with safe_open(adapter_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def convert_key_mlx_to_peft(mlx_key: str) -> Optional[str]:
    """
    Convert MLX layer name to PEFT format.

    MLX format: model.layers.{n}.self_attn.{proj}.lora_{a,b}
                model.layers.{n}.mlp.{proj}.lora_{a,b}
    PEFT format: base_model.model.model.layers.{n}.self_attn.{proj}.lora_{A,B}.weight
                 base_model.model.model.layers.{n}.mlp.{proj}.lora_{A,B}.weight
    """
    # Pattern for MLX LoRA keys (with model. prefix)
    # Matches both self_attn and mlp modules
    pattern = r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)\.lora_([ab])"
    match = re.match(pattern, mlx_key)

    if not match:
        print(f"Warning: Unrecognized key format: {mlx_key}")
        return None

    layer_num = match.group(1)
    module_type = match.group(2)  # 'self_attn' or 'mlp'
    proj_name = match.group(3)
    lora_type = match.group(4).upper()  # 'a' -> 'A', 'b' -> 'B'

    peft_key = f"base_model.model.model.layers.{layer_num}.{module_type}.{proj_name}.lora_{lora_type}.weight"
    return peft_key


def convert_adapters(
    input_path: Path,
    output_dir: Path,
    rank: int = LORA_RANK,
    alpha: float = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
    base_model: str = BASE_MODEL,
    target_modules: list = TARGET_MODULES,
) -> None:
    """Convert MLX adapters to PEFT format and save."""
    print(f"Loading MLX adapters from: {input_path}")
    mlx_tensors = load_mlx_adapters(input_path)
    print(f"Found {len(mlx_tensors)} tensors")

    # Convert tensor names
    peft_tensors = {}
    for mlx_key, tensor in mlx_tensors.items():
        peft_key = convert_key_mlx_to_peft(mlx_key)
        if peft_key:
            # MLX stores as float16, keep as-is or convert to float32 for compatibility
            peft_tensors[peft_key] = tensor.to(torch.float16)
            print(f"  {mlx_key} -> {peft_key} {tensor.shape}")

    if not peft_tensors:
        print("Error: No tensors converted!")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save converted weights
    weights_path = output_dir / "adapter_model.safetensors"
    save_file(peft_tensors, str(weights_path))
    print(f"\nSaved weights to: {weights_path}")

    # Create PEFT adapter_config.json
    peft_config = {
        "auto_mapping": None,
        "base_model_name_or_path": base_model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": int(alpha),
        "lora_dropout": dropout,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }

    config_path = output_dir / "adapter_config.json"
    with open(config_path, "w") as f:
        json.dump(peft_config, f, indent=2)
    print(f"Saved config to: {config_path}")

    # Verify conversion
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)
    print(f"Total tensors converted: {len(peft_tensors)}")
    print(f"Output directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Upload to Modal volume:")
    print("     modal volume put orwellbot-adapters deployment/peft_adapters /orwellbot")
    print("  2. Deploy Modal app:")
    print("     modal deploy deployment/modal_app.py")


def main():
    """Main entry point."""
    input_path = ADAPTER_DIR / "adapters.safetensors"

    if not input_path.exists():
        print(f"Error: Adapter file not found: {input_path}")
        print("Please ensure training has been completed.")
        return

    convert_adapters(input_path, OUTPUT_DIR)


if __name__ == "__main__":
    main()
