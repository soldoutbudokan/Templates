#!/usr/bin/env python3
"""
Fine-tune a language model on Orwell's writing using MLX.
Optimized for M3 MacBook Pro with 18GB RAM.
"""

import subprocess
import sys
from pathlib import Path

# Configuration
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"  # 4-bit quantized, fits in 18GB
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "adapters"

# Training hyperparameters (tuned for 18GB M3)
CONFIG = {
    "model": BASE_MODEL,
    "train": True,
    "data": str(DATA_DIR),
    "iters": 1000,              # Training iterations
    "batch_size": 2,            # Small batch for memory
    "num_layers": 16,           # Number of layers to fine-tune
    "learning_rate": 1e-5,      # Conservative LR
    "steps_per_report": 10,
    "steps_per_eval": 100,
    "save_every": 200,
    "adapter_path": str(OUTPUT_DIR),
    "max_seq_length": 512,      # Context length
    "grad_checkpoint": True,    # Save memory with gradient checkpointing
}


def check_dependencies():
    """Check if required packages are installed."""
    try:
        import mlx
        import mlx_lm
        print("MLX and MLX-LM loaded successfully")
        return True
    except ImportError:
        print("MLX-LM not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mlx-lm"])
        return True


def train():
    """Run the training using mlx_lm."""

    # Build command (using new mlx_lm CLI format)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", CONFIG["model"],
        "--data", CONFIG["data"],
        "--train",
        "--iters", str(CONFIG["iters"]),
        "--batch-size", str(CONFIG["batch_size"]),
        "--num-layers", str(CONFIG["num_layers"]),
        "--learning-rate", str(CONFIG["learning_rate"]),
        "--steps-per-report", str(CONFIG["steps_per_report"]),
        "--steps-per-eval", str(CONFIG["steps_per_eval"]),
        "--save-every", str(CONFIG["save_every"]),
        "--adapter-path", CONFIG["adapter_path"],
        "--max-seq-length", str(CONFIG["max_seq_length"]),
    ]

    if CONFIG.get("grad_checkpoint"):
        cmd.append("--grad-checkpoint")

    print("=" * 60)
    print("OrwellBot Training")
    print("=" * 60)
    print(f"Base model: {CONFIG['model']}")
    print(f"Data directory: {CONFIG['data']}")
    print(f"Output: {CONFIG['adapter_path']}")
    print(f"Iterations: {CONFIG['iters']}")
    print("=" * 60)
    print("\nStarting training...\n")

    # Run training
    subprocess.run(cmd)


def main():
    # Check data exists
    train_file = DATA_DIR / "train.jsonl"
    if not train_file.exists():
        print("Training data not found. Run prepare_data.py first.")
        print(f"Expected: {train_file}")
        sys.exit(1)

    # Check dependencies
    check_dependencies()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    train()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Adapters saved to: {OUTPUT_DIR}")
    print("\nTo generate text, run: python generate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
