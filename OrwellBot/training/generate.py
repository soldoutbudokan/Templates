#!/usr/bin/env python3
"""
Generate text using the fine-tuned OrwellBot model.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Paths
ADAPTER_DIR = Path(__file__).parent.parent / "adapters"
BASE_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def generate(prompt: str, max_tokens: int = 200, temperature: float = 0.7):
    """Generate text using mlx_lm CLI."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", BASE_MODEL,
        "--max-tokens", str(max_tokens),
        "--temp", str(temperature),
        "--prompt", prompt,
    ]

    # Add adapter if exists
    adapter_file = ADAPTER_DIR / "adapters.safetensors"
    if adapter_file.exists():
        cmd.extend(["--adapter-path", str(ADAPTER_DIR)])
        print("Using OrwellBot (fine-tuned on Orwell's works)\n")
    else:
        print("Warning: No adapters found, using base model\n")

    print(f"Prompt: {prompt}\n")
    print("-" * 60)

    subprocess.run(cmd)

    print("-" * 60)


def interactive_mode():
    """Interactive generation mode."""
    adapter_file = ADAPTER_DIR / "adapters.safetensors"

    print("=" * 60)
    print("OrwellBot Interactive Mode")
    print("=" * 60)

    if adapter_file.exists():
        print("Model: OrwellBot (fine-tuned on Orwell's works)")
    else:
        print("Model: Base model (no fine-tuning)")

    print("\nType your prompt and press Enter. Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not prompt:
                continue

            print("\nOrwellBot:")
            cmd = [
                sys.executable, "-m", "mlx_lm", "generate",
                "--model", BASE_MODEL,
                "--max-tokens", "200",
                "--temp", "0.7",
                "--prompt", prompt,
            ]

            if adapter_file.exists():
                cmd.extend(["--adapter-path", str(ADAPTER_DIR)])

            # Run and capture output
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Parse output to get just the generated text
            output = result.stdout
            if "==========" in output:
                # Extract text between ========== markers
                parts = output.split("==========")
                if len(parts) >= 2:
                    generated = parts[1].strip()
                    # Remove the prompt echo if present
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    print(generated)
            else:
                print(output)

            print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except EOFError:
            break


def main():
    parser = argparse.ArgumentParser(description="Generate text with OrwellBot")
    parser.add_argument("--prompt", "-p", type=str, help="Text prompt")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--max-tokens", "-m", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.prompt:
        generate(args.prompt, args.max_tokens, args.temperature)
    else:
        # Default demo
        print("=" * 60)
        print("OrwellBot Demo")
        print("=" * 60)
        print("\nNo prompt provided. Running demo...\n")

        demo_prompts = [
            "The greatest enemy of clear language is",
            "In our time, political speech consists largely of",
        ]

        for prompt in demo_prompts:
            generate(prompt, max_tokens=150, temperature=0.7)
            print("\n")


if __name__ == "__main__":
    main()
