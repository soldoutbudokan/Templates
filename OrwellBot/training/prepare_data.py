#!/usr/bin/env python3
"""
Prepare Orwell corpus for MLX fine-tuning.
Converts the scraped text into training format.
"""

import json
import random
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
OUTPUT_DIR = Path(__file__).parent / "data"


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list:
    """Split text into overlapping chunks for training."""
    words = text.split()
    chunks = []

    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 100:  # Skip tiny chunks
            chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def load_corpus() -> list:
    """Load all texts from the corpus."""
    jsonl_file = CORPUS_DIR / "orwell_training.jsonl"

    texts = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            texts.append({
                "text": data["text"],
                "title": data["title"],
                "category": data["category"]
            })

    return texts


def create_training_samples(texts: list, chunk_size: int = 512) -> list:
    """Create training samples from texts."""
    samples = []

    for item in texts:
        text = item["text"]

        # Clean up the text
        text = text.replace("\n\n---\n\n", "\n\n")  # Remove section separators
        text = " ".join(text.split())  # Normalize whitespace

        # Create chunks
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=64)

        for chunk in chunks:
            # Format for instruction-following (can also do plain text)
            samples.append({
                "text": chunk
            })

    return samples


def create_prompt_samples(texts: list) -> list:
    """Create instruction-style samples for more controlled generation."""
    samples = []

    prompts = [
        "Write in the style of George Orwell about",
        "Compose a passage in Orwell's voice discussing",
        "In the manner of George Orwell, describe",
        "Write an Orwellian analysis of",
        "Channel George Orwell to write about",
    ]

    for item in texts:
        text = item["text"]
        title = item["title"]
        category = item["category"]

        # Clean text
        text = text.replace("\n\n---\n\n", "\n\n")
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]

        for para in paragraphs[:50]:  # Limit per work
            prompt = random.choice(prompts)
            topic = f"the themes in {title}" if random.random() > 0.5 else "society and truth"

            samples.append({
                "text": f"<|user|>\n{prompt} {topic}.\n<|assistant|>\n{para}"
            })

    return samples


def main():
    print("Preparing training data...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load corpus
    texts = load_corpus()
    print(f"Loaded {len(texts)} works from corpus")

    # Create plain text samples (better for style transfer)
    plain_samples = create_training_samples(texts, chunk_size=384)
    print(f"Created {len(plain_samples)} plain text chunks")

    # Shuffle
    random.seed(42)
    random.shuffle(plain_samples)

    # Split into train/valid
    split_idx = int(len(plain_samples) * 0.95)
    train_samples = plain_samples[:split_idx]
    valid_samples = plain_samples[split_idx:]

    # Save as JSONL
    train_file = OUTPUT_DIR / "train.jsonl"
    valid_file = OUTPUT_DIR / "valid.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    with open(valid_file, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nSaved {len(train_samples)} training samples to {train_file}")
    print(f"Saved {len(valid_samples)} validation samples to {valid_file}")

    # Stats
    total_tokens_approx = sum(len(s["text"].split()) for s in plain_samples) * 1.3
    print(f"\nApproximate total tokens: {int(total_tokens_approx):,}")
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
