# OrwellBot

A language model fine-tuned on George Orwell's complete works using LoRA adapters. Built with MLX for Apple Silicon.

## Requirements

- Apple Silicon Mac (M1/M2/M3)
- Python 3.9+
- ~18GB RAM recommended

## Quick Start

```bash
# Run the full pipeline (setup, scrape corpus, prepare data, train)
./run.sh all

# Or run the interactive mode directly (if already trained)
./run.sh interactive
```

## Commands

| Command | Description |
|---------|-------------|
| `./run.sh setup` | Install dependencies (MLX, transformers, etc.) |
| `./run.sh scrape` | Scrape Orwell's works from orwell.ru |
| `./run.sh prepare` | Prepare training data from corpus |
| `./run.sh train` | Fine-tune the model with LoRA |
| `./run.sh generate` | Generate text with a prompt |
| `./run.sh interactive` | Start interactive chat mode |
| `./run.sh all` | Run the full pipeline |

## Generating Text

### Single prompt

```bash
./run.sh generate --prompt "The nature of power is"
```

### With options

```bash
./run.sh generate --prompt "War is peace" --max-tokens 300 --temperature 0.8
```

Options:
- `--prompt, -p` - The text prompt
- `--max-tokens, -m` - Maximum tokens to generate (default: 200)
- `--temperature, -t` - Sampling temperature (default: 0.7, higher = more creative)

### Interactive mode

```bash
./run.sh interactive
```

Type prompts and get responses. Type `quit` to exit.

## Example Prompts

Try these prompts that align with Orwell's themes:

```
The greatest enemy of clear language is
In our time, political speech consists largely of
The purpose of Newspeak was not only to provide
Power is not a means; it is
```

## Project Structure

```
OrwellBot/
├── run.sh              # Main entry point
├── adapters/           # Trained LoRA adapters
├── corpus/             # Scraped Orwell texts
├── scraper/            # Web scraping scripts
└── training/
    ├── prepare_data.py # Data preparation
    ├── train.py        # LoRA fine-tuning
    └── generate.py     # Text generation
```

## Model Details

- **Base model**: Qwen2.5-1.5B-Instruct (4-bit quantized)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Training**: 1000 iterations on Orwell's novels, essays, and articles
