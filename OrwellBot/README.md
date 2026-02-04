# OrwellBot

A language model fine-tuned on George Orwell's complete works using LoRA adapters. Built with MLX for Apple Silicon.

**Live Demo**: https://orwell-bot.vercel.app

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
├── training/
│   ├── prepare_data.py # Data preparation
│   ├── train.py        # LoRA fine-tuning
│   └── generate.py     # Text generation
├── deployment/
│   ├── convert_adapters.py  # MLX → PEFT converter
│   ├── modal_app.py         # Modal serverless backend
│   └── peft_adapters/       # Converted adapters
└── web/                     # Next.js frontend
    ├── app/
    │   ├── page.tsx         # Main UI
    │   └── api/generate/    # API proxy
    └── components/          # React components
```

## Model Details

- **Base model**: Qwen2.5-1.5B-Instruct (4-bit quantized)
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Training**: 1000 iterations on Orwell's novels, essays, and articles

## Web Deployment

OrwellBot can be deployed as a web app with:
- **Frontend**: Next.js on Vercel
- **Backend**: Modal serverless GPU for inference

### Prerequisites

```bash
pip install torch safetensors peft transformers modal
npm install -g vercel
modal setup  # Configure Modal account
```

### Step 1: Convert Adapters (One-time)

The MLX adapters need to be converted to HuggingFace PEFT format for deployment:

```bash
cd OrwellBot
python deployment/convert_adapters.py
```

This creates `deployment/peft_adapters/` with PEFT-compatible weights.

### Step 2: Deploy Modal Backend

```bash
# Create the Modal volume for adapters
modal volume create orwellbot-adapters

# Upload converted adapters
modal volume put orwellbot-adapters deployment/peft_adapters /orwellbot

# Deploy the app (note the endpoint URL in output)
modal deploy deployment/modal_app.py
```

Test the endpoint:
```bash
curl -X POST https://<your-app>--generate-text.modal.run \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Power is", "max_tokens": 100}'
```

### Step 3: Deploy Vercel Frontend

```bash
cd web

# Install dependencies
npm install

# Test locally (create .env.local with MODAL_ENDPOINT first)
cp .env.example .env.local
# Edit .env.local with your Modal endpoint URL
npm run dev

# Deploy to Vercel
npx vercel --prod
# Add MODAL_ENDPOINT environment variable in Vercel dashboard
```

### Cost Estimates

- **Modal free tier**: 30 GPU-hours/month (~100-500 generations)
- **Vercel free tier**: 100GB bandwidth, 100k function calls

### Performance Notes

- **Cold start**: First request ~30s (GPU container startup)
- **Warm requests**: <5s response time
- **Concurrent**: Up to 5 concurrent requests per container
