#!/bin/bash
# OrwellBot - Run script for training and generation

set -e
cd "$(dirname "$0")"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}OrwellBot${NC}"
echo "=========================================="

case "$1" in
    setup)
        echo "Setting up environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -q --upgrade pip
        pip install -q mlx mlx-lm transformers datasets huggingface_hub tqdm requests beautifulsoup4
        echo -e "${GREEN}Setup complete!${NC}"
        ;;

    scrape)
        echo "Scraping Orwell corpus..."
        source venv/bin/activate
        python scraper/scrape_orwell.py
        ;;

    prepare)
        echo "Preparing training data..."
        source venv/bin/activate
        python training/prepare_data.py
        ;;

    train)
        echo "Starting training..."
        source venv/bin/activate
        python training/train.py
        ;;

    generate)
        echo "Generating text..."
        source venv/bin/activate
        shift
        python training/generate.py "$@"
        ;;

    interactive)
        echo "Starting interactive mode..."
        source venv/bin/activate
        python training/generate.py --interactive
        ;;

    all)
        echo "Running full pipeline..."
        $0 setup
        $0 scrape
        $0 prepare
        $0 train
        echo -e "${GREEN}Pipeline complete!${NC}"
        ;;

    *)
        echo "Usage: ./run.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup       - Install dependencies"
        echo "  scrape      - Scrape Orwell corpus from orwell.ru"
        echo "  prepare     - Prepare training data"
        echo "  train       - Fine-tune the model"
        echo "  generate    - Generate text (use --prompt 'text' or --interactive)"
        echo "  interactive - Start interactive chat mode"
        echo "  all         - Run full pipeline (setup -> scrape -> prepare -> train)"
        echo ""
        echo "Example:"
        echo "  ./run.sh train"
        echo "  ./run.sh generate --prompt 'The nature of power is'"
        echo "  ./run.sh interactive"
        ;;
esac
