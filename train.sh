#!/bin/bash
# Football Prediction System - Daily Training Script
# Usage: ./train.sh [status|scrape-only|train-only|force]
#
# This script automates daily training by:
# 1. Scraping yesterday's match results from Forebet
# 2. Checking for new leagues
# 3. Retraining the model
# 4. Showing statistics
#
# Cron setup (run at 8 AM daily):
#   0 8 * * * cd /path/to/game && ./train.sh >> /var/log/football_training.log 2>&1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/football_env"
DAILY_TRAIN="${SCRIPT_DIR}/daily_train.py"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    exit 1
fi

if [ ! -f "$DAILY_TRAIN" ]; then
    echo "Error: daily_train.py not found at $DAILY_TRAIN"
    exit 1
fi

case "$1" in
    status|--status|-s)
        source "$VENV_DIR/bin/activate"
        python "$DAILY_TRAIN" --status
        ;;
    scrape-only|--scrape-only|-s)
        source "$VENV_DIR/bin/activate"
        python "$DAILY_TRAIN" --scrape-only
        ;;
    train-only|--train-only|-t)
        source "$VENV_DIR/bin/activate"
        python "$DAILY_TRAIN" --train-only
        ;;
    force|--force|-f)
        source "$VENV_DIR/bin/activate"
        python "$DAILY_TRAIN" --force
        ;;
    "")
        source "$VENV_DIR/bin/activate"
        python "$DAILY_TRAIN"
        ;;
    help|--help|-h)
        echo "Usage: $0 [status|scrape-only|train-only|force]"
        echo ""
        echo "Commands:"
        echo "  (none)      - Run full daily training (scrape + train)"
        echo "  status       - Show current training status"
        echo "  scrape-only  - Only scrape yesterday's results"
        echo "  train-only   - Only train models (skip scraping)"
        echo "  force        - Force training regardless of timing"
        echo ""
        echo "Cron setup (8 AM daily):"
        echo "  0 8 * * * cd $(pwd) && ./train.sh >> /var/log/football_training.log 2>&1"
        ;;
    *)
        echo "Usage: $0 [status|scrape-only|train-only|force]"
        exit 1
        ;;
esac
