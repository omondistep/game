#!/bin/bash
# Football Prediction System - Quick Training Script
# Usage: ./train.sh [status|force]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/football_env"
AUTO_TRAIN="${SCRIPT_DIR}/auto_train.py"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    exit 1
fi

if [ ! -f "$AUTO_TRAIN" ]; then
    echo "Error: auto_train.py not found at $AUTO_TRAIN"
    exit 1
fi

case "$1" in
    status|--status|-s)
        source "$VENV_DIR/bin/activate"
        python "$AUTO_TRAIN" --status
        ;;
    force|--force|-f)
        source "$VENV_DIR/bin/activate"
        python "$AUTO_TRAIN" --force
        ;;
    "")
        source "$VENV_DIR/bin/activate"
        python "$AUTO_TRAIN"
        ;;
    *)
        echo "Usage: $0 [status|force]"
        echo ""
        echo "Commands:"
        echo "  (none)   - Auto-train if 20+ hours since last training"
        echo "  status    - Show training status without training"
        echo "  force     - Force training regardless of time elapsed"
        exit 1
        ;;
esac
