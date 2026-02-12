#!/usr/bin/env python3
"""
Daily Training Script for Football Prediction Model

This script:
1. Scrapes historical data up to yesterday
2. Rebuilds the training database
3. Retrains all models (league-specific and global)

Usage:
    python daily_train.py                    # Train with all data up to yesterday
    python daily_train.py --force            # Force retrain even if already trained today
    python daily_train.py --dry-run          # Show what would be done without executing

Can be run as a cron job:
    0 6 * * * cd /home/stom/game && ./football_env/bin/python daily_train.py >> logs/daily_train.log 2>&1
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import subprocess

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Constants - use absolute paths
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
LOGS_DIR = os.path.join(SCRIPT_DIR, "logs")
LAST_TRAINING_FILE = os.path.join(DATA_DIR, "last_training.json")


def get_yesterday_date() -> str:
    """Get yesterday's date in YYYY-MM-DD format."""
    yesterday = datetime.now() - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def get_today_date() -> str:
    """Get today's date in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d")


def check_already_trained_today() -> bool:
    """Check if training was already done today."""
    if not os.path.exists(LAST_TRAINING_FILE):
        return False
    
    try:
        with open(LAST_TRAINING_FILE, 'r') as f:
            data = json.load(f)
        last_date = data.get('date', '')
        return last_date == get_today_date()
    except:
        return False


def mark_training_done():
    """Mark that training was done today."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LAST_TRAINING_FILE, 'w') as f:
        json.dump({
            'date': get_today_date(),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)


def get_available_data_files(end_date: str) -> List[str]:
    """Get all historical data files up to end_date."""
    pattern = os.path.join(DATA_DIR, "historical_matches_*.json")
    files = glob.glob(pattern)
    
    # Filter files by date
    valid_files = []
    for f in files:
        # Extract date from filename
        basename = os.path.basename(f)
        # Pattern: historical_matches_YYYY-MM-DD.json
        date_match = basename.replace('historical_matches_', '').replace('.json', '')
        try:
            file_date = datetime.strptime(date_match, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            if file_date <= end:
                valid_files.append(f)
        except:
            continue
    
    return sorted(valid_files)


def run_training(dry_run: bool = False) -> bool:
    """Run the training process."""
    yesterday = get_yesterday_date()
    
    print("=" * 60)
    print("DAILY TRAINING SCRIPT")
    print("=" * 60)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Training with data up to: {yesterday}")
    
    # Check available files
    files = get_available_data_files(yesterday)
    print(f"\nFound {len(files)} data files")
    
    if not files:
        print("ERROR: No data files found!")
        return False
    
    if dry_run:
        print("\n[DRY RUN] Would train with the following files:")
        for f in files:
            print(f"  - {f}")
        return True
    
    # Run rebuild_data.py
    print("\n" + "-" * 60)
    print("Running model training...")
    print("-" * 60)
    
    rebuild_script = os.path.join(SCRIPT_DIR, 'rebuild_data.py')
    result = subprocess.run(
        [sys.executable, rebuild_script],
        capture_output=False,
        text=True,
        cwd=SCRIPT_DIR
    )
    
    if result.returncode != 0:
        print("ERROR: Training failed!")
        return False
    
    # Mark training as done
    mark_training_done()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Daily training script for football prediction model')
    parser.add_argument('--force', '-f', action='store_true',
                       help='Force retrain even if already trained today')
    parser.add_argument('--dry-run', '-n', action='store_true',
                       help='Show what would be done without executing')
    parser.add_argument('--status', '-s', action='store_true',
                       help='Show training status and exit')
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Show status if requested
    if args.status:
        if check_already_trained_today():
            print("Training status: Already trained today")
        else:
            print("Training status: Not yet trained today")
        
        # Show last training info
        if os.path.exists(LAST_TRAINING_FILE):
            with open(LAST_TRAINING_FILE, 'r') as f:
                data = json.load(f)
            print(f"Last training: {data.get('timestamp', 'unknown')}")
        
        # Show model stats
        model_dirs = glob.glob(os.path.join(MODELS_DIR, '*'))
        print(f"Trained models: {len(model_dirs)}")
        return
    
    # Check if already trained today
    if check_already_trained_today() and not args.force:
        print("Already trained today. Use --force to retrain.")
        return
    
    # Run training
    success = run_training(dry_run=args.dry_run)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
