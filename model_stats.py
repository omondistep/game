#!/usr/bin/env python3
"""
Model Statistics and Prediction Accuracy Tracker

This script:
1. Shows model training statistics
2. Tracks prediction accuracy over time
3. Shows correct/incorrect prediction percentages

Usage:
    python model_stats.py                    # Show all model stats
    python model_stats.py --league It2       # Show stats for specific league
    python model_stats.py --predictions      # Show prediction accuracy
    python model_stats.py --update           # Update prediction tracking
"""

import os
import sys
import json
import glob
import pickle
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import argparse

# Constants
DATA_DIR = "data"
MODELS_DIR = "models"
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.json")
RESULTS_FILE = os.path.join(DATA_DIR, "results.json")
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "training_data.pkl")


def load_model_metadata() -> Dict:
    """Load metadata for all trained models."""
    models = {}
    
    for model_dir in glob.glob(os.path.join(MODELS_DIR, '*')):
        if not os.path.isdir(model_dir):
            continue
        
        metadata_file = os.path.join(model_dir, 'metadata.json')
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                model_name = os.path.basename(model_dir)
                models[model_name] = metadata
            except:
                continue
    
    return models


def load_predictions() -> List[Dict]:
    """Load saved predictions."""
    if not os.path.exists(PREDICTIONS_FILE):
        return []
    
    try:
        with open(PREDICTIONS_FILE, 'r') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except:
        return []


def load_results() -> Dict:
    """Load saved results."""
    if not os.path.exists(RESULTS_FILE):
        return {}
    
    try:
        with open(RESULTS_FILE, 'r') as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except:
        return {}


def load_training_data() -> Dict:
    """Load training data."""
    if not os.path.exists(TRAINING_DATA_FILE):
        return {}
    
    try:
        with open(TRAINING_DATA_FILE, 'rb') as f:
            return pickle.load(f)
    except:
        return {}


def show_model_stats(league_code: Optional[str] = None):
    """Show statistics for all trained models."""
    models = load_model_metadata()
    training_data = load_training_data()
    
    if not models:
        print("No trained models found.")
        return
    
    print("=" * 70)
    print("MODEL STATISTICS")
    print("=" * 70)
    print(f"Total models: {len(models)}")
    print(f"Training data file: {'Present' if training_data else 'Missing'}")
    
    # Group by country
    by_country = defaultdict(list)
    for name, meta in models.items():
        country = meta.get('league_info', {}).get('country', 'Unknown')
        by_country[country].append((name, meta))
    
    # Show stats
    total_examples = 0
    for country in sorted(by_country.keys()):
        if league_code and league_code.lower() not in country.lower():
            # Also check league name
            found = False
            for name, meta in by_country[country]:
                league = meta.get('league_info', {}).get('league', '')
                league_key = meta.get('league_key', '')
                if (league_code.lower() in league.lower() or 
                    league_code.lower() == league_key.lower()):
                    found = True
                    break
            if not found:
                continue
        
        print(f"\n{country}:")
        print("-" * 50)
        
        for name, meta in sorted(by_country[country], key=lambda x: x[0]):
            league_key = meta.get('league_key', 'N/A')
            league_name = meta.get('league_info', {}).get('league', 'Unknown')
            example_count = meta.get('example_count', 0)
            result_acc = meta.get('result_accuracy', 0)
            ou_acc = meta.get('ou_accuracy', 0)
            trained_at = meta.get('trained_at', 'Unknown')
            
            total_examples += example_count
            
            # Filter by league code if specified
            if league_code and league_code.lower() != league_key.lower():
                if league_code.lower() not in league_name.lower():
                    continue
            
            print(f"  {league_key}: {league_name}")
            print(f"    Examples: {example_count}")
            print(f"    Result accuracy: {result_acc:.1%}")
            print(f"    O/U accuracy: {ou_acc:.1%}")
            print(f"    Trained: {trained_at[:10] if trained_at else 'Unknown'}")


def show_prediction_stats():
    """Show prediction accuracy statistics."""
    predictions = load_predictions()
    results = load_results()
    
    print("\n" + "=" * 70)
    print("PREDICTION ACCURACY")
    print("=" * 70)
    
    if not predictions:
        print("No predictions recorded yet.")
        print("\nTo track predictions, use the prediction system and save results.")
        return
    
    # Analyze predictions
    total = len(predictions)
    with_results = 0
    correct_result = 0
    correct_ou = 0
    
    by_league = defaultdict(lambda: {'total': 0, 'correct_result': 0, 'correct_ou': 0})
    
    for pred in predictions:
        url = pred.get('url', '')
        league_code = pred.get('league_code', 'Unknown')
        
        # Check if we have a result for this prediction
        result = results.get(url)
        if not result:
            continue
        
        with_results += 1
        by_league[league_code]['total'] += 1
        
        # Check result prediction
        pred_result = pred.get('prediction', {}).get('result', {}).get('prediction')
        actual_result = result.get('result')
        
        if pred_result and actual_result:
            # Normalize predictions
            pred_map = {'1': 'home', 'X': 'draw', '2': 'away', 'home': 'home', 'draw': 'draw', 'away': 'away'}
            pred_normalized = pred_map.get(str(pred_result), pred_result)
            if pred_normalized == actual_result:
                correct_result += 1
                by_league[league_code]['correct_result'] += 1
        
        # Check O/U prediction
        pred_ou = pred.get('prediction', {}).get('over_under', {}).get('prediction')
        actual_ou = result.get('over_under_2_5')
        
        if pred_ou and actual_ou is not None:
            if pred_ou == actual_ou:
                correct_ou += 1
                by_league[league_code]['correct_ou'] += 1
    
    print(f"\nTotal predictions: {total}")
    print(f"Predictions with results: {with_results}")
    
    if with_results > 0:
        result_pct = correct_result / with_results * 100
        ou_pct = correct_ou / with_results * 100
        
        print(f"\nOverall Accuracy:")
        print(f"  Result predictions: {correct_result}/{with_results} ({result_pct:.1f}%)")
        print(f"  O/U predictions: {correct_ou}/{with_results} ({ou_pct:.1f}%)")
        
        # Show by league
        print(f"\nBy League:")
        print("-" * 50)
        
        for league, stats in sorted(by_league.items(), key=lambda x: x[1]['total'], reverse=True):
            if stats['total'] < 3:
                continue
            
            res_pct = stats['correct_result'] / stats['total'] * 100 if stats['total'] > 0 else 0
            ou_pct = stats['correct_ou'] / stats['total'] * 100 if stats['total'] > 0 else 0
            
            print(f"  {league}:")
            print(f"    Result: {stats['correct_result']}/{stats['total']} ({res_pct:.1f}%)")
            print(f"    O/U: {stats['correct_ou']}/{stats['total']} ({ou_pct:.1f}%)")


def show_global_model_stats():
    """Show global model statistics."""
    global_meta_file = os.path.join(MODELS_DIR, 'Global_Model', 'metadata.json')
    
    if not os.path.exists(global_meta_file):
        return
    
    try:
        with open(global_meta_file, 'r') as f:
            meta = json.load(f)
        
        print("\n" + "=" * 70)
        print("GLOBAL MODEL (All Leagues Combined)")
        print("=" * 70)
        print(f"Total examples: {meta.get('example_count', 'N/A')}")
        print(f"Result accuracy: {meta.get('result_accuracy', 0):.1%}")
        print(f"O/U accuracy: {meta.get('ou_accuracy', 0):.1%}")
        print(f"Trained: {meta.get('trained_at', 'Unknown')[:19]}")
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description='Model statistics and prediction accuracy tracker')
    parser.add_argument('--league', '-l', help='Show stats for specific league code')
    parser.add_argument('--predictions', '-p', action='store_true',
                       help='Show prediction accuracy statistics')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Show all statistics')
    
    args = parser.parse_args()
    
    if args.predictions:
        show_prediction_stats()
    elif args.all:
        show_model_stats(args.league)
        show_global_model_stats()
        show_prediction_stats()
    else:
        show_model_stats(args.league)
        show_global_model_stats()
        show_prediction_stats()


if __name__ == '__main__':
    main()
