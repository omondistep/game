#!/usr/bin/env python3
"""
Model Statistics Command
Shows model accuracy stats and tracks improvement over time.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import argparse


def load_training_history() -> List[Dict]:
    """Load training history from file."""
    history_file = "training_history.json"
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            return json.load(f)
    return []


def save_training_history(history: List[Dict]):
    """Save training history to file."""
    history_file = "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, default=str)


def get_training_stats() -> Dict:
    """Get comprehensive training statistics."""
    history = load_training_history()
    matches_file = "match_data/results.json"
    training_file = "match_data/training_data.pkl"
    
    stats = {
        'total_trainings': len(history),
        'training_history': history[-10:] if history else [],  # Last 10
        'training_count': 0,
        'result_accuracy': 0.0,
        'ou_accuracy': 0.0,
        'improvement_trend': None,
        'recent_improvement': None
    }
    
    # Count training examples
    if os.path.exists(training_file):
        import pickle
        with open(training_file, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, list):
                stats['training_count'] = len(data)
    
    # Get latest training metrics
    if history:
        latest = history[-1]
        stats['result_accuracy'] = latest.get('result_accuracy', 0)
        stats['ou_accuracy'] = latest.get('ou_accuracy', 0)
        stats['last_training'] = latest.get('timestamp')
        
        # Calculate improvement trend
        if len(history) >= 2:
            prev = history[-2]
            result_diff = stats['result_accuracy'] - prev.get('result_accuracy', 0)
            ou_diff = stats['ou_accuracy'] - prev.get('ou_accuracy', 0)
            stats['improvement_trend'] = {
                'result_change': result_diff,
                'ou_change': ou_diff
            }
        
        # Recent improvement (last 5 trainings)
        if len(history) >= 5:
            first_of_5 = history[-5]
            last_of_5 = history[-1]
            stats['recent_improvement'] = {
                'result_change': last_of_5.get('result_accuracy', 0) - first_of_5.get('result_accuracy', 0),
                'ou_change': last_of_5.get('ou_accuracy', 0) - first_of_5.get('ou_accuracy', 0),
                'samples_added': last_of_5.get('training_examples', 0) - first_of_5.get('training_examples', 0)
            }
    
    return stats


def format_percentage(value: float) -> str:
    """Format percentage with color indicator."""
    if value >= 0.7:
        return f"{value:.1%} ✓"
    elif value >= 0.5:
        return f"{value:.1%} ~"
    else:
        return f"{value:.1%} ⚠"


def main():
    """Display model statistics."""
    stats = get_training_stats()
    
    print("=" * 60)
    print("FOOTBALL PREDICTION MODEL - STATISTICS")
    print("=" * 60)
    print()
    
    # Training count
    print(f"Training Dataset:")
    print(f"  Total examples: {stats['training_count']}")
    print()
    
    # Latest accuracy
    print("Latest Training Accuracy:")
    result_str = format_percentage(stats['result_accuracy'])
    ou_str = format_percentage(stats['ou_accuracy'])
    print(f"  Match Result (1/X/2): {result_str}")
    print(f"  Over/Under 2.5:      {ou_str}")
    print()
    
    # Training count
    if stats['total_trainings'] > 0:
        print(f"Total Trainings: {stats['total_trainings']}")
        last_train = stats.get('last_training')
        if last_train:
            last_date = datetime.fromisoformat(last_train)
            print(f"Last Training: {last_date.strftime('%Y-%m-%d %H:%M')}")
        print()
    
    # Improvement trend (last training)
    if stats['improvement_trend']:
        trend = stats['improvement_trend']
        result_change = trend['result_change']
        ou_change = trend['ou_change']
        
        print("Last Training Improvement:")
        if result_change > 0:
            print(f"  Result Accuracy: +{result_change:.2%} ↑")
        elif result_change < 0:
            print(f"  Result Accuracy: {result_change:.2%} ↓")
        else:
            print(f"  Result Accuracy: 0.00% -")
        
        if ou_change > 0:
            print(f"  O/U Accuracy:    +{ou_change:.2%} ↑")
        elif ou_change < 0:
            print(f"  O/U Accuracy:    {ou_change:.2%} ↓")
        else:
            print(f"  O/U Accuracy:    0.00% -")
        print()
    
    # Recent improvement (last 5 trainings)
    if stats['recent_improvement']:
        recent = stats['recent_improvement']
        print("Recent Improvement (Last 5 Trainings):")
        result_change = recent['result_change']
        ou_change = recent['ou_change']
        samples = recent['samples_added']
        
        if result_change > 0:
            print(f"  Result Accuracy: +{result_change:.2%} ↑")
        elif result_change < 0:
            print(f"  Result Accuracy: {result_change:.2%} ↓")
        else:
            print(f"  Result Accuracy: 0.00% -")
        
        if ou_change > 0:
            print(f"  O/U Accuracy:    +{ou_change:.2%} ↑")
        elif ou_change < 0:
            print(f"  O/U Accuracy:    {ou_change:.2%} ↓")
        else:
            print(f"  O/U Accuracy:    0.00% -")
        
        print(f"  Samples added:   +{samples}")
        print()
    
    # Training history
    if stats['training_history']:
        print("Training History (Last 10):")
        print("-" * 50)
        print(f"{'Date':<20} {'Result':<10} {'O/U':<10} {'Samples'}")
        print("-" * 50)
        for entry in reversed(stats['training_history']):
            timestamp = entry.get('timestamp', '')[:10]  # Just date
            result = entry.get('result_accuracy', 0)
            ou = entry.get('ou_accuracy', 0)
            samples = entry.get('training_examples', 0)
            print(f"{timestamp:<20} {result:>8.1%} {ou:>8.1%} {samples:>8}")
        print("-" * 50)
        print()
    
    # Summary
    print("Summary:")
    if stats['result_accuracy'] >= 0.7:
        print("  ✓ Model accuracy is good (>70%)")
    elif stats['result_accuracy'] >= 0.5:
        print("  ~ Model accuracy is moderate (50-70%)")
        print("    More training data should improve accuracy")
    else:
        print("  ⚠ Model accuracy needs improvement (<50%)")
        print("    Consider adding more match results and retraining")
    
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
