#!/usr/bin/env python3
"""
Calculate data-driven weights for the weighted prediction model.

This script analyzes historical match data to determine which factors
are most predictive for each league, and calculates optimal weights.

Usage:
    python calculate_weights.py                    # Calculate weights for all leagues
    python calculate_weights.py --league It2       # Calculate for specific league
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Constants
DATA_DIR = "data"
MODELS_DIR = "models"
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "training_data.pkl")
WEIGHTS_FILE = os.path.join(DATA_DIR, "factor_weights.json")


class WeightCalculator:
    """Calculate data-driven weights for prediction factors."""
    
    # Factors to analyze
    FACTORS = [
        'league_position',
        'odds_analysis', 
        'recent_form',
        'top5_performance',
        'goals_stats',
        'h2h',
        'shots_possession'
    ]
    
    def __init__(self):
        self.training_data = {}
        self.weights = {
            'global': {},
            'leagues': {}
        }
    
    def load_training_data(self):
        """Load training data from pickle file."""
        if not os.path.exists(TRAINING_DATA_FILE):
            print("No training data found. Run rebuild_data.py first.")
            return False
        
        try:
            with open(TRAINING_DATA_FILE, 'rb') as f:
                self.training_data = pickle.load(f)
            print(f"Loaded training data for {len(self.training_data)} leagues")
            return True
        except Exception as e:
            print(f"Error loading training data: {e}")
            return False
    
    def calculate_factor_correlation(self, factor_values: List[float], results: List[int]) -> float:
        """Calculate correlation between a factor and match results.
        
        Higher correlation = more predictive factor.
        """
        if len(factor_values) < 10:
            return 0.0
        
        try:
            # Convert to numpy arrays
            X = np.array(factor_values)
            y = np.array(results)
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(X, y)[0, 1]
            
            # Return absolute correlation (predictive power)
            return abs(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def analyze_league(self, league_key: str, league_data: Dict) -> Dict:
        """Analyze a single league to determine optimal weights."""
        examples = league_data.get('examples', [])
        
        if len(examples) < 20:
            return None
        
        # Extract factor values and results
        factor_data = {f: {'home': [], 'away': [], 'result': []} for f in self.FACTORS}
        
        for ex in examples:
            features = ex.get('features', {})
            labels = ex.get('labels', {})
            
            result = labels.get('result', 1)  # 0=home, 1=draw, 2=away
            
            # Extract factor values from features
            # League position (inverse - lower is better)
            home_pos = features.get('home_position', 10)
            away_pos = features.get('away_position', 10)
            factor_data['league_position']['home'].append(1 / max(home_pos, 1))
            factor_data['league_position']['away'].append(1 / max(away_pos, 1))
            factor_data['league_position']['result'].append(result)
            
            # Odds analysis
            prob_home = features.get('prob_home', 0.33)
            prob_away = features.get('prob_away', 0.33)
            factor_data['odds_analysis']['home'].append(prob_home)
            factor_data['odds_analysis']['away'].append(prob_away)
            factor_data['odds_analysis']['result'].append(result)
            
            # Recent form (using prob_home_away_ratio as proxy)
            ratio = features.get('prob_home_away_ratio', 1.0)
            factor_data['recent_form']['home'].append(min(ratio, 3) / 3)
            factor_data['recent_form']['away'].append(min(1/ratio, 3) / 3)
            factor_data['recent_form']['result'].append(result)
            
            # Goals stats (using predicted_avg_goals)
            avg_goals = features.get('predicted_avg_goals', 2.5)
            factor_data['goals_stats']['home'].append(avg_goals / 4)
            factor_data['goals_stats']['away'].append(avg_goals / 4)
            factor_data['goals_stats']['result'].append(result)
            
            # Other factors use defaults
            factor_data['top5_performance']['home'].append(0.5)
            factor_data['top5_performance']['away'].append(0.5)
            factor_data['top5_performance']['result'].append(result)
            
            factor_data['h2h']['home'].append(0.5)
            factor_data['h2h']['away'].append(0.5)
            factor_data['h2h']['result'].append(result)
            
            factor_data['shots_possession']['home'].append(0.5)
            factor_data['shots_possession']['away'].append(0.5)
            factor_data['shots_possession']['result'].append(result)
        
        # Calculate correlations for each factor
        correlations = {}
        for factor in self.FACTORS:
            # Combine home and away correlations
            home_corr = self.calculate_factor_correlation(
                factor_data[factor]['home'],
                factor_data[factor]['result']
            )
            away_corr = self.calculate_factor_correlation(
                factor_data[factor]['away'],
                factor_data[factor]['result']
            )
            correlations[factor] = (home_corr + away_corr) / 2
        
        return correlations
    
    def calculate_weights_from_correlations(self, correlations: Dict[str, float]) -> Dict[str, float]:
        """Convert correlations to weights (normalized to sum to 1)."""
        # Add small epsilon to avoid zero weights
        epsilon = 0.01
        adjusted = {k: max(v, epsilon) for k, v in correlations.items()}
        
        total = sum(adjusted.values())
        if total == 0:
            # Default weights if no correlations
            return {f: 1/len(self.FACTORS) for f in self.FACTORS}
        
        return {k: v/total for k, v in adjusted.items()}
    
    def calculate_all_weights(self):
        """Calculate weights for all leagues and global."""
        print("\nCalculating data-driven weights...")
        
        all_correlations = defaultdict(list)
        league_count = 0
        
        for league_key, league_data in self.training_data.items():
            examples = league_data.get('examples', [])
            
            if len(examples) >= 20:
                correlations = self.analyze_league(league_key, league_data)
                
                if correlations:
                    weights = self.calculate_weights_from_correlations(correlations)
                    self.weights['leagues'][league_key] = {
                        'weights': weights,
                        'correlations': correlations,
                        'example_count': len(examples)
                    }
                    
                    # Accumulate for global
                    for factor, corr in correlations.items():
                        all_correlations[factor].append(corr)
                    
                    league_count += 1
        
        # Calculate global weights
        global_correlations = {}
        for factor in self.FACTORS:
            corrs = all_correlations[factor]
            global_correlations[factor] = np.mean(corrs) if corrs else 0.1
        
        self.weights['global'] = {
            'weights': self.calculate_weights_from_correlations(global_correlations),
            'correlations': global_correlations,
            'league_count': league_count
        }
        
        print(f"Calculated weights for {league_count} leagues")
        
        return self.weights
    
    def save_weights(self):
        """Save calculated weights to JSON file."""
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(self.weights, f, indent=2)
        
        print(f"Saved weights to {WEIGHTS_FILE}")
    
    def print_weights(self, league_key: Optional[str] = None):
        """Print calculated weights."""
        if league_key and league_key in self.weights['leagues']:
            print(f"\nWeights for {league_key}:")
            league_weights = self.weights['leagues'][league_key]
            for factor, weight in sorted(league_weights['weights'].items(), key=lambda x: -x[1]):
                corr = league_weights['correlations'].get(factor, 0)
                print(f"  {factor}: {weight:.1%} (correlation: {corr:.3f})")
        else:
            print("\nGlobal weights:")
            global_weights = self.weights['global']
            for factor, weight in sorted(global_weights['weights'].items(), key=lambda x: -x[1]):
                corr = global_weights['correlations'].get(factor, 0)
                print(f"  {factor}: {weight:.1%} (correlation: {corr:.3f})")


def main():
    calculator = WeightCalculator()
    
    if not calculator.load_training_data():
        sys.exit(1)
    
    calculator.calculate_all_weights()
    calculator.save_weights()
    calculator.print_weights()


if __name__ == '__main__':
    main()
