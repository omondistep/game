#!/usr/bin/env python3
"""
Feature Importance Analyzer for Football Prediction Model

This script uses machine learning techniques to automatically identify
the most important factors for predicting match outcomes.

Methods used:
1. Random Forest Feature Importance
2. Permutation Importance (more reliable)
3. SHAP values (if available)

Usage:
    python feature_importance_analyzer.py                # Analyze all leagues
    python feature_importance_analyzer.py --league Br1   # Analyze specific league
    python feature_importance_analyzer.py --update       # Update factor_weights.json
"""

import os
import sys
import json
import pickle
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

# Constants
DATA_DIR = "data"
MODELS_DIR = "models"
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "training_data.pkl")
WEIGHTS_FILE = os.path.join(DATA_DIR, "factor_weights.json")
IMPORTANCE_FILE = os.path.join(DATA_DIR, "feature_importance.json")


class FeatureImportanceAnalyzer:
    """Analyze feature importance using ML techniques."""
    
    # All possible features that can be extracted from match data
    ALL_FEATURES = [
        # Team strength indicators
        'home_position',
        'away_position',
        'home_form_points',      # W=3, D=1, L=0 from last 6 games
        'away_form_points',
        
        # Odds-based features
        'odds_home',
        'odds_draw', 
        'odds_away',
        'prob_home',             # Implied probability
        'prob_draw',
        'prob_away',
        'odds_ratio',            # home_odds / away_odds
        
        # Goal statistics
        'home_scored_avg',
        'home_conceded_avg',
        'away_scored_avg',
        'away_conceded_avg',
        'home_goal_diff',
        'away_goal_diff',
        'total_expected_goals',
        
        # Head-to-head
        'h2h_home_wins',
        'h2h_away_wins',
        'h2h_draws',
        'h2h_home_win_pct',
        'h2h_away_win_pct',
        
        # Match context
        'is_derby',              # Rivalry match
        'home_advantage',        # Calculated home advantage
        
        # Advanced stats (when available)
        'home_possession_avg',
        'away_possession_avg',
        'home_shots_avg',
        'away_shots_avg',
        'home_shots_on_target_avg',
        'away_shots_on_target_avg',
        
        # Form streaks
        'home_win_streak',
        'away_win_streak',
        'home_unbeaten_streak',
        'away_unbeaten_streak',
        'home_lose_streak',
        'away_lose_streak',
    ]
    
    # Feature groups for summary
    FEATURE_GROUPS = {
        'league_position': ['home_position', 'away_position'],
        'odds_analysis': ['odds_home', 'odds_draw', 'odds_away', 'prob_home', 
                          'prob_draw', 'prob_away', 'odds_ratio'],
        'recent_form': ['home_form_points', 'away_form_points', 'home_win_streak',
                        'away_win_streak', 'home_unbeaten_streak', 'away_unbeaten_streak',
                        'home_lose_streak', 'away_lose_streak'],
        'goals_stats': ['home_scored_avg', 'home_conceded_avg', 'away_scored_avg',
                        'away_conceded_avg', 'home_goal_diff', 'away_goal_diff',
                        'total_expected_goals'],
        'h2h': ['h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 
                'h2h_home_win_pct', 'h2h_away_win_pct'],
        'shots_possession': ['home_possession_avg', 'away_possession_avg',
                             'home_shots_avg', 'away_shots_avg'],
        'match_context': ['is_derby', 'home_advantage'],
    }
    
    def __init__(self):
        self.training_data = {}
        self.importance_results = {
            'global': {},
            'leagues': {},
            'analyzed_at': None
        }
    
    def load_training_data(self) -> bool:
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
    
    def extract_features_from_example(self, example: Dict) -> Tuple[Dict, int]:
        """
        Extract all possible features from a training example.
        
        Returns:
            Tuple of (features_dict, result_label)
            result_label: 0=home_win, 1=draw, 2=away_win
        """
        features = {}
        raw_features = example.get('features', {})
        labels = example.get('labels', {})
        match_data = example.get('match_data', {})
        
        # The training data already has computed features - use them directly
        # Probabilities (already computed from odds)
        features['prob_home'] = raw_features.get('prob_home', 33) or 33
        features['prob_draw'] = raw_features.get('prob_draw', 33) or 33
        features['prob_away'] = raw_features.get('prob_away', 33) or 33
        
        # Normalize to sum to 100
        total_prob = features['prob_home'] + features['prob_draw'] + features['prob_away']
        if total_prob > 0:
            features['prob_home'] = features['prob_home'] / total_prob * 100
            features['prob_draw'] = features['prob_draw'] / total_prob * 100
            features['prob_away'] = features['prob_away'] / total_prob * 100
        
        # Odds
        features['odds'] = raw_features.get('odds', 2.5) or 2.5
        features['odds_ratio'] = raw_features.get('prob_home_away_ratio', 1.0) or 1.0
        features['odds_draw'] = features['prob_draw']  # Use prob as proxy
        
        # Goals
        features['predicted_avg_goals'] = raw_features.get('predicted_avg_goals', 2.5) or 2.5
        features['home_scored_avg'] = raw_features.get('home_scored_avg', 1.4) or 1.4
        features['home_conceded_avg'] = raw_features.get('home_conceded_avg', 1.2) or 1.2
        features['away_scored_avg'] = raw_features.get('away_scored_avg', 1.2) or 1.2
        features['away_conceded_avg'] = raw_features.get('away_conceded_avg', 1.4) or 1.4
        
        features['home_goal_diff'] = features['home_scored_avg'] - features['home_conceded_avg']
        features['away_goal_diff'] = features['away_scored_avg'] - features['away_conceded_avg']
        
        # Position (if available)
        features['home_position'] = raw_features.get('home_position', 10) or 10
        features['away_position'] = raw_features.get('away_position', 10) or 10
        
        # Form points (calculate from form string if available)
        form = match_data.get('form', {})
        home_form = form.get('home', 'DDDDDD')
        away_form = form.get('away', 'DDDDDD')
        
        form_map = {'W': 3, 'D': 1, 'L': 0}
        features['home_form_points'] = sum(form_map.get(r, 1) for r in home_form[:6])
        features['away_form_points'] = sum(form_map.get(r, 1) for r in away_form[:6])
        
        # H2H
        h2h = match_data.get('head_to_head', {})
        h2h_sum = h2h.get('summary', {})
        features['h2h_home_wins'] = h2h_sum.get('home_wins', 0) or 0
        features['h2h_away_wins'] = h2h_sum.get('away_wins', 0) or 0
        features['h2h_draws'] = h2h_sum.get('draws', 0) or 0
        features['h2h_home_win_pct'] = (h2h_sum.get('home_win_pct', 33) or 33) / 100
        features['h2h_away_win_pct'] = (h2h_sum.get('away_win_pct', 33) or 33) / 100
        
        # Possession and shots
        features['home_possession_avg'] = raw_features.get('home_possession', 50) or 50
        features['away_possession_avg'] = raw_features.get('away_possession', 50) or 50
        features['home_shots_avg'] = raw_features.get('home_shots_avg', 13) or 13
        features['away_shots_avg'] = raw_features.get('away_shots_avg', 11) or 11
        
        # Match context
        features['home_advantage'] = features['prob_home'] - features['prob_away']
        
        # Streaks (would need to calculate from form)
        features['home_win_streak'] = self._calc_streak(home_form, 'W')
        features['away_win_streak'] = self._calc_streak(away_form, 'W')
        features['home_unbeaten_streak'] = self._calc_unbeaten_streak(home_form)
        features['away_unbeaten_streak'] = self._calc_unbeaten_streak(away_form)
        features['home_lose_streak'] = self._calc_streak(home_form, 'L')
        features['away_lose_streak'] = self._calc_streak(away_form, 'L')
        
        # Result label
        result = labels.get('result', 1)  # 0=home, 1=draw, 2=away
        
        return features, result
    
    def _calc_streak(self, form: str, result_char: str) -> int:
        """Calculate current streak of a result."""
        streak = 0
        for char in form:
            if char == result_char:
                streak += 1
            else:
                break
        return streak
    
    def _calc_unbeaten_streak(self, form: str) -> int:
        """Calculate unbeaten streak (W or D)."""
        streak = 0
        for char in form:
            if char in ('W', 'D'):
                streak += 1
            else:
                break
        return streak
    
    def prepare_dataset(self, league_key: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare dataset for ML analysis.
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        all_features = []
        all_labels = []
        
        leagues_to_analyze = [league_key] if league_key else list(self.training_data.keys())
        
        for league in leagues_to_analyze:
            if league not in self.training_data:
                continue
            
            league_data = self.training_data[league]
            examples = league_data.get('examples', [])
            
            for example in examples:
                features, label = self.extract_features_from_example(example)
                all_features.append(features)
                all_labels.append(label)
        
        if not all_features:
            return np.array([]), np.array([]), []
        
        # Get feature names (excluding any with all NaN/zero values)
        feature_names = list(all_features[0].keys())
        
        # Convert to arrays
        X = np.array([[f.get(fn, 0) for fn in feature_names] for f in all_features])
        y = np.array(all_labels)
        
        # Remove features with no variance
        valid_features = []
        for i, fn in enumerate(feature_names):
            if np.std(X[:, i]) > 0:
                valid_features.append(i)
        
        X = X[:, valid_features]
        feature_names = [feature_names[i] for i in valid_features]
        
        # Check if we have any valid features left
        if len(feature_names) == 0:
            return np.array([]).reshape(len(y), 0), y, []
        
        return X, y, feature_names
    
    def analyze_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                                   feature_names: List[str]) -> Dict:
        """
        Analyze feature importance using multiple methods.
        
        Returns dict with importance scores from each method.
        """
        if len(X) < 30:
            return {'error': 'Not enough data (need at least 30 samples)'}
        
        if len(feature_names) == 0 or X.shape[1] == 0:
            return {'error': 'No valid features (all have zero variance)'}
        
        results = {}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1. Random Forest Feature Importance
        print("  Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)
        
        rf_importance = dict(zip(feature_names, rf.feature_importances_))
        results['random_forest'] = rf_importance
        
        # Cross-validation score
        cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
        results['rf_cv_accuracy'] = float(np.mean(cv_scores))
        
        # 2. Gradient Boosting Feature Importance
        print("  Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X_scaled, y)
        
        gb_importance = dict(zip(feature_names, gb.feature_importances_))
        results['gradient_boosting'] = gb_importance
        
        # 3. Permutation Importance (more reliable)
        print("  Calculating Permutation Importance...")
        try:
            perm_importance = permutation_importance(rf, X_scaled, y, n_repeats=10, 
                                                      random_state=42, n_jobs=-1)
            perm_importance_dict = dict(zip(feature_names, perm_importance.importances_mean))
            results['permutation'] = perm_importance_dict
        except Exception as e:
            print(f"  Warning: Permutation importance failed: {e}")
        
        # 4. Combine into aggregate importance
        aggregate = {}
        for fn in feature_names:
            scores = []
            if fn in rf_importance:
                scores.append(rf_importance[fn])
            if fn in gb_importance:
                scores.append(gb_importance[fn])
            if fn in results.get('permutation', {}):
                # Normalize permutation importance
                perm_val = results['permutation'][fn]
                if perm_val < 0:
                    perm_val = 0
                scores.append(perm_val)
            
            aggregate[fn] = float(np.mean(scores)) if scores else 0.0
        
        results['aggregate'] = aggregate
        
        # 5. Group-level importance
        group_importance = {}
        for group, group_features in self.FEATURE_GROUPS.items():
            group_scores = [aggregate.get(f, 0) for f in group_features if f in aggregate]
            group_importance[group] = float(np.mean(group_scores)) if group_scores else 0.0
        
        # Normalize group importance
        total_group = sum(group_importance.values())
        if total_group > 0:
            group_importance = {k: v/total_group for k, v in group_importance.items()}
        
        results['group_importance'] = group_importance
        
        return results
    
    def analyze_all_leagues(self):
        """Analyze feature importance for all leagues."""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        self.importance_results['analyzed_at'] = datetime.now().isoformat()
        
        # Global analysis
        print("\nAnalyzing global dataset...")
        X, y, feature_names = self.prepare_dataset()
        
        if len(X) > 0:
            print(f"  Total samples: {len(X)}")
            print(f"  Features: {len(feature_names)}")
            
            global_results = self.analyze_feature_importance(X, y, feature_names)
            self.importance_results['global'] = global_results
            
            print(f"  CV Accuracy: {global_results.get('rf_cv_accuracy', 0):.1%}")
        else:
            print("  No data available for global analysis")
        
        # Per-league analysis
        for league_key in sorted(self.training_data.keys()):
            league_data = self.training_data[league_key]
            examples = league_data.get('examples', [])
            
            if len(examples) < 30:
                continue
            
            print(f"\nAnalyzing {league_key} ({len(examples)} samples)...")
            X, y, feature_names = self.prepare_dataset(league_key)
            
            if len(X) >= 30:
                league_results = self.analyze_feature_importance(X, y, feature_names)
                league_results['sample_count'] = len(X)
                self.importance_results['leagues'][league_key] = league_results
                
                print(f"  CV Accuracy: {league_results.get('rf_cv_accuracy', 0):.1%}")
        
        return self.importance_results
    
    def print_results(self, league_key: str = None):
        """Print feature importance results."""
        if league_key and league_key in self.importance_results['leagues']:
            results = self.importance_results['leagues'][league_key]
            print(f"\n{'='*60}")
            print(f"FEATURE IMPORTANCE FOR {league_key}")
            print('='*60)
        else:
            results = self.importance_results['global']
            print(f"\n{'='*60}")
            print("GLOBAL FEATURE IMPORTANCE")
            print('='*60)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        # Print group importance
        print("\nFACTOR GROUP IMPORTANCE:")
        print("-" * 40)
        group_imp = results.get('group_importance', {})
        for group, importance in sorted(group_imp.items(), key=lambda x: -x[1]):
            bar = '!' * int(importance * 50)
            print(f"  {group:20s} {importance:6.1%} {bar}")
        
        # Print top individual features
        print("\nTOP 15 INDIVIDUAL FEATURES:")
        print("-" * 40)
        aggregate = results.get('aggregate', {})
        sorted_features = sorted(aggregate.items(), key=lambda x: -x[1])[:15]
        
        for feature, importance in sorted_features:
            bar = '!' * int(importance * 50)
            print(f"  {feature:25s} {importance:6.1%} {bar}")
        
        # Print model performance
        print("\nMODEL PERFORMANCE:")
        print("-" * 40)
        print(f"  Random Forest CV Accuracy: {results.get('rf_cv_accuracy', 0):.1%}")
    
    def save_results(self):
        """Save importance results to JSON file."""
        # Convert numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        converted = convert_types(self.importance_results)
        
        with open(IMPORTANCE_FILE, 'w') as f:
            json.dump(converted, f, indent=2)
        
        print(f"\nSaved results to {IMPORTANCE_FILE}")
    
    def update_factor_weights(self):
        """Update factor_weights.json with new importance-based weights."""
        if not self.importance_results.get('global'):
            print("No analysis results to use for weights update")
            return
        
        # Load existing weights
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, 'r') as f:
                weights_data = json.load(f)
        else:
            weights_data = {'global': {}, 'leagues': {}}
        
        # Update global weights
        global_group = self.importance_results['global'].get('group_importance', {})
        if global_group:
            weights_data['global']['weights'] = global_group
            weights_data['global']['importance_source'] = 'ml_analysis'
            weights_data['global']['analyzed_at'] = self.importance_results['analyzed_at']
        
        # Update league-specific weights
        for league_key, results in self.importance_results.get('leagues', {}).items():
            group_imp = results.get('group_importance', {})
            if group_imp:
                if league_key not in weights_data['leagues']:
                    weights_data['leagues'][league_key] = {}
                
                weights_data['leagues'][league_key]['weights'] = group_imp
                weights_data['leagues'][league_key]['importance_source'] = 'ml_analysis'
                weights_data['leagues'][league_key]['sample_count'] = results.get('sample_count', 0)
        
        # Save updated weights
        with open(WEIGHTS_FILE, 'w') as f:
            json.dump(weights_data, f, indent=2)
        
        print(f"Updated factor weights in {WEIGHTS_FILE}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze feature importance for football prediction')
    parser.add_argument('--league', type=str, help='Analyze specific league only')
    parser.add_argument('--update', action='store_true', help='Update factor_weights.json with results')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    analyzer = FeatureImportanceAnalyzer()
    
    if not analyzer.load_training_data():
        sys.exit(1)
    
    analyzer.analyze_all_leagues()
    
    if not args.quiet:
        analyzer.print_results(args.league)
    
    analyzer.save_results()
    
    if args.update:
        analyzer.update_factor_weights()
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()