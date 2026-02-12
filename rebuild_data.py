#!/usr/bin/env python3
"""
Rebuild Database and Train Models from Historical JSON Data
Reads match data from historical_matches_*.json files and trains models.

Usage:
    python rebuild_data.py                          # Rebuild from all historical data
    python rebuild_data.py --date 2026-02-01        # Rebuild from specific date
    python rebuild_data.py --file data/historical_matches_2026-02-01.json
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

# Constants
DATA_DIR = "data"
MODELS_DIR = "models"
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "training_data.pkl")
LEAGUES_DB_FILE = os.path.join(DATA_DIR, "leagues_db.json")
COMPREHENSIVE_DB_FILE = os.path.join(DATA_DIR, "comprehensive_leagues_db.json")


class LeagueDatabase:
    """Manages league information from JSON data using comprehensive database."""
    
    def __init__(self):
        self.leagues = {}
        self.short_code_lookup = {}
        self.teams = {}
        self.load_existing()
    
    def load_existing(self):
        """Load existing league database - prefer comprehensive database."""
        # Try comprehensive database first
        if os.path.exists(COMPREHENSIVE_DB_FILE):
            try:
                with open(COMPREHENSIVE_DB_FILE, 'r', encoding='utf-8') as f:
                    comp_db = json.load(f)
                
                lookups = comp_db.get('lookups', {})
                self.short_code_lookup = lookups.get('by_short_code', {})
                
                # Build leagues from short_code_lookup
                self.leagues = {}
                for code, info in self.short_code_lookup.items():
                    self.leagues[code] = {
                        'league_code': code,
                        'country': info.get('country', ''),
                        'league': info.get('league', ''),
                        'league_url_path': info.get('league_url_path', ''),
                        'country_code': info.get('country_code', ''),
                        'match_count': info.get('match_count', 0),
                        'teams': info.get('teams', {})
                    }
                
                print(f"Loaded {len(self.leagues)} leagues from comprehensive database")
                return
            except Exception as e:
                print(f"Error loading comprehensive database: {e}")
        
        # Fallback to old leagues_db.json
        if os.path.exists(LEAGUES_DB_FILE):
            try:
                with open(LEAGUES_DB_FILE, 'r', encoding='utf-8') as f:
                    self.leagues = json.load(f)
                print(f"Loaded {len(self.leagues)} leagues from database")
            except Exception as e:
                print(f"Error loading league database: {e}")
                self.leagues = {}
    
    def save(self):
        """Save league database."""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(LEAGUES_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.leagues, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.leagues)} leagues to database")
    
    def get_league_info(self, league_code: str) -> Dict:
        """Get league info from short_code lookup."""
        if league_code in self.short_code_lookup:
            return self.short_code_lookup[league_code]
        return self.leagues.get(league_code, {})
    
    def add_league(self, league_info: Dict) -> str:
        """Add or update a league from match data. Returns league_key (short_code)."""
        league_code = league_info.get('league_code', '')
        
        # Use short_code as the primary key (consistent with scraper)
        league_key = league_code if league_code else 'Unknown'
        
        if league_key not in self.leagues:
            self.leagues[league_key] = {
                'league_code': league_code,
                'country': league_info.get('country', ''),
                'league': league_info.get('league', ''),
                'league_url_path': league_info.get('league_url_path', ''),
                'country_code': league_info.get('country_code', ''),
                'match_count': 0,
                'teams': {}
            }
        
        self.leagues[league_key]['match_count'] += 1
        
        # Add teams
        home_team = league_info.get('home_team', '')
        away_team = league_info.get('away_team', '')
        
        if home_team and home_team not in self.leagues[league_key]['teams']:
            self.leagues[league_key]['teams'][home_team] = {'home_matches': 0, 'away_matches': 0}
        if home_team:
            self.leagues[league_key]['teams'][home_team]['home_matches'] += 1
        
        if away_team and away_team not in self.leagues[league_key]['teams']:
            self.leagues[league_key]['teams'][away_team] = {'home_matches': 0, 'away_matches': 0}
        if away_team:
            self.leagues[league_key]['teams'][away_team]['away_matches'] += 1
        
        return league_key


class TrainingDataBuilder:
    """Builds training data from historical JSON matches."""
    
    def __init__(self):
        self.training_data = {}
        self.league_db = LeagueDatabase()
    
    def extract_features(self, match: Dict) -> Dict:
        """Extract features from a match for training."""
        features = {}
        
        # Probabilities from Forebet
        features['prob_home'] = match.get('prob_home', 0.33)
        features['prob_draw'] = match.get('prob_draw', 0.33)
        features['prob_away'] = match.get('prob_away', 0.34)
        
        # Derived features
        features['prob_home_away_ratio'] = features['prob_home'] / max(features['prob_away'], 0.01)
        features['prob_draw_diff'] = abs(features['prob_home'] - features['prob_away'])
        
        # Predicted average goals
        features['predicted_avg_goals'] = match.get('predicted_avg_goals', 2.5)
        
        # Odds (if available)
        odds = match.get('odds')
        if odds:
            features['odds'] = float(odds)
        else:
            # Calculate implied odds from probabilities
            total = features['prob_home'] + features['prob_draw'] + features['prob_away']
            features['odds'] = 1.0 / max(features['prob_home'] / total, 0.01)
        
        # League encoding
        league_code = match.get('league_code', 'Unknown')
        features['league_code_encoded'] = hash(league_code) % 100
        
        return features
    
    def extract_labels(self, match: Dict) -> Dict:
        """Extract labels (actual results) from match."""
        labels = {}
        
        # Result
        actual_result = match.get('actual_result', '')
        if actual_result == 'home':
            labels['result'] = 0  # home win
        elif actual_result == 'away':
            labels['result'] = 2  # away win
        else:
            labels['result'] = 1  # draw
        
        # Scores
        labels['home_score'] = match.get('home_score', 0)
        labels['away_score'] = match.get('away_score', 0)
        labels['total_goals'] = labels['home_score'] + labels['away_score']
        
        # Over/Under 2.5
        labels['over_2_5'] = 1 if labels['total_goals'] > 2.5 else 0
        
        # Correct prediction (did Forebet get it right?)
        prediction = match.get('prediction', '')
        if prediction and actual_result:
            if prediction.lower() in ['home', '1']:
                pred_result = 0
            elif prediction.lower() in ['away', '2']:
                pred_result = 2
            else:
                pred_result = 1
            
            labels['forebet_correct'] = 1 if pred_result == labels['result'] else 0
        else:
            labels['forebet_correct'] = 0
        
        return labels
    
    def process_match(self, match: Dict) -> Optional[Tuple[str, Dict, Dict]]:
        """Process a single match and return features/labels."""
        if not match.get('has_result', False):
            return None
        
        # Get league key (use short_code as primary key)
        league_code = match.get('league_code', '')
        league_key = self.league_db.add_league(match)
        
        # Extract features and labels
        features = self.extract_features(match)
        labels = self.extract_labels(match)
        
        return (league_key, features, labels)
    
    def load_json_file(self, filepath: str) -> List[Dict]:
        """Load matches from a JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return []
    
    def build_from_files(self, patterns: List[str]):
        """Build training data from JSON files matching patterns."""
        all_matches = []
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for filepath in sorted(files):
                print(f"Loading: {filepath}")
                matches = self.load_json_file(filepath)
                all_matches.extend(matches)
                print(f"  Loaded {len(matches)} matches")
        
        print(f"\nTotal matches loaded: {len(all_matches)}")
        
        # Process matches
        processed = 0
        skipped_no_result = 0
        
        for match in all_matches:
            result = self.process_match(match)
            if result:
                league_key, features, labels = result
                
                if league_key not in self.training_data:
                    self.training_data[league_key] = {
                        'league_info': self.league_db.leagues.get(league_key, {}),
                        'examples': []
                    }
                
                self.training_data[league_key]['examples'].append({
                    'features': features,
                    'labels': labels,
                    'match_id': match.get('match_id', ''),
                    'url': match.get('url', ''),
                    'home_team': match.get('home_team', ''),
                    'away_team': match.get('away_team', ''),
                    'date': match.get('date', '')
                })
                processed += 1
            else:
                skipped_no_result += 1
        
        print(f"Processed: {processed} matches")
        print(f"Skipped (no result): {skipped_no_result} matches")
        print(f"Leagues found: {len(self.training_data)}")
        
        # Save league database
        self.league_db.save()
        
        return self.training_data
    
    def save_training_data(self, filepath: str = TRAINING_DATA_FILE):
        """Save training data to pickle file."""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.training_data, f)
        print(f"Training data saved to {filepath}")


class ModelTrainer:
    """Trains models from the training data."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
    
    def train_league(self, league_key: str, league_data: Dict) -> bool:
        """Train models for a specific league with train/test split."""
        examples = league_data.get('examples', [])
        if len(examples) < 5:
            print(f"  Skipping {league_key}: only {len(examples)} examples")
            return False
        
        print(f"\nTraining for {league_key}:")
        print(f"  League: {league_data['league_info'].get('league', 'Unknown')}")
        print(f"  Country: {league_data['league_info'].get('country', 'Unknown')}")
        print(f"  Examples: {len(examples)}")
        
        # Extract features and labels
        X = []
        y_result = []
        y_over_2_5 = []
        
        for ex in examples:
            feat = ex['features']
            label = ex['labels']
            
            feature_vec = [
                feat.get('prob_home', 0.33),
                feat.get('prob_draw', 0.33),
                feat.get('prob_away', 0.34),
                feat.get('prob_home_away_ratio', 1.0),
                feat.get('prob_draw_diff', 0.0),
                feat.get('predicted_avg_goals', 2.5),
                feat.get('odds', 2.0),
                feat.get('league_code_encoded', 0),
            ]
            
            X.append(feature_vec)
            y_result.append(label.get('result', 1))
            y_over_2_5.append(label.get('over_2_5', 0))
        
        X = np.array(X)
        y_result = np.array(y_result)
        y_over_2_5 = np.array(y_over_2_5)
        
        # Split data into train/test sets (80/20)
        if len(examples) >= 10:
            X_train, X_test, y_result_train, y_result_test, y_ou_train, y_ou_test = train_test_split(
                X, y_result, y_over_2_5, test_size=0.2, random_state=42
            )
        else:
            # Too few examples for split, use all for training
            X_train, X_test = X, X
            y_result_train, y_result_test = y_result, y_result
            y_ou_train, y_ou_test = y_over_2_5, y_over_2_5
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else X_train_scaled
        
        # Train result model
        result_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight='balanced'
        )
        
        try:
            result_model.fit(X_train_scaled, y_result_train)
        except Exception as e:
            print(f"  Error training result model: {e}")
            return False
        
        # Train over/under model
        ou_model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42
        )
        
        try:
            ou_model.fit(X_train_scaled, y_ou_train)
        except Exception as e:
            print(f"  Error training O/U model: {e}")
            return False
        
        # Calculate accuracy on test set
        result_acc = result_model.score(X_test_scaled, y_result_test)
        ou_acc = ou_model.score(X_test_scaled, y_ou_test)
        
        print(f"  Result accuracy (test): {result_acc:.2%}")
        print(f"  O/U accuracy (test): {ou_acc:.2%}")
        
        # Store models
        league_name = league_data['league_info'].get('league', 'Unknown')
        country = league_data['league_info'].get('country', 'Unknown')
        model_name = f"{country} {league_name}".replace(' ', '_')
        
        self.models[league_key] = {
            'result_model': result_model,
            'ou_model': ou_model,
            'scaler': scaler,
            'league_info': league_data['league_info'],
            'example_count': len(examples),
            'result_accuracy': result_acc,
            'ou_accuracy': ou_acc
        }
        
        # Save models
        self.save_models(league_key, model_name)
        
        return True
    
    def save_models(self, league_key: str, model_name: str):
        """Save trained models to disk."""
        if league_key not in self.models:
            return
        
        model_data = self.models[league_key]
        league_dir = os.path.join(MODELS_DIR, model_name)
        os.makedirs(league_dir, exist_ok=True)
        
        # Save models
        with open(os.path.join(league_dir, 'result_model.pkl'), 'wb') as f:
            pickle.dump(model_data['result_model'], f)
        
        with open(os.path.join(league_dir, 'ou_model.pkl'), 'wb') as f:
            pickle.dump(model_data['ou_model'], f)
        
        with open(os.path.join(league_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(model_data['scaler'], f)
        
        # Save metadata
        metadata = {
            'league_key': league_key,
            'league_info': model_data['league_info'],
            'example_count': model_data['example_count'],
            'result_accuracy': model_data.get('result_accuracy', 0),
            'ou_accuracy': model_data.get('ou_accuracy', 0),
            'trained_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(league_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Models saved to {league_dir}")
    
    def train_all(self, training_data: Dict):
        """Train models for all leagues."""
        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)
        
        success_count = 0
        for league_key, league_data in sorted(training_data.items()):
            if self.train_league(league_key, league_data):
                success_count += 1
        
        # Train global model with all data
        print("\n" + "-" * 60)
        print("TRAINING GLOBAL MODEL (all leagues combined)")
        print("-" * 60)
        self.train_global_model(training_data)
        
        print(f"\n" + "=" * 60)
        print(f"TRAINING COMPLETE")
        print(f"=" * 60)
        print(f"Leagues trained successfully: {success_count}/{len(training_data)}")
        print(f"Models saved to: {MODELS_DIR}")
    
    def train_global_model(self, training_data: Dict):
        """Train a global model using all available data."""
        # Collect all examples from all leagues
        all_examples = []
        for league_data in training_data.values():
            all_examples.extend(league_data.get('examples', []))
        
        if len(all_examples) < 50:
            print(f"  Skipping global model: only {len(all_examples)} examples")
            return
        
        print(f"  Total examples: {len(all_examples)}")
        
        # Extract features and labels
        X = []
        y_result = []
        y_over_2_5 = []
        
        for ex in all_examples:
            feat = ex['features']
            label = ex['labels']
            
            feature_vec = [
                feat.get('prob_home', 0.33),
                feat.get('prob_draw', 0.33),
                feat.get('prob_away', 0.34),
                feat.get('prob_home_away_ratio', 1.0),
                feat.get('prob_draw_diff', 0.0),
                feat.get('predicted_avg_goals', 2.5),
                feat.get('odds', 2.0),
                feat.get('league_code_encoded', 0),
            ]
            
            X.append(feature_vec)
            y_result.append(label.get('result', 1))
            y_over_2_5.append(label.get('over_2_5', 0))
        
        X = np.array(X)
        y_result = np.array(y_result)
        y_over_2_5 = np.array(y_over_2_5)
        
        # Split for validation
        X_train, X_test, y_result_train, y_result_test, y_ou_train, y_ou_test = train_test_split(
            X, y_result, y_over_2_5, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train result model
        result_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        try:
            result_model.fit(X_train_scaled, y_result_train)
            result_acc = result_model.score(X_test_scaled, y_result_test)
            print(f"  Result model accuracy: {result_acc:.2%}")
        except Exception as e:
            print(f"  Error training result model: {e}")
            return
        
        # Train over/under model
        ou_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        try:
            ou_model.fit(X_train_scaled, y_ou_train)
            ou_acc = ou_model.score(X_test_scaled, y_ou_test)
            print(f"  O/U model accuracy: {ou_acc:.2%}")
        except Exception as e:
            print(f"  Error training O/U model: {e}")
            return
        
        # Save global model
        global_dir = os.path.join(MODELS_DIR, 'Global_Model')
        os.makedirs(global_dir, exist_ok=True)
        
        with open(os.path.join(global_dir, 'result_model.pkl'), 'wb') as f:
            pickle.dump(result_model, f)
        
        with open(os.path.join(global_dir, 'ou_model.pkl'), 'wb') as f:
            pickle.dump(ou_model, f)
        
        with open(os.path.join(global_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        
        metadata = {
            'league_key': 'Global',
            'league_info': {'league': 'All Leagues', 'country': 'Global'},
            'example_count': len(all_examples),
            'result_accuracy': result_acc,
            'ou_accuracy': ou_acc,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(global_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Global model saved to {global_dir}")


def main():
    parser = argparse.ArgumentParser(description='Rebuild database and train models from JSON data')
    parser.add_argument('--date', '-d', help='Specific date (YYYY-MM-DD)')
    parser.add_argument('--file', '-f', help='Specific JSON file path')
    parser.add_argument('--dir', default=DATA_DIR, help='Directory containing JSON files')
    parser.add_argument('--pattern', default='data/historical_matches_*.json', 
                       help='Glob pattern for JSON files')
    parser.add_argument('--leagues', '-l', action='store_true',
                       help='Print league database and exit')
    parser.add_argument('--json', action='store_true',
                       help='Output league data as JSON')
    
    args = parser.parse_args()
    
    # Print league data if requested
    if args.leagues:
        print_leagues(args.json)
        return
    
    print("=" * 60)
    print("FOOTBALL PREDICTION MODEL REBUILD")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Determine which files to process
    if args.file:
        patterns = [args.file]
    elif args.date:
        patterns = [f"{args.dir}/historical_matches_{args.date}.json"]
    else:
        patterns = [args.pattern]
    
    # Build training data
    print("\n" + "-" * 60)
    print("BUILDING TRAINING DATA")
    print("-" * 60)
    
    builder = TrainingDataBuilder()
    training_data = builder.build_from_files(patterns)
    
    if not training_data:
        print("No training data found. Exiting.")
        sys.exit(1)
    
    # Save training data
    builder.save_training_data()
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_all(training_data)
    
    print(f"\nCompleted: {datetime.now().isoformat()}")


def print_leagues(as_json: bool = False):
    """Print league database information."""
    leagues_db = {}
    
    if os.path.exists(LEAGUES_DB_FILE):
        try:
            with open(LEAGUES_DB_FILE, 'r', encoding='utf-8') as f:
                leagues_db = json.load(f)
        except Exception as e:
            print(f"Error loading league database: {e}")
            return
    
    if not leagues_db:
        print("No leagues found in database.")
        return
    
    if as_json:
        # Print as JSON
        print(json.dumps(leagues_db, indent=2, ensure_ascii=False))
    else:
        # Print as formatted table
        print("=" * 80)
        print("LEAGUE DATABASE")
        print("=" * 80)
        print(f"\nTotal leagues: {len(leagues_db)}\n")
        
        # Sort by country then league name
        sorted_leagues = sorted(leagues_db.items(), key=lambda x: (x[1].get('country', ''), x[1].get('league', '')))
        
        print(f"{'Key':<15} {'Country':<20} {'League':<30} {'Matches'}")
        print("-" * 80)
        
        total_matches = 0
        for key, info in sorted_leagues:
            country = info.get('country', 'Unknown')[:20]
            league = info.get('league', 'Unknown')[:30]
            matches = info.get('match_count', 0)
            total_matches += matches
            print(f"{key:<15} {country:<20} {league:<30} {matches}")
        
        print("-" * 80)
        print(f"{'TOTAL':<66} {total_matches}")
        print("\n" + "=" * 80)
        
        # Group by country
        print("\n\nLEAGUES BY COUNTRY:")
        print("-" * 80)
        
        countries = {}
        for key, info in leagues_db.items():
            country = info.get('country', 'Unknown')
            if country not in countries:
                countries[country] = []
            countries[country].append(info.get('league', 'Unknown'))
        
        for country, leagues in sorted(countries.items()):
            print(f"\n{country}:")
            for league in sorted(set(leagues)):
                print(f"  - {league}")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
