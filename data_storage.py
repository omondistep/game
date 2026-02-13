#!/usr/bin/env python3
"""
Data Storage Module for Football Prediction System
Handles storing and retrieving match data for training.
"""

import json
import os
import re
from typing import Dict, List, Optional
from datetime import datetime
import pickle


class MatchDataStorage:
    """Storage system for match data and results."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.matches_file = os.path.join(data_dir, "matches.json")
        self.results_file = os.path.join(data_dir, "results.json")
        self.predictions_file = os.path.join(data_dir, "predictions.json")
        self.training_data_file = os.path.join(data_dir, "training_data.pkl")
        os.makedirs(data_dir, exist_ok=True)
        self._init_files()

    def _init_files(self):
        for path in (self.matches_file, self.results_file, self.predictions_file):
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump([], f)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_match_data(self, match_data: Dict) -> bool:
        try:
            matches = self._load_json(self.matches_file)
            match_data['saved_at'] = datetime.now().isoformat()
            idx = next((i for i, m in enumerate(matches) if m.get('url') == match_data.get('url')), None)
            if idx is not None:
                matches[idx] = match_data
            else:
                matches.append(match_data)
            self._save_json(self.matches_file, matches)
            return True
        except Exception as e:
            print(f"Error saving match data: {e}")
            return False

    def save_match_result(self, url: str, result_data: Dict) -> bool:
        try:
            results = self._load_json(self.results_file)
            entry = {'url': url, 'result': result_data, 'saved_at': datetime.now().isoformat()}
            idx = next((i for i, r in enumerate(results) if r.get('url') == url), None)
            if idx is not None:
                results[idx] = entry
            else:
                results.append(entry)
            self._save_json(self.results_file, results)
            
            # Check for saved prediction and calculate correctness
            saved_prediction = self._get_saved_prediction(url)
            if saved_prediction:
                self._update_prediction_correctness(url, saved_prediction, result_data)
            
            self._update_training_data(url, result_data)
            return True
        except Exception as e:
            print(f"Error saving result: {e}")
            return False

    def save_prediction(self, url: str, prediction_data: Dict) -> bool:
        """Save a prediction for a match."""
        try:
            predictions = self._load_json(self.predictions_file)
            entry = {
                'url': url,
                'prediction': prediction_data,
                'saved_at': datetime.now().isoformat()
            }
            idx = next((i for i, p in enumerate(predictions) if p.get('url') == url), None)
            if idx is not None:
                predictions[idx] = entry
            else:
                predictions.append(entry)
            self._save_json(self.predictions_file, predictions)
            return True
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False

    def _get_saved_prediction(self, url: str) -> Optional[Dict]:
        """Get saved prediction for a URL."""
        predictions = self._load_json(self.predictions_file)
        return next((p for p in predictions if p.get('url') == url), None)

    def _update_prediction_correctness(self, url: str, prediction: Dict, actual_result: Dict):
        """Calculate if prediction was correct and update training features."""
        try:
            predicted_result = prediction.get('prediction', {}).get('result', {}).get('prediction')
            predicted_ou = prediction.get('prediction', {}).get('over_under', {}).get('prediction')
            actual_res = actual_result.get('result')
            actual_ou = actual_result.get('over_under_2_5')
            
            result_correct = predicted_result == actual_res if predicted_result and actual_res else None
            ou_correct = predicted_ou == actual_ou if predicted_ou and actual_ou else None
            
            print(f"  Prediction check: {predicted_result} vs {actual_res} ({'✓' if result_correct else '✗'})")
            print(f"  O/U check: {predicted_ou} vs {actual_ou} ({'✓' if ou_correct else '✗'})")
            
            # Store correctness for team-based features
            matches = self._load_json(self.matches_file)
            match_data = next((m for m in matches if m.get('url') == url), None)
            if match_data:
                teams = match_data.get('teams', {})
                home_team = teams.get('home', '')
                away_team = teams.get('away', '')
                
                # Update prediction history for teams
                self._update_team_prediction_history(home_team, result_correct, ou_correct, is_home=True)
                self._update_team_prediction_history(away_team, result_correct, ou_correct, is_home=False)
                
        except Exception as e:
            print(f"Error updating prediction correctness: {e}")

    def _update_team_prediction_history(self, team: str, result_correct: Optional[bool], 
                                         ou_correct: Optional[bool], is_home: bool):
        """Update prediction history for a team."""
        if not team:
            return
        
        history_file = os.path.join(self.data_dir, f"team_history_{hash(team) % 1000}.json")
        history = self._load_json(history_file)
        
        entry = {
            'team': team,
            'result_correct': result_correct,
            'ou_correct': ou_correct,
            'timestamp': datetime.now().isoformat(),
            'is_home': is_home
        }
        
        history.append(entry)
        # Keep only last 10 entries per team
        history = history[-10:]
        
        self._save_json(history_file, history)

    def save_match_with_result(self, match_data: Dict) -> bool:
        """Save match data that already contains actual_result."""
        try:
            # Ensure match_data is a dict
            if not isinstance(match_data, dict):
                print(f"Error: match_data is {type(match_data).__name__}, expected dict")
                return False
            
            url = match_data.get('url')
            if not url:
                return False
            
            # Save the match data
            self.save_match_data(match_data)
            
            # If actual_result is present, update training data
            actual_result = match_data.get('actual_result')
            if actual_result and isinstance(actual_result, dict) and actual_result.get('home_score') is not None:
                self._update_training_data(url, actual_result)
                return True
            
            return False
        except Exception as e:
            import traceback
            print(f"Error saving match with result: {e}")
            traceback.print_exc()
            return False

    def _update_training_data(self, url: str, result_data: Dict, prediction_correct: Dict = None):
        # Lazy import to avoid circular dependency
        from prediction_model import FootballPredictor
        
        matches = self._load_json(self.matches_file)
        match_data = next((m for m in matches if isinstance(m, dict) and m.get('url') == url), None)
        if not match_data:
            return
        
        match_info = match_data.get('match_info', {})
        if isinstance(match_info, str):
            match_info = {}
        league = match_info.get('league', 'Unknown') if isinstance(match_info, dict) else 'Unknown'
        
        features = FootballPredictor.extract_features(match_data)
        features['league'] = league  # Add league to features
        
        # Add prediction history features
        teams = match_data.get('teams', {})
        if isinstance(teams, str):
            teams = {}
        home_team = teams.get('home', '') if isinstance(teams, dict) else ''
        away_team = teams.get('away', '') if isinstance(teams, dict) else ''
        
        # Get team prediction history
        home_history = self._get_team_prediction_history(home_team)
        away_history = self._get_team_prediction_history(away_team)
        league_history = self._get_league_prediction_history(league)
        
        # Calculate accuracy percentages
        features['home_team_prediction_correct_pct'] = self._calc_accuracy(home_history, is_home=True)
        features['away_team_prediction_correct_pct'] = self._calc_accuracy(away_history, is_home=False)
        features['league_prediction_correct_pct'] = league_history.get('accuracy', 0.5) if isinstance(league_history, dict) else 0.5
        
        # Add overall model accuracy as feature
        features['overall_model_accuracy'] = self._get_overall_model_accuracy()
        
        labels = {
            'result': result_data.get('result'),
            'home_score': result_data.get('home_score'),
            'away_score': result_data.get('away_score'),
            'total_goals': result_data.get('total_goals'),
            'over_under_2_5': result_data.get('over_under_2_5'),
        }
        
        training_data = self.get_training_data()
        
        # Group by league
        league_entry = next((t for t in training_data if isinstance(t, dict) and t.get('league') == league), None)
        if not league_entry:
            league_entry = {
                'league': league,
                'examples': [],
            }
            training_data.append(league_entry)
        
        league_entry['examples'].append({
            'features': features,
            'labels': labels,
            'url': url,
            'timestamp': datetime.now().isoformat(),
        })
        
        with open(self.training_data_file, 'wb') as f:
            pickle.dump(training_data, f)
    
    def _get_team_prediction_history(self, team: str) -> List[Dict]:
        """Get prediction history for a team."""
        if not team:
            return []
        history_file = os.path.join(self.data_dir, f"team_history_{hash(team) % 1000}.json")
        return self._load_json(history_file)
    
    def _get_league_prediction_history(self, league: str) -> Dict:
        """Get overall prediction accuracy for a league."""
        training = self.get_training_data()
        for entry in training:
            if not isinstance(entry, dict):
                continue
            if entry.get('league') == league:
                examples = entry.get('examples', [])
                if len(examples) < 3:
                    return {'accuracy': 0.5, 'count': len(examples)}
                # Calculate accuracy from examples with labels
                correct = sum(1 for ex in examples if isinstance(ex, dict) and 
                             ex.get('labels', {}).get('result') == 
                             ex.get('features', {}).get('_last_prediction_result'))
                return {
                    'accuracy': correct / len(examples) if examples else 0.5,
                    'count': len(examples)
                }
        return {'accuracy': 0.5, 'count': 0}
    
    def _calc_accuracy(self, history: List[Dict], is_home: bool) -> float:
        """Calculate prediction accuracy from history."""
        if not history:
            return 0.5  # Default neutral
        
        # Filter to only dict items and correct is_home
        filtered = [h for h in history if isinstance(h, dict) and h.get('is_home') == is_home]
        if len(filtered) < 2:
            return 0.5
        
        correct = sum(1 for h in filtered if h.get('result_correct') is True)
        return correct / len(filtered)
    
    def _get_overall_model_accuracy(self) -> float:
        """Get overall model accuracy from all predictions with results."""
        results = self._load_json(self.results_file)
        predictions = self._load_json(self.predictions_file)
        
        if len(results) < 5:
            return 0.5
        
        correct = 0
        total = 0
        for r in results:
            url = r.get('url')
            actual = r.get('result', {}).get('result')
            pred = next((p for p in predictions if p.get('url') == url), None)
            if pred and actual:
                predicted = pred.get('prediction', {}).get('result', {}).get('prediction')
                if predicted:
                    total += 1
                    if predicted == actual:
                        correct += 1
        
        return correct / total if total > 0 else 0.5

    def get_league_training_data(self, league: str = None) -> List[Dict]:
        """Get training data for a specific league or all data."""
        training_data = self.get_training_data()
        
        # Handle new dictionary format
        if isinstance(training_data, dict):
            if not league:
                # Return all examples flattened
                all_examples = []
                for league_data in training_data.values():
                    examples = league_data.get('examples', [])
                    all_examples.extend(examples)
                return all_examples
            
            # Return specific league - try exact match first
            for league_key, league_data in training_data.items():
                info = league_data.get('league_info', {})
                if info.get('league') == league or info.get('league_code') == league:
                    return league_data.get('examples', [])
            return []
        
        # Handle old list format
        if not league:
            # Return all examples flattened
            all_examples = []
            for entry in training_data:
                all_examples.extend(entry.get('examples', []))
            return all_examples
        
        # Return specific league
        for entry in training_data:
            if entry.get('league') == league:
                return entry.get('examples', [])
        return []

    def get_leagues(self) -> List[str]:
        """Get list of leagues with training data."""
        training_data = self.get_training_data()
        return [entry.get('league') for entry in training_data if entry.get('league')]
    
    def update_league_name(self, old_name: str, new_name: str) -> int:
        """Update league name in all data files. Returns total number of updates."""
        total_updates = 0
        
        # 1. Update training_data.pkl
        try:
            training_data = self.get_training_data()
            for entry in training_data:
                if entry.get('league') == old_name:
                    entry['league'] = new_name
                    total_updates += 1
            if total_updates > 0:
                with open(self.training_data_file, 'wb') as f:
                    pickle.dump(training_data, f)
                print(f"  Updated {total_matches} entries in training_data.pkl")
        except Exception as e:
            print(f"  Error updating training_data.pkl: {e}")
        
        # 2. Update matches.json
        try:
            matches = self._load_json(self.matches_file)
            match_count = 0
            for match in matches:
                info = match.get('match_info', {})
                if info.get('league') == old_name:
                    info['league'] = new_name
                    match_count += 1
            if match_count > 0:
                self._save_json(self.matches_file, matches)
                print(f"  Updated {match_count} entries in matches.json")
                total_updates += match_count
        except Exception as e:
            print(f"  Error updating matches.json: {e}")
        
        # 3. Update results.json
        try:
            results = self._load_json(self.results_file)
            result_count = 0
            for result in results:
                info = result.get('match_info', {})
                if info.get('league') == old_name:
                    info['league'] = new_name
                    result_count += 1
            if result_count > 0:
                self._save_json(self.results_file, results)
                print(f"  Updated {result_count} entries in results.json")
                total_updates += result_count
        except Exception as e:
            print(f"  Error updating results.json: {e}")
        
        # 4. Update predictions.json
        try:
            predictions = self._load_json(self.predictions_file)
            pred_count = 0
            for pred in predictions:
                md = pred.get('match_data', {})
                info = md.get('match_info', {})
                if info.get('league') == old_name:
                    info['league'] = new_name
                    pred_count += 1
            if pred_count > 0:
                self._save_json(self.predictions_file, predictions)
                print(f"  Updated {pred_count} entries in predictions.json")
                total_updates += pred_count
        except Exception as e:
            print(f"  Error updating predictions.json: {e}")
        
        return total_updates

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_all_matches(self) -> List[Dict]:
        return self._load_json(self.matches_file)

    def get_all_results(self) -> List[Dict]:
        return self._load_json(self.results_file)

    def get_training_data(self) -> List[Dict]:
        if os.path.exists(self.training_data_file):
            with open(self.training_data_file, 'rb') as f:
                data = pickle.load(f)
            # Handle both dict and list formats
            if isinstance(data, dict):
                # Convert dict format to list format
                result = []
                for league_code, entry in data.items():
                    if isinstance(entry, dict):
                        entry_copy = entry.copy()
                        entry_copy['league_code'] = league_code
                        if 'league' not in entry_copy and 'league_info' in entry_copy:
                            entry_copy['league'] = entry_copy.get('league_info', {}).get('name', league_code)
                        result.append(entry_copy)
                return result
            return data
        return []

    def get_match_by_url(self, url: str) -> Optional[Dict]:
        return next((m for m in self.get_all_matches() if m.get('url') == url), None)

    def get_statistics(self) -> Dict:
        matches = self.get_all_matches()
        results = self.get_all_results()
        training = self.get_training_data()
        result_urls = {r['url'] for r in results}
        # Count total examples across all leagues
        total_examples = sum(len(entry.get('examples', [])) for entry in training)
        return {
            'total_matches': len(matches),
            'total_results': len(results),
            'training_examples': total_examples,
            'matches_with_results': sum(1 for m in matches if m['url'] in result_urls),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(path: str) -> list:
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, Exception):
            pass
        return []

    @staticmethod
    def _save_json(path: str, data):
        # Atomic write with temp file
        temp_path = path + '.tmp'
        try:
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            # Atomic rename
            os.replace(temp_path, path)
        except Exception:
            # Clean up temp file if exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
