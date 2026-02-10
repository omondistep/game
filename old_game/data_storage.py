#!/usr/bin/env python3
"""
Data Storage Module for Football Prediction System
Handles storing and retrieving match data for training.
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import pickle


class MatchDataStorage:
    """Storage system for match data and results."""

    def __init__(self, data_dir: str = "match_data"):
        self.data_dir = data_dir
        self.matches_file = os.path.join(data_dir, "matches.json")
        self.results_file = os.path.join(data_dir, "results.json")
        self.training_data_file = os.path.join(data_dir, "training_data.pkl")
        os.makedirs(data_dir, exist_ok=True)
        self._init_files()

    def _init_files(self):
        for path in (self.matches_file, self.results_file):
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
            self._update_training_data(url, result_data)
            return True
        except Exception as e:
            print(f"Error saving result: {e}")
            return False

    def save_match_with_result(self, match_data: Dict) -> bool:
        """Save match data that already contains actual_result."""
        try:
            url = match_data.get('url')
            if not url:
                return False
            
            # Save the match data
            self.save_match_data(match_data)
            
            # If actual_result is present, update training data
            actual_result = match_data.get('actual_result')
            if actual_result and actual_result.get('home_score') is not None:
                self._update_training_data(url, actual_result)
                return True
            
            return False
        except Exception as e:
            print(f"Error saving match with result: {e}")
            return False

    def _update_training_data(self, url: str, result_data: Dict):
        # Lazy import to avoid circular dependency
        from prediction_model import FootballPredictor
        
        matches = self._load_json(self.matches_file)
        match_data = next((m for m in matches if m.get('url') == url), None)
        if not match_data:
            return
        
        match_info = match_data.get('match_info', {})
        league = match_info.get('league', 'Unknown')
        
        features = FootballPredictor.extract_features(match_data)
        features['league'] = league  # Add league to features
        
        labels = {
            'result': result_data.get('result'),
            'home_score': result_data.get('home_score'),
            'away_score': result_data.get('away_score'),
            'total_goals': result_data.get('total_goals'),
            'over_under_2_5': result_data.get('over_under_2_5'),
        }
        
        training_data = self.get_training_data()
        
        # Group by league
        league_entry = next((t for t in training_data if t.get('league') == league), None)
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

    def get_league_training_data(self, league: str = None) -> List[Dict]:
        """Get training data for a specific league or all data."""
        training_data = self.get_training_data()
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
                return pickle.load(f)
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
