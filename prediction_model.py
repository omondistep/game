#!/usr/bin/env python3
"""
Football Match Prediction Model
Uses scraped data and machine learning to predict match outcomes.
Supports both odds-based fallback and trained ML models.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import pickle
import os
import json
from datetime import datetime
import time


# =============================================================================
# WEIGHTED PREDICTION MODEL
# Based on research and best practices for football prediction
# =============================================================================

class WeightedPredictor:
    """
    Enhanced prediction model using weighted factors based on data-driven weights.
    
    Weights are calculated from historical data analysis:
    - League-specific weights when enough data exists (20+ matches)
    - Global weights as fallback
    
    Default weights (when no calculated weights available):
    - Recent Form: 38% (Most predictive based on data analysis)
    - Odds Analysis: 37% (Market wisdom)
    - Goals Stats: 21% (Attacking and defensive strength)
    - Other factors: 1% each (when data available)
    """
    
    # Default weight configuration (fallback)
    DEFAULT_WEIGHTS = {
        'league_position': 0.01,
        'odds_analysis': 0.37,
        'recent_form': 0.38,
        'top5_performance': 0.01,
        'goals_stats': 0.21,
        'h2h': 0.01,
        'shots_possession': 0.01,
    }
    
    def __init__(self):
        self.weights = self._load_weights()
    
    def _load_weights(self) -> Dict:
        """Load calculated weights from file."""
        weights_file = 'data/factor_weights.json'
        if os.path.exists(weights_file):
            try:
                with open(weights_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {'global': {'weights': self.DEFAULT_WEIGHTS}, 'leagues': {}}
    
    def get_weights(self, league_code: str = None) -> Dict[str, float]:
        """Get weights for a specific league or global weights."""
        if league_code and league_code in self.weights.get('leagues', {}):
            return self.weights['leagues'][league_code].get('weights', self.DEFAULT_WEIGHTS)
        return self.weights.get('global', {}).get('weights', self.DEFAULT_WEIGHTS)
    
    def predict(self, match_data: Dict, features: Dict, league_code: str = None) -> Dict:
        """
        Generate weighted prediction based on all factors.
        
        Args:
            match_data: Raw scraped match data
            features: Extracted features dictionary
            league_code: League code for league-specific weights
            
        Returns:
            Dictionary with prediction, probabilities, and detailed analysis
        """
        # Get data-driven weights for this league
        weights = self.get_weights(league_code)
        
        # Calculate factor scores
        factor_scores = self._calculate_factor_scores(match_data, features, weights)
        
        # Compute final probabilities
        result_probs = self._compute_result_probabilities(factor_scores, features)
        ou_probs = self._compute_over_under_probabilities(features)
        
        # Determine predictions
        result_pred = max(result_probs, key=result_probs.get)
        ou_pred = 'Over' if ou_probs.get('Over', 0) > ou_probs.get('Under', 0) else 'Under'
        
        # Calculate confidence
        confidence = max(result_probs.values())
        
        return {
            'result': {
                'prediction': result_pred,
                'probabilities': {k: round(v, 4) for k, v in result_probs.items()},
                'computed_odds': {k: round(1/v, 2) if v > 0.01 else 99.0 for k, v in result_probs.items()},
                'confidence': round(confidence, 4),
            },
            'over_under': {
                'prediction': ou_pred,
                'probabilities': {k: round(v, 4) for k, v in ou_probs.items()},
                'computed_odds': {k: round(1/v, 2) if v > 0.01 else 99.0 for k, v in ou_probs.items()},
                'confidence': round(max(ou_probs.values()), 4),
            },
            'factor_analysis': factor_scores,
            'key_factors': self._extract_key_factors(factor_scores, features),
            'prediction_method': 'weighted_factors',
            'weights_used': weights,
        }
    
    def _calculate_factor_scores(self, match_data: Dict, features: Dict, weights: Dict = None) -> Dict:
        """Calculate individual factor scores for each team using data-driven weights."""
        
        if weights is None:
            weights = self.DEFAULT_WEIGHTS
        
        # 1. LEAGUE POSITION (weight from data)
        # Lower position = better team. Score from 0 to 1 where 1 is best.
        home_pos = features.get('home_position', 10) or 10
        away_pos = features.get('away_position', 10) or 10
        
        # Position score: position 1 = 1.0, position 20 = 0.1
        max_pos = 20
        pos_score_home = max(0, 1 - (home_pos - 1) / (max_pos - 1))
        pos_score_away = max(0, 1 - (away_pos - 1) / (max_pos - 1))
        
        # 2. ODDS ANALYSIS (15% weight)
        # Use market odds to derive probabilities
        odds_home = features.get('odds_home') or 2.5
        odds_draw = features.get('odds_draw') or 3.0
        odds_away = features.get('odds_away') or 3.5
        
        # Implied probabilities
        imp_home = 1 / odds_home
        imp_draw = 1 / odds_draw
        imp_away = 1 / odds_away
        total = imp_home + imp_draw + imp_away
        
        odds_score_home = imp_home / total
        odds_score_away = imp_away / total
        
        # 3. RECENT FORM (20% weight)
        # Form points: W=3, D=1, L=0, normalized
        form = match_data.get('form', {})
        home_form = form.get('home', [])
        away_form = form.get('away', [])
        
        form_points_home = sum({'W': 3, 'D': 1, 'L': 0}.get(r, 0) for r in home_form[:6])
        form_points_away = sum({'W': 3, 'D': 1, 'L': 0}.get(r, 0) for r in away_form[:6])
        max_form = 18
        
        form_score_home = form_points_home / max_form
        form_score_away = form_points_away / max_form
        
        # 4. TOP 5 PERFORMANCE (10% weight)
        # Performance against league's top 5 teams
        top5_perf = self._calculate_top5_performance(match_data, features)
        
        # 5. GOALS STATS (15% weight)
        # Goals scored vs conceded
        gs = match_data.get('goals_stats', {})
        hg = gs.get('home', {})
        ag = gs.get('away', {})
        
        home_scored = features.get('home_scored_avg') or 1.5
        home_conceded = features.get('home_conceded_avg') or 1.2
        away_scored = features.get('away_scored_avg') or 1.3
        away_conceded = features.get('away_conceded_avg') or 1.4
        
        # Goal difference per game
        home_gd = home_scored - home_conceded
        away_gd = away_scored - away_conceded
        
        # Normalize to [-1, 1], then to [0, 1]
        max_gd = 2.0
        goals_score_home = (home_gd / max_gd + 1) / 2
        goals_score_away = (away_gd / max_gd + 1) / 2
        
        # 6. HEAD-TO-HEAD (10% weight)
        h2h = match_data.get('head_to_head', {})
        h2h_sum = h2h.get('summary', {})
        
        h2h_home_wins = h2h_sum.get('home_win_pct', 33) or 33
        h2h_away_wins = h2h_sum.get('away_win_pct', 33) or 33
        h2h_draws = h2h_sum.get('draw_pct', 33) or 33
        
        h2h_score_home = h2h_home_wins / 100
        h2h_score_away = h2h_away_wins / 100
        
        # 7. SHOTS & POSSESSION (10% weight)
        js = match_data.get('js_detailed_stats', {})
        hs_stats = js.get('home_stats', {})
        as_stats = js.get('away_stats', {})
        
        home_possession = features.get('home_possession') or 50
        away_possession = features.get('away_possession') or 50
        
        home_shots = features.get('home_shots_avg') or 14
        away_shots = features.get('away_shots_avg') or 12
        
        # Normalize shots (typical range 10-20)
        shots_score_home = min(1, max(0, (home_shots - 8) / 12))
        shots_score_away = min(1, max(0, (away_shots - 8) / 12))
        
        # Possession advantage
        poss_score_home = home_possession / 100
        poss_score_away = away_possession / 100
        
        shots_possession_home = (shots_score_home * 0.4 + poss_score_home * 0.6)
        shots_possession_away = (shots_score_away * 0.4 + poss_score_away * 0.6)
        
        return {
            'league_position': {'home': pos_score_home, 'away': pos_score_away, 'weight': weights.get('league_position', 0.01)},
            'odds_analysis': {'home': odds_score_home, 'away': odds_score_away, 'weight': weights.get('odds_analysis', 0.37)},
            'recent_form': {'home': form_score_home, 'away': form_score_away, 'weight': weights.get('recent_form', 0.38)},
            'top5_performance': {'home': top5_perf.get('home', 0.5), 'away': top5_perf.get('away', 0.5), 'weight': weights.get('top5_performance', 0.01)},
            'goals_stats': {'home': goals_score_home, 'away': goals_score_away, 'weight': weights.get('goals_stats', 0.21)},
            'h2h': {'home': h2h_score_home, 'away': h2h_score_away, 'weight': weights.get('h2h', 0.01)},
            'shots_possession': {'home': shots_possession_home, 'away': shots_possession_away, 'weight': weights.get('shots_possession', 0.01)},
            # Additional data for analysis
            'h2h_draws': h2h_draws / 100,
            'expected_goals': features.get('expected_total_goals', 2.5),
        }
    
    def _calculate_top5_performance(self, match_data: Dict, features: Dict) -> Dict:
        """Calculate performance against top 5 league teams."""
        # Extract top 5 teams from league table
        league_table = match_data.get('league_table', [])
        top5_teams = [entry.get('team', '') for entry in league_table[:5]]
        
        home_team = match_data.get('teams', {}).get('home', '')
        away_team = match_data.get('teams', {}).get('away', '')
        
        # Get home/away matches
        home_matches = match_data.get('home_matches', [])
        away_matches = match_data.get('away_matches', [])
        
        def calc_perf(teams_matches: List[Dict], current_team: str, is_home: bool) -> float:
            if not teams_matches:
                return 0.5  # Default neutral
            
            top5_wins = 0
            top5_matches = 0
            
            for m in teams_matches[:6]:  # Last 6 matches
                opponent = m.get('away_team', '') if is_home else m.get('home_team', '')
                
                # Check if opponent is in top 5
                is_top5 = any(t.lower() in opponent.lower() or opponent.lower() in t.lower() 
                            for t in top5_teams if t)
                
                if is_top5:
                    top5_matches += 1
                    home_score = m.get('home_score', 0)
                    away_score = m.get('away_score', 0)
                    
                    if is_home:
                        if home_score > away_score:
                            top5_wins += 1
                    else:
                        if away_score > home_score:
                            top5_wins += 1
            
            return top5_wins / top5_matches if top5_matches > 0 else 0.5
        
        return {
            'home': calc_perf(home_matches, home_team, True),
            'away': calc_perf(away_matches, away_team, False),
        }
    
    def _compute_result_probabilities(self, factor_scores: Dict, features: Dict) -> Dict:
        """Compute final result probabilities using weighted factors."""
        
        # Weighted scores
        home_score = 0.0
        away_score = 0.0
        
        for factor, scores in factor_scores.items():
            if isinstance(scores, dict) and 'home' in scores and 'away' in scores:
                weight = scores.get('weight', 0)
                home_score += scores['home'] * weight
                away_score += scores['away'] * weight
        
        # Draw probability based on:
        # - Similar weighted scores
        # - H2H draw history
        # - Expected goals close to 2.0-2.5
        score_diff = abs(home_score - away_score)
        
        # Base draw probability inversely related to score difference
        base_draw = max(0.15, 0.35 - score_diff * 0.3)
        
        # Adjust based on H2H draws
        h2h_draws = factor_scores.get('h2h_draws', 0.25)
        h2h_adjust = (h2h_draws - 0.25) * 0.2
        draw_prob = max(0.10, min(0.50, base_draw + h2h_adjust))
        
        # Adjust based on expected goals (low-scoring games = more draws)
        exp_goals = factor_scores.get('expected_goals', 2.5)
        if exp_goals < 2.0:
            draw_prob += 0.05
        elif exp_goals > 3.0:
            draw_prob -= 0.05
        
        # Normalize to probabilities
        total_home = home_score + away_score + draw_prob
        
        p1 = home_score / total_home
        p2 = away_score / total_home
        px = draw_prob / total_home
        
        return {'1': max(0.02, min(0.95, p1)), 'X': max(0.02, min(0.95, px)), '2': max(0.02, min(0.95, p2))}
    
    def _compute_over_under_probabilities(self, features: Dict) -> Dict:
        """Compute Over/Under 2.5 goals probabilities."""
        
        exp_goals = features.get('expected_total_goals', 2.5)
        
        # Use Poisson-like distribution
        # Probability of > 2.5 goals
        if exp_goals >= 3.5:
            po = 0.65
            pu = 0.35
        elif exp_goals >= 3.0:
            po = 0.58
            pu = 0.42
        elif exp_goals >= 2.5:
            po = 0.52
            pu = 0.48
        elif exp_goals >= 2.0:
            po = 0.42
            pu = 0.58
        elif exp_goals >= 1.5:
            po = 0.32
            pu = 0.68
        else:
            po = 0.22
            pu = 0.78
        
        return {'Over': po, 'Under': pu}
    
    def _extract_key_factors(self, factor_scores: Dict, features: Dict) -> Dict:
        """Extract and summarize key factors supporting the prediction."""
        
        key_factors = {
            'match_result': [],
            'over_under': [],
        }
        
        # Identify strongest factors for match result
        factor_impacts = []
        for factor, scores in factor_scores.items():
            if isinstance(scores, dict) and 'home' in scores and 'away' in scores:
                diff = scores['home'] - scores['away']
                impact = abs(diff) * scores.get('weight', 0)
                factor_impacts.append((factor, diff, impact))
        
        # Sort by impact
        factor_impacts.sort(key=lambda x: x[2], reverse=True)
        
        # Add top factors
        for factor, diff, impact in factor_impacts[:4]:
            if impact > 0.02:
                direction = 'home' if diff > 0 else 'away'
                key_factors['match_result'].append({
                    'factor': factor,
                    'direction': direction,
                    'impact': round(impact * 100, 1),
                })
        
        # Key factors for Over/Under
        exp_goals = factor_scores.get('expected_goals', 2.5)
        if exp_goals >= 2.5:
            key_factors['over_under'].append({
                'factor': 'expected_goals',
                'value': round(exp_goals, 2),
                'direction': 'Over',
            })
        else:
            key_factors['over_under'].append({
                'factor': 'expected_goals',
                'value': round(exp_goals, 2),
                'direction': 'Under',
            })
        
        # Add goal difference factor
        gs = (features.get('home_scored_avg') or 1.5) + (features.get('away_scored_avg') or 1.3)
        key_factors['over_under'].append({
            'factor': 'combined_scoring',
            'value': round(gs, 2),
            'direction': 'Over' if gs > 2.5 else 'Under',
        })
        
        return key_factors


class FootballPredictor:
    """Machine learning model for football match prediction."""

    FEATURE_NAMES = [
        # Standings
        'home_position', 'away_position', 'position_diff',
        'home_points', 'away_points',
        # Form (last 6)
        'home_form_points', 'away_form_points',
        'home_recent_form', 'away_recent_form',
        'home_wins_l6', 'home_draws_l6', 'home_losses_l6',
        'away_wins_l6', 'away_draws_l6', 'away_losses_l6',
        # Goals
        'home_scored_avg', 'home_conceded_avg',
        'away_scored_avg', 'away_conceded_avg',
        'expected_total_goals',
        # Home / Away specific
        'home_home_win_rate', 'away_away_win_rate',
        'home_home_goals_avg', 'away_away_goals_avg',
        # Head-to-head
        'h2h_home_win_pct', 'h2h_draw_pct', 'h2h_away_win_pct',
        # Shots & attacks
        'home_shots_avg', 'away_shots_avg',
        'home_shots_on_target_pct', 'away_shots_on_target_pct',
        'home_dangerous_attacks_avg', 'away_dangerous_attacks_avg',
        # Possession & passes
        'home_possession', 'away_possession',
        'home_pass_accuracy', 'away_pass_accuracy',
        # Discipline
        'home_fouls_avg', 'away_fouls_avg',
        'home_yellow_avg', 'away_yellow_avg',
        # Odds (market expectations)
        'odds_home', 'odds_draw', 'odds_away',
        'odds_over', 'odds_under',
        # Forebet
        'forebet_probability',
        # Prediction History (meta-features from model learning)
        'home_team_prediction_correct_pct',
        'away_team_prediction_correct_pct',
        'league_prediction_correct_pct',
        'overall_model_accuracy',
    ]

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.result_model: Optional[RandomForestClassifier] = None
        self.ou_model: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self._load_models()
        
        # League-specific models
        self.league_models: Dict[str, Dict] = {}
        self._load_league_models()
        
        # Prediction feedback tracking
        self.prediction_feedback: Dict[str, Dict] = {}
        self._load_prediction_feedback()

    def _load_models(self):
        paths = {
            'result': os.path.join(self.model_dir, 'result_model.pkl'),
            'ou': os.path.join(self.model_dir, 'ou_model.pkl'),
            'scaler': os.path.join(self.model_dir, 'scaler.pkl'),
        }
        try:
            if os.path.exists(paths['result']):
                with open(paths['result'], 'rb') as f:
                    self.result_model = pickle.load(f)
            if os.path.exists(paths['ou']):
                with open(paths['ou'], 'rb') as f:
                    self.ou_model = pickle.load(f)
            if os.path.exists(paths['scaler']):
                with open(paths['scaler'], 'rb') as f:
                    self.scaler = pickle.load(f)
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def _load_league_models(self):
        """Load league-specific models from models/ directory.
        
        Models are stored in models/{Country}_{League_Name}/ format.
        Also loads league database for matching.
        """
        # Load from root models directory (where rebuild_data.py saves them)
        try:
            for entry in os.listdir(self.model_dir):
                league_path = os.path.join(self.model_dir, entry)
                if os.path.isdir(league_path) and not entry.startswith('.'):
                    models = {}
                    for fname in ['result_model.pkl', 'ou_model.pkl', 'scaler.pkl']:
                        fpath = os.path.join(league_path, fname)
                        if os.path.exists(fpath):
                            with open(fpath, 'rb') as f:
                                models[fname.replace('.pkl', '')] = pickle.load(f)
                    if models:
                        # Store with multiple keys for flexible lookup
                        self.league_models[entry] = models
                        # Also store with normalized key (lowercase, underscores)
                        normalized_key = entry.lower().replace(' ', '_').replace('-', '_')
                        if normalized_key != entry:
                            self.league_models[normalized_key] = models
        except Exception as e:
            print(f"Error loading league models: {e}")
        
        # Load leagues database for additional matching
        self.leagues_db = {}
        try:
            db_path = 'data/leagues_db.json'
            if os.path.exists(db_path):
                with open(db_path, 'r') as f:
                    self.leagues_db = json.load(f)
        except Exception as e:
            print(f"Error loading leagues database: {e}")
    
    def _load_prediction_feedback(self):
        """Load prediction feedback data."""
        feedback_file = os.path.join(self.model_dir, "prediction_feedback.json")
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r') as f:
                    self.prediction_feedback = json.load(f)
                    print(f"Loaded {len(self.prediction_feedback)} feedback entries")
            except:
                self.prediction_feedback = {}
    
    def save_prediction_feedback(self, url: str, prediction: Dict, actual_result: Dict):
        """Save prediction for later feedback comparison."""
        self.prediction_feedback[url] = {
            'prediction': prediction,
            'actual': actual_result,
            'timestamp': time.time()
        }
        # Save periodically
        if len(self.prediction_feedback) % 10 == 0:
            self._save_prediction_feedback()
    
    def _save_prediction_feedback(self):
        """Save prediction feedback to file."""
        feedback_file = os.path.join(self.model_dir, "prediction_feedback.json")
        with open(feedback_file, 'w') as f:
            json.dump(self.prediction_feedback, f, indent=2)
    
    def get_prediction_accuracy(self, league: str = None) -> Dict:
        """Calculate prediction accuracy from feedback."""
        correct = 0
        total = 0
        for url, feedback in self.prediction_feedback.items():
            pred = feedback.get('prediction', {}).get('result', {}).get('prediction')
            actual = feedback.get('actual', {}).get('result')
            if pred and actual:
                # Convert actual result to prediction format
                if actual.get('home_score') > actual.get('away_score'):
                    actual_pred = '1'
                elif actual.get('home_score') < actual.get('away_score'):
                    actual_pred = '2'
                else:
                    actual_pred = 'X'
                
                if pred == actual_pred:
                    correct += 1
                total += 1
        
        return {'correct': correct, 'total': total, 'accuracy': correct/total if total > 0 else 0}

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def _load_models(self):
        paths = {
            'result': os.path.join(self.model_dir, 'result_model.pkl'),
            'ou': os.path.join(self.model_dir, 'ou_model.pkl'),
            'scaler': os.path.join(self.model_dir, 'scaler.pkl'),
        }
        try:
            if os.path.exists(paths['result']):
                with open(paths['result'], 'rb') as f:
                    self.result_model = pickle.load(f)
            if os.path.exists(paths['ou']):
                with open(paths['ou'], 'rb') as f:
                    self.ou_model = pickle.load(f)
            if os.path.exists(paths['scaler']):
                with open(paths['scaler'], 'rb') as f:
                    self.scaler = pickle.load(f)
        except Exception as e:
            print(f"Error loading models: {e}")

    def _save_models(self):
        try:
            if self.result_model:
                with open(os.path.join(self.model_dir, 'result_model.pkl'), 'wb') as f:
                    pickle.dump(self.result_model, f)
            if self.ou_model:
                with open(os.path.join(self.model_dir, 'ou_model.pkl'), 'wb') as f:
                    pickle.dump(self.ou_model, f)
            if self.scaler:
                with open(os.path.join(self.model_dir, 'scaler.pkl'), 'wb') as f:
                    pickle.dump(self.scaler, f)
        except Exception as e:
            print(f"Error saving models: {e}")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def extract_features(match_data: Dict) -> Dict:
        """Build a flat feature dictionary from raw scraped match data."""
        f: Dict = {}
        
        def _safe_float(val, default=None):
            """Safely convert value to float."""
            if val is None:
                return default
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                try:
                    return float(val.replace(',', '.'))
                except ValueError:
                    return default
            return default
        
        def _safe_int(val, default=None):
            """Safely convert value to int."""
            if val is None:
                return default
            if isinstance(val, int):
                return val
            if isinstance(val, float):
                return int(val)
            if isinstance(val, str):
                try:
                    return int(float(val.replace(',', '.')))
                except ValueError:
                    return default
            return default
        
        # --- Standings ---
        st = match_data.get('standings', {})
        f['home_position'] = _safe_int(st.get('home'))
        f['away_position'] = _safe_int(st.get('away'))
        hp = f['home_position'] or 10
        ap = f['away_position'] or 10
        f['position_diff'] = hp - ap

        # Points from league table
        lt = match_data.get('league_table', [])
        teams = match_data.get('teams', {})
        for entry in lt:
            if entry.get('team') == teams.get('home'):
                f['home_points'] = entry.get('points')
            elif entry.get('team') == teams.get('away'):
                f['away_points'] = entry.get('points')

        # --- Form ---
        form = match_data.get('form', {})
        form_map = {'W': 3, 'D': 1, 'L': 0}
        hf = form.get('home', [])
        af = form.get('away', [])
        f['home_form_points'] = sum(form_map.get(x, 0) for x in hf)
        f['away_form_points'] = sum(form_map.get(x, 0) for x in af)
        f['home_recent_form'] = sum(form_map.get(x, 0) for x in hf[:3])
        f['away_recent_form'] = sum(form_map.get(x, 0) for x in af[:3])
        f['home_wins_l6'] = hf.count('W')
        f['home_draws_l6'] = hf.count('D')
        f['home_losses_l6'] = hf.count('L')
        f['away_wins_l6'] = af.count('W')
        f['away_draws_l6'] = af.count('D')
        f['away_losses_l6'] = af.count('L')

        # --- Goals ---
        gs = match_data.get('goals_stats', {})
        hg = gs.get('home', {})
        ag = gs.get('away', {})
        f['home_scored_avg'] = _safe_float(hg.get('scored_avg'))
        f['home_conceded_avg'] = _safe_float(hg.get('conceded_avg'))
        f['away_scored_avg'] = _safe_float(ag.get('scored_avg'))
        f['away_conceded_avg'] = _safe_float(ag.get('conceded_avg'))
        hs = f['home_scored_avg'] or 1.3
        ac = f['away_conceded_avg'] or 1.2
        as_ = f['away_scored_avg'] or 1.1
        hc = f['home_conceded_avg'] or 1.0
        f['expected_total_goals'] = (hs + ac) / 2 + (as_ + hc) / 2

        # --- Home / Away specific ---
        hm = match_data.get('home_matches', [])
        am = match_data.get('away_matches', [])
        if hm:
            home_team = teams.get('home', '')
            hw = sum(1 for m in hm if
                     (m.get('home_team', '') == home_team and (_safe_int(m.get('home_score')) or 0) > (_safe_int(m.get('away_score')) or 0)) or
                     (m.get('away_team', '') == home_team and (_safe_int(m.get('away_score')) or 0) > (_safe_int(m.get('home_score')) or 0)))
            f['home_home_win_rate'] = hw / len(hm) * 100 if hm else 50
            hg_total = sum((_safe_int(m.get('home_score')) or 0) if m.get('home_team', '') == home_team else (_safe_int(m.get('away_score')) or 0) for m in hm)
            f['home_home_goals_avg'] = hg_total / len(hm) if hm else 1.5
        else:
            f['home_home_win_rate'] = None
            f['home_home_goals_avg'] = None

        if am:
            away_team = teams.get('away', '')
            aw = sum(1 for m in am if
                     (m.get('away_team', '') == away_team and (_safe_int(m.get('away_score')) or 0) > (_safe_int(m.get('home_score')) or 0)) or
                     (m.get('home_team', '') == away_team and (_safe_int(m.get('home_score')) or 0) > (_safe_int(m.get('away_score')) or 0)))
            f['away_away_win_rate'] = aw / len(am) * 100 if am else 33
            ag_total = sum((_safe_int(m.get('away_score')) or 0) if m.get('away_team', '') == away_team else (_safe_int(m.get('home_score')) or 0) for m in am)
            f['away_away_goals_avg'] = ag_total / len(am) if am else 1.0
        else:
            f['away_away_win_rate'] = None
            f['away_away_goals_avg'] = None

        # --- Head-to-head ---
        h2h = match_data.get('head_to_head', {})
        h2h_sum = h2h.get('summary', {})
        f['h2h_home_win_pct'] = _safe_float(h2h_sum.get('home_win_pct'))
        f['h2h_draw_pct'] = _safe_float(h2h_sum.get('draw_pct'))
        f['h2h_away_win_pct'] = _safe_float(h2h_sum.get('away_win_pct'))

        # --- Shots & attacks ---
        js = match_data.get('js_detailed_stats', {})
        hs_stats = js.get('home_stats', {})
        as_stats = js.get('away_stats', {})

        hgames = _safe_int(gs.get('home', {}).get('games')) or 33
        agames = _safe_int(gs.get('away', {}).get('games')) or 24

        def _arr_avg(arr, games=None):
            if isinstance(arr, list) and len(arr) >= 1:
                total = _safe_float(arr[0])
                g = games or 1
                return total / g if g and total is not None else None
            return None

        f['home_shots_avg'] = _arr_avg(hs_stats.get('shots_total'), hgames)
        f['away_shots_avg'] = _arr_avg(as_stats.get('shots_total'), agames)

        st_h = _safe_float(hs_stats.get('shots_on_target', [0])[0]) if isinstance(hs_stats.get('shots_on_target'), list) else 0
        st_t = _safe_float(hs_stats.get('shots_total', [1])[0]) if isinstance(hs_stats.get('shots_total'), list) else 1
        f['home_shots_on_target_pct'] = (st_h / st_t * 100) if st_t and st_h is not None else None

        st_a = _safe_float(as_stats.get('shots_on_target', [0])[0]) if isinstance(as_stats.get('shots_on_target'), list) else 0
        st_at = _safe_float(as_stats.get('shots_total', [1])[0]) if isinstance(as_stats.get('shots_total'), list) else 1
        f['away_shots_on_target_pct'] = (st_a / st_at * 100) if st_at and st_a is not None else None

        f['home_dangerous_attacks_avg'] = _arr_avg(hs_stats.get('dan_attacks'), hgames)
        f['away_dangerous_attacks_avg'] = _arr_avg(as_stats.get('dan_attacks'), agames)

        # --- Possession & passes ---
        bp_h = hs_stats.get('ball_poss')
        bp_a = as_stats.get('ball_poss')
        f['home_possession'] = _safe_float(bp_h[0]) if isinstance(bp_h, list) and bp_h else None
        f['away_possession'] = _safe_float(bp_a[0]) if isinstance(bp_a, list) and bp_a else None

        pa_h = _safe_float(hs_stats.get('passes_accurate', [0])[0]) if isinstance(hs_stats.get('passes_accurate'), list) else 0
        pt_h = _safe_float(hs_stats.get('passes_total', [1])[0]) if isinstance(hs_stats.get('passes_total'), list) else 1
        f['home_pass_accuracy'] = (pa_h / pt_h * 100) if pt_h and pa_h is not None else None

        pa_a = _safe_float(as_stats.get('passes_accurate', [0])[0]) if isinstance(as_stats.get('passes_accurate'), list) else 0
        pt_a = _safe_float(as_stats.get('passes_total', [1])[0]) if isinstance(as_stats.get('passes_total'), list) else 1
        f['away_pass_accuracy'] = (pa_a / pt_a * 100) if pt_a and pa_a is not None else None

        # --- Discipline ---
        fl_h = _safe_float(hs_stats.get('fouls', [0])[0]) if isinstance(hs_stats.get('fouls'), list) else 0
        f['home_fouls_avg'] = fl_h / hgames if hgames and fl_h is not None else None
        fl_a = _safe_float(as_stats.get('fouls', [0])[0]) if isinstance(as_stats.get('fouls'), list) else 0
        f['away_fouls_avg'] = fl_a / agames if agames and fl_a is not None else None

        yc_h = _safe_float(hs_stats.get('yellowcards', [0])[0]) if isinstance(hs_stats.get('yellowcards'), list) else 0
        f['home_yellow_avg'] = yc_h / hgames if hgames and yc_h is not None else None
        yc_a = _safe_float(as_stats.get('yellowcards', [0])[0]) if isinstance(as_stats.get('yellowcards'), list) else 0
        f['away_yellow_avg'] = yc_a / agames if agames and yc_a is not None else None

        # --- Odds ---
        odds = match_data.get('odds', {})
        f['odds_home'] = odds.get('1')
        f['odds_draw'] = odds.get('X')
        f['odds_away'] = odds.get('2')
        f['odds_over'] = odds.get('over')
        f['odds_under'] = odds.get('under')

        # --- Forebet ---
        preds = match_data.get('predictions', {})
        f['forebet_probability'] = preds.get('probability')

        return f

    def _features_to_array(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numpy array, filling defaults for None."""
        defaults = {
            'home_position': 10, 'away_position': 10, 'position_diff': 0,
            'home_points': 30, 'away_points': 30,
            'home_form_points': 6, 'away_form_points': 6,
            'home_recent_form': 3, 'away_recent_form': 3,
            'home_wins_l6': 2, 'home_draws_l6': 2, 'home_losses_l6': 2,
            'away_wins_l6': 2, 'away_draws_l6': 2, 'away_losses_l6': 2,
            'home_scored_avg': 1.3, 'home_conceded_avg': 1.0,
            'away_scored_avg': 1.1, 'away_conceded_avg': 1.2,
            'expected_total_goals': 2.5,
            'home_home_win_rate': 50, 'away_away_win_rate': 33,
            'home_home_goals_avg': 1.5, 'away_away_goals_avg': 1.0,
            'h2h_home_win_pct': 33, 'h2h_draw_pct': 33, 'h2h_away_win_pct': 33,
            'home_shots_avg': 14, 'away_shots_avg': 12,
            'home_shots_on_target_pct': 35, 'away_shots_on_target_pct': 35,
            'home_dangerous_attacks_avg': 55, 'away_dangerous_attacks_avg': 50,
            'home_possession': 50, 'away_possession': 50,
            'home_pass_accuracy': 80, 'away_pass_accuracy': 80,
            'home_fouls_avg': 12, 'away_fouls_avg': 12,
            'home_yellow_avg': 1.5, 'away_yellow_avg': 1.5,
            'odds_home': 2.5, 'odds_draw': 3.0, 'odds_away': 3.5,
            'odds_over': 1.9, 'odds_under': 1.9,
            'forebet_probability': 33,
            # Prediction history defaults (0.5 = neutral/unknown)
            'home_team_prediction_correct_pct': 0.5,
            'away_team_prediction_correct_pct': 0.5,
            'league_prediction_correct_pct': 0.5,
            'overall_model_accuracy': 0.5,
        }
        vec = []
        for name in self.FEATURE_NAMES:
            val = features.get(name)
            if val is None:
                val = defaults.get(name, 0)
            vec.append(float(val))
        return np.array(vec).reshape(1, -1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, training_data: List[Dict] = None, league: str = None, 
              test_examples: List[Dict] = None, test_size: float = 0.2) -> Dict:
        """
        Train models with optional time-based test split for accurate accuracy.
        
        Args:
            training_data: Training examples (league-formatted or flat list)
            league: League name for league-specific model
            test_examples: Optional pre-split test examples
            test_size: Fraction of data to use for testing (default 20%)
        
        Returns:
            Dict with training and test accuracy metrics
        """
        # Lazy import to avoid circular dependency
        from data_storage import MatchDataStorage
        storage = MatchDataStorage()
        
        if training_data is None:
            if league:
                training_data = storage.get_league_training_data(league)
            else:
                training_data = storage.get_training_data()
        
        # For new structure: [{'league': 'Serie A', 'examples': [...]}, ...]
        if training_data and isinstance(training_data[0], dict) and 'examples' in training_data[0]:
            # League-specific training with combined data
            if not league:
                league = training_data[0].get('league')
            examples = training_data[0].get('examples', [])
            # If league not specified and we want global model, combine all examples
            if not league and len(training_data) > 1:
                examples = []
                for entry in training_data:
                    examples.extend(entry.get('examples', []))
        else:
            # Old structure: [{'features': ..., 'labels': ...}, ...]
            examples = training_data
        
        # Split examples into train/test (time-based: oldest for train, newest for test)
        if test_examples is None:
            # Sort by timestamp if available, otherwise use order
            examples_sorted = sorted(examples, key=lambda x: x.get('timestamp', ''))
            split_idx = int(len(examples_sorted) * (1 - test_size))
            train_examples = examples_sorted[:split_idx]
            test_examples = examples_sorted[split_idx:]
        else:
            train_examples = examples
        
        if len(train_examples) < 10:
            return {
                'error': 'Insufficient training data', 
                'required': 10, 
                'available': len(train_examples),
                'league': league or 'global'
            }

        # Convert training examples to arrays
        X_train, y_result_train, y_ou_train = [], [], []
        for ex in train_examples:
            feats = ex.get('features', {})
            labels = ex.get('labels', {})
            vec = self._features_to_array(feats).flatten().tolist()
            X_train.append(vec)
            y_result_train.append(labels.get('result', 'X'))
            y_ou_train.append(labels.get('over_under_2_5', 'Under'))

        X_train = np.array(X_train)
        y_result_train = np.array(y_result_train)
        y_ou_train = np.array(y_ou_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        result_model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
        result_model.fit(X_train_scaled, y_result_train)

        ou_model = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
        ou_model.fit(X_train_scaled, y_ou_train)

        # Calculate training accuracy
        train_result_acc = float(result_model.score(X_train_scaled, y_result_train))
        train_ou_acc = float(ou_model.score(X_train_scaled, y_ou_train))
        
        # Calculate test accuracy if test data available
        test_result_acc = None
        test_ou_acc = None
        test_examples_count = 0
        
        if test_examples and len(test_examples) > 0:
            X_test, y_result_test, y_ou_test = [], [], []
            for ex in test_examples:
                feats = ex.get('features', {})
                labels = ex.get('labels', {})
                vec = self._features_to_array(feats).flatten().tolist()
                X_test.append(vec)
                y_result_test.append(labels.get('result', 'X'))
                y_ou_test.append(labels.get('over_under_2_5', 'Under'))
            
            if X_test:
                X_test = np.array(X_test)
                X_test_scaled = scaler.transform(X_test)
                y_result_test = np.array(y_result_test)
                y_ou_test = np.array(y_ou_test)
                
                test_result_acc = float(result_model.score(X_test_scaled, y_result_test))
                test_ou_acc = float(ou_model.score(X_test_scaled, y_ou_test))
                test_examples_count = len(test_examples)

        if league:
            # Save league-specific model
            league_dir = os.path.join(self.model_dir, 'leagues', league)
            os.makedirs(league_dir, exist_ok=True)
            
            with open(os.path.join(league_dir, 'result_model.pkl'), 'wb') as f:
                pickle.dump(result_model, f)
            with open(os.path.join(league_dir, 'ou_model.pkl'), 'wb') as f:
                pickle.dump(ou_model, f)
            with open(os.path.join(league_dir, 'scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
            
            # Reload league models
            self._load_league_models()
            
            # Save training history
            self._save_training_history({
                'training_examples': len(train_examples),
                'test_examples': test_examples_count,
                'result_accuracy': test_result_acc if test_result_acc is not None else train_result_acc,
                'ou_accuracy': test_ou_acc if test_ou_acc is not None else train_ou_acc,
                'train_result_acc': train_result_acc,
                'train_ou_acc': train_ou_acc,
                'test_result_acc': test_result_acc,
                'test_ou_acc': test_ou_acc,
                'model_type': 'league_specific',
                'league': league
            })
            
            return {
                'league': league,
                'training_examples': len(train_examples),
                'test_examples': test_examples_count,
                'train_result_accuracy': train_result_acc,
                'train_ou_accuracy': train_ou_acc,
                'result_accuracy': test_result_acc if test_result_acc is not None else train_result_acc,
                'ou_accuracy': test_ou_acc if test_ou_acc is not None else train_ou_acc,
                'model_type': 'league_specific'
            }
        else:
            # Save global model
            self.result_model = result_model
            self.ou_model = ou_model
            self.scaler = scaler
            self._save_models()
            
            # Save training history
            self._save_training_history({
                'training_examples': len(train_examples),
                'test_examples': test_examples_count,
                'result_accuracy': test_result_acc if test_result_acc is not None else train_result_acc,
                'ou_accuracy': test_ou_acc if test_ou_acc is not None else train_ou_acc,
                'train_result_acc': train_result_acc,
                'train_ou_acc': train_ou_acc,
                'test_result_acc': test_result_acc,
                'test_ou_acc': test_ou_acc,
                'model_type': 'global'
            })
            
            return {
                'training_examples': len(train_examples),
                'test_examples': test_examples_count,
                'train_result_accuracy': train_result_acc,
                'train_ou_accuracy': train_ou_acc,
                'result_accuracy': test_result_acc if test_result_acc is not None else train_result_acc,
                'ou_accuracy': test_ou_acc if test_ou_acc is not None else train_ou_acc,
                'model_type': 'global'
            }

    def _save_training_history(self, result: Dict):
        """Save training result to history file."""
        history_file = "training_history.json"
        history = []
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            **result
        })
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: Dict, league: str = None, country: str = None, match_id: str = None) -> Dict:
        """
        Predict match outcome. Uses league-specific model if available.
        Falls back to global model, then statistical prediction.
        
        Args:
            features: Feature dictionary for the match
            league: League code (e.g., "Nl2", "En1")
            country: Country name (e.g., "Netherlands", "England")
            match_id: Match ID from URL (used for league lookup)
        """
        # Try league-specific model first
        model_league = None
        
        # Method 1: Look up in leagues_db using league_code + match_id prefix
        if league and match_id:
            prefix = match_id[:3] if len(match_id) >= 3 else match_id
            lookup_key = f"{league}_{prefix}"
            if lookup_key in self.leagues_db:
                db_entry = self.leagues_db[lookup_key]
                db_country = db_entry.get('country', '')
                db_league_name = db_entry.get('league', '')
                # Try to find model by country_league_name format
                model_key_candidate = f"{db_country}_{db_league_name}".replace(' ', '_')
                if model_key_candidate in self.league_models:
                    model_league = model_key_candidate
        
        # Method 2: Try exact match with league code
        if not model_league and league and league in self.league_models:
            model_league = league
        
        # Method 3: Try to find a model by checking if any key starts with the league
        if not model_league and league:
            for model_key in self.league_models:
                if model_key.startswith(league + '_') or model_key == league:
                    model_league = model_key
                    break
        
        # Method 4: If no league model found, try country name
        if not model_league and country:
            # Try country name as stored in models (e.g., "Netherlands_")
            country_model_key = country + '_'
            if country_model_key in self.league_models:
                model_league = country_model_key
            else:
                # Try to find any model that matches the country
                for model_key in self.league_models:
                    if model_key.lower().startswith(country.lower() + '_'):
                        model_league = model_key
                        break
        
        # Method 5: Look up in leagues_db by league_code only
        if not model_league and league:
            for db_key, db_entry in self.leagues_db.items():
                if db_entry.get('league_code') == league:
                    db_country = db_entry.get('country', '')
                    db_league_name = db_entry.get('league', '')
                    model_key_candidate = f"{db_country}_{db_league_name}".replace(' ', '_')
                    if model_key_candidate in self.league_models:
                        model_league = model_key_candidate
                        break
        
        if model_league:
            models = self.league_models[model_league]
            return self._ml_prediction_with_models(features, models, is_league_specific=True)
        
        # Fall back to global model (trained with all data)
        if 'Global_Model' in self.league_models:
            models = self.league_models['Global_Model']
            return self._ml_prediction_with_models(features, models, is_league_specific=False)
        
        # Fall back to main model if available
        if self.result_model and self.ou_model and self.scaler:
            return self._ml_prediction(features, model_trained=True)
        
        # Final fallback
        return self._statistical_prediction(features)

    def _ml_prediction_with_models(self, features: Dict, models: Dict, is_league_specific: bool = False) -> Dict:
        """Use specific models (league or global)."""
        scaler = models.get('scaler')
        result_model = models.get('result_model')
        ou_model = models.get('ou_model')
        
        if not (scaler and result_model and ou_model):
            return self._statistical_prediction(features)
        
        # Check if this is a league-specific model trained with 8 features
        # (from rebuild_data.py) vs the full 51-feature model
        n_features = getattr(scaler, 'n_features_in_', 51)
        
        if n_features == 8:
            # Use simplified 8-feature format for league models
            X = self._features_to_array_8(features)
        else:
            X = self._features_to_array(features)
        
        X_scaled = scaler.transform(X)

        rp = result_model.predict(X_scaled)[0]
        rproba = dict(zip(result_model.classes_, result_model.predict_proba(X_scaled)[0]))

        op = ou_model.predict(X_scaled)[0]
        oproba = dict(zip(ou_model.classes_, ou_model.predict_proba(X_scaled)[0]))

        return self._build_prediction(rp, rproba, op, oproba, model_trained=True, prediction_method='league_ml' if is_league_specific else 'ml')

    def _features_to_array_8(self, features: Dict) -> np.ndarray:
        """Convert feature dict to 8-feature array for league-specific models.
        
        These are the features used by rebuild_data.py for training:
        - prob_home, prob_draw, prob_away
        - prob_home_away_ratio, prob_draw_diff
        - predicted_avg_goals, odds, league_code_encoded
        """
        # Get odds
        odds_home = features.get('odds_home') or 2.5
        odds_draw = features.get('odds_draw') or 3.0
        odds_away = features.get('odds_away') or 3.5
        
        # Calculate implied probabilities
        prob_home = 1 / odds_home
        prob_draw = 1 / odds_draw
        prob_away = 1 / odds_away
        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total
        
        # Derived features
        prob_home_away_ratio = prob_home / max(prob_away, 0.01)
        prob_draw_diff = abs(prob_home - prob_away)
        
        # Predicted goals from expected_total_goals
        predicted_avg_goals = features.get('expected_total_goals', 2.5) or 2.5
        
        # Use home odds as primary odds
        odds = odds_home
        
        # League code encoding (default 0)
        league_code_encoded = 0
        
        vec = [
            prob_home,
            prob_draw,
            prob_away,
            prob_home_away_ratio,
            prob_draw_diff,
            predicted_avg_goals,
            odds,
            league_code_encoded,
        ]
        return np.array(vec).reshape(1, -1)

    def _ml_prediction(self, features: Dict, model_trained: bool = True) -> Dict:
        X = self._features_to_array(features)
        X_scaled = self.scaler.transform(X)

        rp = self.result_model.predict(X_scaled)[0]
        rproba = dict(zip(self.result_model.classes_, self.result_model.predict_proba(X_scaled)[0]))

        op = self.ou_model.predict(X_scaled)[0]
        oproba = dict(zip(self.ou_model.classes_, self.ou_model.predict_proba(X_scaled)[0]))

        return self._build_prediction(rp, rproba, op, oproba, model_trained=model_trained, prediction_method='ml')

    def _statistical_prediction(self, features: Dict) -> Dict:
        """
        Advanced statistical prediction combining:
        1. Odds-implied probabilities
        2. Form-based adjustments
        3. Home/away performance
        4. Head-to-head record
        5. Goals statistics (Poisson-like)
        """
        # --- 1. Odds-implied probabilities ---
        oh = features.get('odds_home') or 2.5
        od = features.get('odds_draw') or 3.0
        oa = features.get('odds_away') or 3.5
        oo = features.get('odds_over') or 1.9
        ou = features.get('odds_under') or 1.9

        p1 = 1 / oh
        px = 1 / od
        p2 = 1 / oa
        total = p1 + px + p2
        p1 /= total; px /= total; p2 /= total

        po = 1 / oo
        pu = 1 / ou
        total_ou = po + pu
        po /= total_ou; pu /= total_ou

        # --- 2. Form-based adjustment ---
        hfp = features.get('home_form_points') or 6
        afp = features.get('away_form_points') or 6
        form_diff = (hfp - afp) / 18  # normalise to [-1, 1]
        p1 += form_diff * 0.05
        p2 -= form_diff * 0.05

        # --- 3. Home/away performance ---
        hwr = (features.get('home_home_win_rate') or 50) / 100
        awr = (features.get('away_away_win_rate') or 33) / 100
        p1 += (hwr - 0.5) * 0.08
        p2 += (awr - 0.33) * 0.08

        # --- 4. Head-to-head ---
        h2h_h = (features.get('h2h_home_win_pct') or 33) / 100
        h2h_a = (features.get('h2h_away_win_pct') or 33) / 100
        h2h_draw = (features.get('h2h_draw_pct') or 33) / 100
        p1 += (h2h_h - 0.33) * 0.05
        p2 += (h2h_a - 0.33) * 0.05
        px += (h2h_draw - 0.33) * 0.08  # H2H draws more predictive

        # --- 5. Expected total goals (needed for draw analysis) ---
        etg = features.get('expected_total_goals') or 2.5

        # --- 6. DRAW-SPECIFIC ANALYSIS ---
        # Draw indicators based on analysis of historical draw matches:
        # - Similar form strength: +8%
        # - Both teams have draws in recent form: +5-10%
        # - Low-scoring teams: +5%
        # - Similar league positions: +4%
        # - High H2H draw %: +6%
        # - Moderate home/away win rates: +5%
        
        draw_bonus = 0.0
        
        # Form similarity (both teams similar strength)
        form_diff = abs(hfp - afp) / 18  # Already normalized
        if form_diff <= 0.2:  # Nearly equal form
            draw_bonus += 0.08
        elif form_diff <= 0.35:  # Similar form
            draw_bonus += 0.04
        
        # Recent draws in form
        home_losses = features.get('home_losses_l6', 2)
        away_losses = features.get('away_losses_l6', 2)
        
        # Teams that don't lose much = draws
        if home_losses <= 1 and away_losses <= 1:
            draw_bonus += 0.10
        elif home_losses <= 2 and away_losses <= 2:
            draw_bonus += 0.05
        
        # Low-scoring teams increase draw probability
        if etg < 1.8:
            draw_bonus += 0.05
        elif etg < 2.0:
            draw_bonus += 0.03
        
        # Position similarity
        home_pos = features.get('home_position', 10)
        away_pos = features.get('away_position', 10)
        if home_pos and away_pos:
            pos_diff = abs(home_pos - away_pos)
            if pos_diff <= 3:
                draw_bonus += 0.04
        
        # H2H draw history
        if h2h_draw > 0.40:
            draw_bonus += 0.06
        elif h2h_draw > 0.25:
            draw_bonus += 0.03
        
        # Moderate home/away win rates (neither dominant)
        hwr = (features.get('home_home_win_rate') or 50) / 100
        awr = (features.get('away_away_win_rate') or 33) / 100
        if 0.40 <= hwr <= 0.65 and 0.20 <= awr <= 0.45:
            draw_bonus += 0.05
        
        # Apply draw bonus
        px += draw_bonus
        
        # --- 6. Goals-based O/U adjustment ---
        if etg > 2.8:
            po += 0.05
            pu -= 0.05
        elif etg < 2.2:
            po -= 0.05
            pu += 0.05

        # Normalise
        total = p1 + px + p2
        p1 /= total; px /= total; p2 /= total
        total_ou = po + pu
        po /= total_ou; pu /= total_ou

        # Clamp
        p1 = max(0.02, min(0.95, p1))
        px = max(0.02, min(0.95, px))
        p2 = max(0.02, min(0.95, p2))

        probs = {'1': p1, 'X': px, '2': p2}
        rp = max(probs, key=probs.get)
        oprobs = {'Over': po, 'Under': pu}
        op = max(oprobs, key=oprobs.get)

        return self._build_prediction(rp, probs, op, oprobs, model_trained=False, prediction_method='statistical')

    @staticmethod
    def _build_prediction(result_pred, result_probs, ou_pred, ou_probs, model_trained, prediction_method: str = 'ml'):
        def safe_odds(p):
            return round(1 / p, 2) if p and p > 0.01 else 99.0
        
        def convert_key(k):
            """Convert numpy types to Python types and map result codes."""
            if hasattr(k, 'item'):  # numpy type
                k = k.item()
            # Map integer codes to string codes for result
            if k == 0:
                return '1'  # home win
            elif k == 1:
                return 'X'  # draw
            elif k == 2:
                return '2'  # away win
            return str(k)
        
        def convert_ou_key(k):
            """Convert over/under keys."""
            if hasattr(k, 'item'):
                k = k.item()
            # Map integer codes for over/under
            if k == 0:
                return 'Under'
            elif k == 1:
                return 'Over'
            return str(k)
        
        # Convert result_pred to Python string
        result_pred = convert_key(result_pred)
        ou_pred = convert_ou_key(ou_pred)
        
        # Convert dictionary keys to Python strings
        result_probs = {convert_key(k): round(float(v), 4) for k, v in result_probs.items()}
        ou_probs = {convert_ou_key(k): round(float(v), 4) for k, v in ou_probs.items()}
        
        return {
            'prediction_method': prediction_method,
            'result': {
                'prediction': result_pred,
                'probabilities': result_probs,
                'computed_odds': {k: round(1/v, 2) if v > 0.01 else 99.0 for k, v in result_probs.items()},
                'confidence': round(float(max(result_probs.values())), 4),
            },
            'over_under': {
                'prediction': ou_pred,
                'probabilities': ou_probs,
                'computed_odds': {k: round(1/v, 2) if v > 0.01 else 99.0 for k, v in ou_probs.items()},
                'confidence': round(float(max(ou_probs.values())), 4),
            },
            'model_trained': model_trained,
        }
