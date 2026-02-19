#!/usr/bin/env python3
"""
Football Match Outcome Prediction System
Main system that integrates scraping, storage, and prediction.
"""

import argparse
import json
import os
from typing import Dict
from datetime import datetime

from football_scraper import ForebetScraper
from data_storage import MatchDataStorage
from prediction_model import FootballPredictor, WeightedPredictor

# JSON output directory
PREDICTIONS_DIR = "predictions"


# ======================================================================
# ANSI colour helpers (works in most modern terminals)
# ======================================================================
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    WHITE  = "\033[97m"
    BG_BLUE= "\033[44m"
    BG_GREEN="\033[42m"
    BG_RED = "\033[41m"
    BG_YELLOW="\033[43m"
    UNDERLINE="\033[4m"


def _bar(pct: float, width: int = 20, fill_char="█", empty_char="░") -> str:
    filled = int(pct / 100 * width)
    return fill_char * filled + empty_char * (width - filled)


def _colour_form(letter: str) -> str:
    if letter == 'W':
        return f"{C.BG_GREEN}{C.WHITE}{C.BOLD} W {C.RESET}"
    elif letter == 'D':
        return f"{C.BG_YELLOW}{C.WHITE}{C.BOLD} D {C.RESET}"
    elif letter == 'L':
        return f"{C.BG_RED}{C.WHITE}{C.BOLD} L {C.RESET}"
    return f" {letter} "


def _get_opponent_position(opponent: str, league_table: list) -> int:
    """Look up opponent's position in league table."""
    if not opponent or not league_table:
        return None
    
    # Clean opponent name for comparison
    opponent_clean = opponent.lower().strip()
    # Common team name variations
    variations = {
        'manchester utd': 'manchester united',
        'man city': 'manchester city',
        'm. united': 'manchester united',
        'real madrid': 'real madrid',
        'sevilla': 'sevilla',
        'athletic': 'athletic club',
        'athletic bilbao': 'athletic club',
        'bilbao': 'athletic club',
        'sociedad': 'real sociedad',
        'betis': 'real betis',
        'villarreal': 'villarreal cf',
        'valencia': 'valencia cf',
        'levante': 'levante ud',
        'osAsuna': 'osAsuna',
        'celta': 'celta vigo',
        'celta vigo': 'celta vigo',
        'alaves': 'deportivo alaves',
        'alavés': 'deportivo alaves',
        'getafe': 'getafe cf',
        'granada': 'granada cf',
        'espanyol': 'rcd espanyol',
        'barcelona': 'fc barcelona',
        'atletico': 'atletico madrid',
        'atletico madrid': 'atletico madrid',
    }
    
    # Try to normalize the name
    opponent_normalized = variations.get(opponent_clean, opponent_clean)
    
    for entry in league_table:
        team_name = entry.get('team', '').lower()
        team_normalized = variations.get(team_name, team_name)
        
        # Exact match or first word match (e.g., "Real" matches "Real Sociedad")
        if (opponent_normalized == team_normalized or 
            opponent_clean == team_name or
            (len(opponent_clean) > 3 and (opponent_clean in team_name or team_name in opponent_clean))):
            return entry.get('position')
    
    # Try matching first significant word
    words = opponent_clean.split()
    if words:
        first_word = words[0]
        for entry in league_table:
            team_name = entry.get('team', '').lower()
            if first_word in team_name and len(first_word) > 3:
                return entry.get('position')
    
    return None


def _format_match_result(home_score: int, away_score: int, is_home_team: bool) -> str:
    """Format match result with appropriate color."""
    if home_score == away_score:
        return f"{C.YELLOW}{home_score}-{away_score}{C.RESET}"
    elif (home_score > away_score and is_home_team) or (away_score > home_score and not is_home_team):
        return f"{C.GREEN}{home_score}-{away_score}{C.RESET}"
    else:
        return f"{C.RED}{home_score}-{away_score}{C.RESET}"


def _get_venue_icon(is_home: bool) -> str:
    return "🏠" if is_home else "🛣️"


def _get_result_icon(result: str, is_home_team: bool) -> str:
    """Get icon for match result (W/D/L)."""
    if result == 'W':
        return "✅"
    elif result == 'D':
        return "⚪"
    else:
        return "❌"


def _pct_colour(pct: float) -> str:
    if pct >= 50:
        return C.GREEN
    elif pct >= 35:
        return C.YELLOW
    return C.RED


# ======================================================================
# Main system
# ======================================================================

class FootballPredictionSystem:
    """Complete football prediction system."""

    def __init__(self):
        self.scraper = ForebetScraper()
        self.storage = MatchDataStorage()
        self.predictor = FootballPredictor()
        self.weighted_predictor = WeightedPredictor()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def predict_match(self, url: str, save_data: bool = True) -> Dict:
        """Scrape, analyse, and predict a match."""
        # prompt_user=False so auto-extraction is used, with placeholder if unknown
        match_data = self.scraper.scrape_match(url, prompt_user=False)
        if not match_data:
            return {'error': 'Failed to scrape match data', 'url': url}

        features = FootballPredictor.extract_features(match_data)
        
        # Extract league for league-specific prediction
        # Note: scraper stores info as 'match_info' with league_code and league fields
        league_info = match_data.get('match_info', match_data.get('info', {}))
        league_code = league_info.get('league_code')
        league = league_info.get('league')
        country = league_info.get('country')
        match_id = league_info.get('match_id')
        
        # Extract match_id from URL if not in league_info
        if not match_id:
            import re
            match_id_match = re.search(r'-(\d{5,7})$', url)
            if match_id_match:
                match_id = match_id_match.group(1)
        
        # Get ML prediction with league, country, and match_id for model lookup
        ml_prediction = self.predictor.predict(features, league_code, country, match_id)
        
        # Generate weighted prediction with league-specific weights
        weighted_prediction = self.weighted_predictor.predict(match_data, features, league_code)
        
        # Calculate convergence
        ml_result = ml_prediction.get('result', {}).get('prediction')
        weighted_result = weighted_prediction.get('result', {}).get('prediction')
        ml_ou = ml_prediction.get('over_under', {}).get('prediction')
        weighted_ou = weighted_prediction.get('over_under', {}).get('prediction')
        
        convergence = {
            'result_match': ml_result == weighted_result,
            'ou_match': ml_ou == weighted_ou,
            'ml_result': ml_result,
            'weighted_result': weighted_result,
            'ml_ou': ml_ou,
            'weighted_ou': weighted_ou,
        }
        
        # Prefer ML prediction over weighted factors when ML model is trained
        if ml_prediction.get('prediction_method') in ['ml', 'league_ml']:
            primary_prediction = ml_prediction
        else:
            primary_prediction = weighted_prediction
        
        # Apply key factor adjustment to prediction
        primary_prediction = self._adjust_prediction_by_key_factors(
            primary_prediction, match_data, features, ml_prediction, weighted_prediction
        )

        # Scrape injury data for both teams
        teams = match_data.get('teams', {})
        home_team = teams.get('home', '')
        away_team = teams.get('away', '')
        
        injuries_data = {}
        if home_team:
            home_injuries = self.scraper.scrape_team_injuries(home_team)
            injuries_data[home_team] = home_injuries
        if away_team:
            away_injuries = self.scraper.scrape_team_injuries(away_team)
            injuries_data[away_team] = away_injuries

        return {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'match_data': match_data,
            'features': features,
            'prediction': primary_prediction,
            'ml_prediction': ml_prediction,
            'weighted_prediction': weighted_prediction,
            'convergence': convergence,
            'analysis': self._analyse(match_data, features, primary_prediction, convergence),
            'injuries': injuries_data,
        }

    def add_match_result(self, url: str) -> bool:
        result = self.scraper.extract_actual_result(url)
        if not result or not result.get('result'):
            print("Could not extract result – match may not have been played yet.")
            return False
        return self.storage.save_match_result(url, result)

    def train_model(self) -> Dict:
        # Get flattened list of all training examples
        td = self.storage.get_league_training_data()
        total_examples = len(td)  # Already flattened
        if total_examples < 10:
            return {'error': 'Insufficient training data', 'available': total_examples, 'required': 10}
        
        # Train with test split (20%)
        result = self.predictor.train(td, test_size=0.2)
        
        # Format result for display
        if 'error' not in result:
            return {
                'success': True,
                'training_examples': result.get('training_examples'),
                'test_examples': result.get('test_examples'),
                'train_accuracy': result.get('train_result_accuracy', 0) * 100,
                'test_accuracy': result.get('result_accuracy', 0) * 100,
                'message': f"Trained on {result.get('training_examples')} examples, tested on {result.get('test_examples')}"
            }
        return result

    def get_statistics(self) -> Dict:
        ss = self.storage.get_statistics()
        return {**ss, 'model_trained': self.predictor.result_model is not None}

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _adjust_prediction_by_key_factors(self, prediction: Dict, match_data: Dict, 
                                            features: Dict, ml_prediction: Dict, 
                                            weighted_prediction: Dict) -> Dict:
        """Adjust prediction based on key factors from both ML and weighted predictions.
        
        This method can override predictions when strong factors indicate a different outcome.
        Currently returns the prediction unchanged - adjustments can be added as needed.
        """
        # For now, return the prediction as-is
        # Future implementations could adjust confidence or prediction based on:
        # - Strong disagreement between ML and weighted predictions
        # - Key factor analysis from weighted prediction
        # - Form trends, H2H data, etc.
        return prediction

    def _analyse(self, md: Dict, feats: Dict, pred: Dict, convergence: Dict = None) -> Dict:
        analysis: Dict = {'key_factors': [], 'value_bets': [], 'confidence': 'medium'}
        
        teams = md.get('teams', {})
        home = teams.get('home', 'Home')
        away = teams.get('away', 'Away')

        # Get the prediction result to ensure key factors are consistent
        result_prediction = pred.get('result', {}).get('prediction', '')
        result_confidence = pred.get('result', {}).get('confidence', 0) * 100

        # Form
        hf = md.get('form', {}).get('home', [])
        af = md.get('form', {}).get('away', [])
        hw = hf.count('W')
        aw = af.count('W')
        hd = hf.count('D')
        ad = af.count('D')
        
        # Show form factors that support the prediction
        if result_prediction == '1':  # Home win predicted
            if hw >= 3:
                analysis['key_factors'].append(f"✅ {home} in good form ({hw} wins in last {len(hf)})")
            if aw <= 2:
                analysis['key_factors'].append(f"⚠️ {away} in poor form ({aw} wins in last {len(af)})")
        elif result_prediction == '2':  # Away win predicted
            if aw >= 3:
                analysis['key_factors'].append(f"✅ {away} in good form ({aw} wins in last {len(af)})")
            if hw <= 2:
                analysis['key_factors'].append(f"⚠️ {home} in poor form ({hw} wins in last {len(hf)})")
        else:  # Draw predicted
            if hw <= 2 and aw <= 2:
                analysis['key_factors'].append(f"🤝 Both teams in similar form")
        
        # Home/away record - show factors that support the prediction
        hwr = feats.get('home_home_win_rate')
        awr = feats.get('away_away_win_rate')
        
        if result_prediction == '1':  # Home win predicted
            if hwr and hwr >= 60:
                analysis['key_factors'].append(f"🏟️ {home} strong at home ({hwr:.0f}% win rate)")
            if awr and awr <= 30:
                analysis['key_factors'].append(f"📉 {away} struggles away ({awr:.0f}% win rate)")
        elif result_prediction == '2':  # Away win predicted
            if awr and awr >= 50:
                analysis['key_factors'].append(f"💪 {away} strong away ({awr:.0f}% win rate)")
            if hwr and hwr <= 40:
                analysis['key_factors'].append(f"📉 {home} weak at home ({hwr:.0f}% win rate)")

        # Goals - make key factors consistent with actual Over/Under prediction
        etg = feats.get('expected_total_goals', 2.5)
        hgs = feats.get('home_avg_goals_scored', 0)
        ags = feats.get('away_avg_goals_scored', 0)
        hgc = feats.get('home_avg_goals_conceded', 0)
        agc = feats.get('away_avg_goals_conceded', 0)
        
        # Get the actual Over/Under prediction to ensure consistency
        ou_prediction = pred.get('over_under', {}).get('prediction', '')
        ou_confidence = pred.get('over_under', {}).get('confidence', 0) * 100
        
        # Only show scoring expectation if it's consistent with the prediction
        if ou_prediction == 'Over':
            analysis['key_factors'].append(f"⚽ High-scoring match expected ({ou_confidence:.0f}% confidence)")
        elif ou_prediction == 'Under':
            analysis['key_factors'].append(f"🛡️ Low-scoring match expected ({ou_confidence:.0f}% confidence)")
        
        # Goal difference comparison - show factors that support the prediction
        if result_prediction == '1':  # Home win predicted
            if hgs and ags and hgs > ags:
                analysis['key_factors'].append(f"⚽ {home} scores more ({hgs:.1f} vs {ags:.1f} goals/game)")
            if hgc and agc and hgc < agc:
                analysis['key_factors'].append(f"🧱 {home} has stronger defense ({hgc:.1f} vs {agc:.1f} conceded/game)")
        elif result_prediction == '2':  # Away win predicted
            if ags and hgs and ags > hgs:
                analysis['key_factors'].append(f"⚽ {away} scores more ({ags:.1f} vs {hgs:.1f} goals/game)")
            if agc and hgc and agc < hgc:
                analysis['key_factors'].append(f"🧱 {away} has stronger defense ({agc:.1f} vs {hgc:.1f} conceded/game)")

        # Standings - show factors that support the prediction
        hp = feats.get('home_position')
        ap = feats.get('away_position')
        if hp and ap:
            pos_diff = abs(hp - ap)
            if result_prediction == '1' and hp < ap:
                analysis['key_factors'].append(f"📊 {home} higher in table (#{hp} vs #{ap})")
            elif result_prediction == '2' and ap < hp:
                analysis['key_factors'].append(f"📊 {away} higher in table (#{ap} vs #{hp})")
            elif pos_diff <= 3:
                analysis['key_factors'].append(f"⚖️ Teams close in table (#{hp} vs #{ap})")

        # H2H - show factors that support the prediction
        h2h = md.get('head_to_head', {}).get('summary', {})
        if result_prediction == '1' and h2h.get('home_win_pct', 0) >= 50:
            analysis['key_factors'].append(f"📜 {home} has H2H advantage ({h2h['home_win_pct']}% wins)")
        elif result_prediction == '2' and h2h.get('away_win_pct', 0) >= 50:
            analysis['key_factors'].append(f"📜 {away} has H2H advantage ({h2h['away_win_pct']}% wins)")
        elif h2h.get('draw_pct', 0) >= 40 and result_prediction == 'X':
            analysis['key_factors'].append(f"📜 H2H often ends in draw ({h2h['draw_pct']}% draws)")

        # Trends
        for t in md.get('trends', []):
            if t.get('percentage', 0) >= 90:
                analysis['key_factors'].append(f"📈 {t['description']} ({t['record']} = {t['percentage']}%)")
        
        # Draw indicators
        if hw <= 2 and aw <= 2 and hd >= 2 and ad >= 2:
            analysis['key_factors'].append(f"🤝 Both teams draw often - draw possible")
        
        # Prediction confidence indicator
        result_conf = pred['result']['confidence']
        if result_conf >= 0.55:
            analysis['key_factors'].append(f"🎯 Strong prediction confidence ({result_conf*100:.0f}%)")

        # Value bets
        market = md.get('odds', {})
        computed = pred['result']['computed_odds']
        for outcome in ('1', 'X', '2'):
            mo = market.get(outcome)
            co = computed.get(outcome)
            if mo and co and mo > co * 1.10:
                val = ((mo / co) - 1) * 100
                analysis['value_bets'].append({
                    'outcome': outcome, 'market': mo, 'fair': co, 'value_pct': round(val, 1),
                })

        # Confidence - consider model agreement
        rc = pred['result']['confidence']
        oc = pred['over_under']['confidence']
        avg = (rc + oc) / 2
        
        # Check if ML and Weighted models agree (use passed convergence parameter)
        result_agreement = convergence.get('result_match', True) if convergence else True
        ou_agreement = convergence.get('ou_match', True) if convergence else True
        
        # If models disagree, significantly reduce confidence
        if not result_agreement or not ou_agreement:
            # When models disagree, set confidence to at most MEDIUM
            if avg > 0.40:
                avg = 0.40
        
        analysis['confidence'] = 'HIGH' if avg > 0.55 else ('MEDIUM' if avg > 0.40 else 'LOW')

        return analysis

    # ------------------------------------------------------------------
    # Beautiful display
    # ------------------------------------------------------------------

    def display_prediction(self, result: Dict):
        import sys
        import os
        import traceback
        trace_id = os.urandom(4).hex()
        call_id = id(result)
        
        md = result['match_data']
        pred = result['prediction']
        feats = result['features']
        analysis = result['analysis']
        convergence = result.get('convergence', {})
        teams = md.get('teams', {})
        info = md.get('match_info', {})
        home = teams.get('home', '???')
        away = teams.get('away', '???')

        w = 70  # display width

        # ── Header ──
        print()
        print(f"{C.BG_BLUE}{C.WHITE}{C.BOLD}{'':^{w}}{C.RESET}")
        print(f"{C.BG_BLUE}{C.WHITE}{C.BOLD}{'⚽  MATCH PREDICTION  ⚽':^{w}}{C.RESET}")
        print(f"{C.BG_BLUE}{C.WHITE}{C.BOLD}{'':^{w}}{C.RESET}")
        print()
        print(f"  {C.BOLD}{C.CYAN}{home}{C.RESET}  {C.DIM}vs{C.RESET}  {C.BOLD}{C.MAGENTA}{away}{C.RESET}")
        print()
        if info.get('league'):
            print(f"  {C.DIM}League:{C.RESET}  {info['league']}")
        if info.get('date'):
            print(f"  {C.DIM}Date:{C.RESET}    {info['date']}  {info.get('time', '')}")
        if info.get('venue'):
            print(f"  {C.DIM}Venue:{C.RESET}   {info['venue']}")
        if info.get('round'):
            print(f"  {C.DIM}Round:{C.RESET}   {info['round']}")

        # ── Standings ──
        st = md.get('standings', {})
        if st.get('home') or st.get('away'):
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}📊 STANDINGS{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            lt = md.get('league_table', [])
            home_entry = next((e for e in lt if e['team'] == home), None)
            away_entry = next((e for e in lt if e['team'] == away), None)
            if home_entry:
                e = home_entry
                print(f"  {C.CYAN}#{e['position']:>2} {home:<20}{C.RESET} "
                      f"Pts:{C.BOLD}{e['points']}{C.RESET}  "
                      f"W:{e['won']} D:{e['drawn']} L:{e['lost']}  "
                      f"GF:{e['gf']} GA:{e['ga']} GD:{e.get('gd', '?')}")
            if away_entry:
                e = away_entry
                print(f"  {C.MAGENTA}#{e['position']:>2} {away:<20}{C.RESET} "
                      f"Pts:{C.BOLD}{e['points']}{C.RESET}  "
                      f"W:{e['won']} D:{e['drawn']} L:{e['lost']}  "
                      f"GF:{e['gf']} GA:{e['ga']} GD:{e.get('gd', '?')}")

        # ── Detailed Last 6 Matches Analysis ──
        l6 = md.get('last_6_matches', {})
        form = md.get('form', {})
        league_table = md.get('league_table', [])
        if l6.get('home') or l6.get('away'):
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}📋 DETAILED LAST 6 MATCHES ANALYSIS{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            
            # Home team last 6 (mix of home and away matches)
            home_pos = feats.get('home_position', '?')
            print(f"\n  {C.CYAN}🏠 {home} (#{home_pos}){C.RESET}")
            print(f"  {C.BOLD}{'─' * 60}{C.RESET}")
            for i, m in enumerate(l6.get('home', [])[:6]):
                # Determine if team was home or away in this match
                match_home_team = m.get('home_team', '')
                is_home_team = (match_home_team.lower() == home.lower())
                
                # Opponent is the other team
                if is_home_team:
                    opp = m.get('away_team', 'Unknown')
                else:
                    opp = match_home_team
                
                opp_pos = _get_opponent_position(opp, league_table)
                opp_pos_str = f"#{opp_pos}" if opp_pos else "N/A"
                venue = _get_venue_icon(is_home_team)
                match_result = _format_match_result(m.get('home_score', 0), m.get('away_score', 0), is_home_team)
                result_icon = _get_result_icon(form.get('home', ['?'])[i] if i < len(form.get('home', [])) else '?', is_home_team)
                comp = m.get('competition', '')
                vs_at = "vs" if is_home_team else "@"
                print(f"  {i+1}. {result_icon} {venue} {match_result} {vs_at} {opp} ({opp_pos_str}) {C.DIM}{comp}{C.RESET}")
            
            # Away team last 6 (mix of home and away matches)
            away_pos = feats.get('away_position', '?')
            print(f"\n  {C.MAGENTA}🛣️ {away} (#{away_pos}){C.RESET}")
            print(f"  {C.BOLD}{'─' * 60}{C.RESET}")
            for i, m in enumerate(l6.get('away', [])[:6]):
                # Determine if team was home or away in this match
                match_home_team = m.get('home_team', '')
                is_home_team = (match_home_team.lower() == away.lower())
                
                # Opponent is the other team
                if is_home_team:
                    opp = m.get('away_team', 'Unknown')
                else:
                    opp = match_home_team
                
                opp_pos = _get_opponent_position(opp, league_table)
                opp_pos_str = f"#{opp_pos}" if opp_pos else "N/A"
                venue = _get_venue_icon(is_home_team)
                match_result = _format_match_result(m.get('home_score', 0), m.get('away_score', 0), is_home_team)
                result_icon = _get_result_icon(form.get('away', ['?'])[i] if i < len(form.get('away', [])) else '?', is_home_team)
                comp = m.get('competition', '')
                vs_at = "vs" if is_home_team else "@"
                print(f"  {i+1}. {result_icon} {venue} {match_result} {vs_at} {opp} ({opp_pos_str}) {C.DIM}{comp}{C.RESET}")

        # ── Home / Away Performance with Positions ──
        hm = md.get('home_matches', [])
        am = md.get('away_matches', [])
        if hm or am:
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}🏟️  HOME / AWAY PERFORMANCE WITH OPPONENT STRENGTH{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            
            if hm:
                hwr = feats.get('home_home_win_rate', 0)
                avg_opp_pos = sum(_get_opponent_position(m.get('away_team', ''), league_table) or 10 for m in hm[:6]) / min(len(hm), 6)
                print(f"\n  {C.CYAN}🏠 {home} at HOME{C.RESET}")
                print(f"  Win Rate: {C.BOLD}{hwr:.0f}%{C.RESET} | Avg Opponent Position: #{avg_opp_pos:.1f}")
                print(f"  {C.BOLD}{'─' * 60}{C.RESET}")
                for m in hm[:6]:
                    opp = m.get('away_team', 'Unknown')
                    opp_pos = _get_opponent_position(opp, league_table)
                    opp_pos_str = f"#{opp_pos}" if opp_pos else "N/A"
                    match_result = _format_match_result(m.get('home_score', 0), m.get('away_score', 0), True)
                    ht = f"({m.get('ht_home', '?')}-{m.get('ht_away', '?')})"
                    print(f"    {m['date']}  {match_result} vs {opp} ({opp_pos_str}) {C.DIM}{ht}{C.RESET}")
            
            if am:
                awr = feats.get('away_away_win_rate', 0)
                avg_opp_pos = sum(_get_opponent_position(m.get('home_team', ''), league_table) or 10 for m in am[:6]) / min(len(am), 6)
                print(f"\n  {C.MAGENTA}🛣️ {away} AWAY{C.RESET}")
                print(f"  Win Rate: {C.BOLD}{awr:.0f}%{C.RESET} | Avg Opponent Position: #{avg_opp_pos:.1f}")
                print(f"  {C.BOLD}{'─' * 60}{C.RESET}")
                for m in am[:6]:
                    opp = m.get('home_team', 'Unknown')
                    opp_pos = _get_opponent_position(opp, league_table)
                    opp_pos_str = f"#{opp_pos}" if opp_pos else "N/A"
                    match_result = _format_match_result(m.get('home_score', 0), m.get('away_score', 0), False)
                    ht = f"({m.get('ht_home', '?')}-{m.get('ht_away', '?')})"
                    print(f"    {m['date']}  {match_result} @ {opp} ({opp_pos_str}) {C.DIM}{ht}{C.RESET}")

        # ── Goals Statistics ──
        gs = md.get('goals_stats', {})
        hg = gs.get('home', {})
        ag = gs.get('away', {})
        if hg.get('scored_avg') or ag.get('scored_avg'):
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}⚽ GOALS STATISTICS{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {'':18} {'Scored':>8} {'Conceded':>10} {'Avg Scr':>9} {'Avg Cnd':>9}")
            # Format averages to 2 decimal places
            h_scr_avg = hg.get('scored_avg', 0)
            h_cnd_avg = hg.get('conceded_avg', 0)
            a_scr_avg = ag.get('scored_avg', 0)
            a_cnd_avg = ag.get('conceded_avg', 0)
            print(f"  {C.CYAN}{home:<18}{C.RESET} {hg.get('scored','?'):>8} {hg.get('conceded','?'):>10} "
                  f"{h_scr_avg:>9.2f} {h_cnd_avg:>9.2f}")
            print(f"  {C.MAGENTA}{away:<18}{C.RESET} {ag.get('scored','?'):>8} {ag.get('conceded','?'):>10} "
                  f"{a_scr_avg:>9.2f} {a_cnd_avg:>9.2f}")

        # ── Home / Away Performance (skip if already shown above) ──
        # This section is now combined with "HOME / AWAY PERFORMANCE WITH OPPONENT STRENGTH" above

        # ── Head to Head ──
        h2h = md.get('head_to_head', {})
        if h2h.get('matches'):
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}🤝 HEAD TO HEAD{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            for m in h2h['matches'][:6]:
                sc = f"{m['home_score']}-{m['away_score']}"
                ht = f"({m['ht_home']}-{m['ht_away']})"
                print(f"    {C.DIM}{m['date']}{C.RESET}  {m['home_team']} {C.BOLD}{sc}{C.RESET} {ht} {m['away_team']} {C.DIM}{m.get('competition','')}{C.RESET}")
            s = h2h.get('summary', {})
            if s:
                print(f"    {C.CYAN}{home}: {s.get('home_wins',0)} wins ({s.get('home_win_pct',0)}%){C.RESET}  "
                      f"{C.YELLOW}Draws: {s.get('draws',0)} ({s.get('draw_pct',0)}%){C.RESET}  "
                      f"{C.MAGENTA}{away}: {s.get('away_wins',0)} wins ({s.get('away_win_pct',0)}%){C.RESET}")

        # ── Shots & Possession ──
        js = md.get('js_detailed_stats', {})
        hs_stats = js.get('home_stats', {})
        as_stats = js.get('away_stats', {})
        if hs_stats or as_stats:
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}🎯 SHOTS & POSSESSION{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            def _s(arr):
                return arr[0] if isinstance(arr, list) and arr else 0
            hgames = gs.get('home', {}).get('games') or 1
            agames = gs.get('away', {}).get('games') or 1
            # Truncate team names if too long
            home_short = home[:15] if len(home) > 15 else home
            away_short = away[:15] if len(away) > 15 else away
            print(f"  {'':20} {C.CYAN}{home_short:>15}{C.RESET}  {C.MAGENTA}{away_short:>15}{C.RESET}")
            # Shots per game
            h_shots = _s(hs_stats.get('shots_total',[]))
            a_shots = _s(as_stats.get('shots_total',[]))
            print(f"  {'Shots/game':20} {h_shots:>10}/{hgames:<4}  {a_shots:>10}/{agames:<4}")
            # On target
            h_target = _s(hs_stats.get('shots_on_target',[]))
            a_target = _s(as_stats.get('shots_on_target',[]))
            print(f"  {'On target':20} {h_target:>15}  {a_target:>15}")
            # Possession
            bp_h = _s(hs_stats.get('ball_poss', []))
            bp_a = _s(as_stats.get('ball_poss', []))
            # Convert to float if string
            try:
                bp_h = float(bp_h) if bp_h else 0
            except (ValueError, TypeError):
                bp_h = 0
            try:
                bp_a = float(bp_a) if bp_a else 0
            except (ValueError, TypeError):
                bp_a = 0
            if bp_h or bp_a:
                print(f"  {'Possession':20} {C.BOLD}{bp_h:.1f}%{C.RESET:>13}  {C.BOLD}{bp_a:.1f}%{C.RESET:>13}")
            # Pass accuracy
            pa_h = _s(hs_stats.get('passes_accurate', [0]))
            pt_h = _s(hs_stats.get('passes_total', [1]))
            pa_a = _s(as_stats.get('passes_accurate', [0]))
            pt_a = _s(as_stats.get('passes_total', [1]))
            if pt_h and pt_a:
                acc_h = pa_h / pt_h * 100 if pt_h else 0
                acc_a = pa_a / pt_a * 100 if pt_a else 0
                print(f"  {'Pass accuracy':20} {acc_h:>14.0f}%  {acc_a:>14.0f}%")

        # ── Trends ──
        trends = md.get('trends', [])
        if trends:
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}📈 KEY TRENDS{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            for t in trends[:8]:
                pct = t.get('percentage', 0)
                col = C.GREEN if pct >= 80 else (C.YELLOW if pct >= 60 else C.RED)
                print(f"    {col}●{C.RESET} {t['description']}  "
                      f"{C.BOLD}{t['record']}{C.RESET} = {col}{pct}%{C.RESET}")

        # ══════════════════════════════════════════════════════════════
        # PREDICTION
        # ══════════════════════════════════════════════════════════════
        
        # ── Key Factors (Weighted Analysis) ──
        if pred.get('factor_analysis'):
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}📊 WEIGHTED FACTOR ANALYSIS{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            
            fa = pred['factor_analysis']
            factors_info = [
                ('league_position', '📊 League Position', '20%'),
                ('odds_analysis', '📈 Odds Analysis', '15%'),
                ('recent_form', '🔥 Recent Form', '20%'),
                ('top5_performance', '⭐ Top 5 Performance', '10%'),
                ('goals_stats', '⚽ Goals Scored/Conceded', '15%'),
                ('h2h', '🤝 Head-to-Head', '10%'),
                ('shots_possession', '🎯 Shots & Possession', '10%'),
            ]
            
            for factor_key, label, weight in factors_info:
                if factor_key in fa:
                    scores = fa[factor_key]
                    if isinstance(scores, dict) and 'home' in scores:
                        home_s = scores['home'] * 100
                        away_s = scores['away'] * 100
                        print(f"  {label} ({weight})")
                        print(f"    {C.CYAN}{home:<12}{C.RESET} {home_s:5.1f}% {C.GREEN}│{C.RESET} {C.MAGENTA}{away:<12}{C.RESET} {away_s:5.1f}%")
        
        # ── Value Bets ──
        if analysis['value_bets']:
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}💰 VALUE BETS{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            for vb in analysis['value_bets']:
                lbl = {'1': f'{home} Win', 'X': 'Draw', '2': f'{away} Win'}.get(vb['outcome'], vb['outcome'])
                print(f"    {C.GREEN}✓{C.RESET} {lbl}: Market {C.BOLD}{vb['market']:.2f}{C.RESET} vs "
                      f"Fair {C.BOLD}{vb['fair']:.2f}{C.RESET}  "
                      f"({C.GREEN}+{vb['value_pct']:.1f}% value{C.RESET})")
        
        # ── OUR PREDICTION ──
        print()
        print(f"  {C.BG_GREEN}{C.WHITE}{C.BOLD}{'':^{w - 4}}{C.RESET}")
        print(f"  {C.BG_GREEN}{C.WHITE}{C.BOLD}{'🎯  OUR PREDICTION  🎯':^{w - 4}}{C.RESET}")
        print(f"  {C.BG_GREEN}{C.WHITE}{C.BOLD}{'':^{w - 4}}{C.RESET}")
        
        # ── MODEL COMPARISON ──
        if 'convergence' in result and isinstance(result, dict):
            conv = result['convergence']
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}📊 MODEL COMPARISON{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            
            ml_r = conv.get('ml_result', '?')
            w_r = conv.get('weighted_result', '?')
            ml_conf = result.get('ml_prediction', {}).get('result', {}).get('confidence', 0) * 100
            w_conf = result.get('weighted_prediction', {}).get('result', {}).get('confidence', 0) * 100
            
            # Result agreement
            r_match = conv.get('result_match', False)
            
            print()
            print(f"  {C.CYAN}ML Model:{C.RESET}      {ml_r} ({ml_conf:.0f}% confidence)")
            print(f"  {C.YELLOW}Weighted Model:{C.RESET} {w_r} ({w_conf:.0f}% confidence)")
            
            # Show agreement status
            if r_match:
                print(f"  {C.GREEN}✓ Both models agree on result{C.RESET}")
            else:
                print(f"  {C.YELLOW}⚠ Models disagree on result{C.RESET}")
            
            # O/U comparison
            ml_o = conv.get('ml_ou', '?')
            w_o = conv.get('weighted_ou', '?')
            o_match = conv.get('ou_match', False)
            
            ml_ou_conf = result.get('ml_prediction', {}).get('over_under', {}).get('confidence', 0) * 100
            w_ou_conf = result.get('weighted_prediction', {}).get('over_under', {}).get('confidence', 0) * 100
            
            print()
            print(f"  {C.CYAN}ML Model O/U:{C.RESET}  {ml_o} ({ml_ou_conf:.0f}%)")
            print(f"  {C.YELLOW}Weighted O/U:{C.RESET}  {w_o} ({w_ou_conf:.0f}%)")
            
            if o_match:
                print(f"  {C.GREEN}✓ Both models agree on O/U{C.RESET}")
            else:
                print(f"  {C.YELLOW}⚠ Models disagree on O/U{C.RESET}")
            
            print()
        
        rp = pred['result']
        op = pred['over_under']
        
        # ── 1X2 PREDICTIONS ──
        print()
        print(f"  {C.BOLD}Match Result:{C.RESET}  ", end="")
        rmap = {'1': f"{C.CYAN}{home} Win{C.RESET}", 'X': f"{C.YELLOW}Draw{C.RESET}", '2': f"{C.MAGENTA}{away} Win{C.RESET}"}
        print(f"{C.BOLD}{rmap.get(rp['prediction'], rp['prediction'])}{C.RESET}")
        print()
        
        # Get ML and Weighted predictions (with fallback for missing keys)
        ml_pred = result.get('ml_prediction', {}).get('result', {}) if isinstance(result, dict) else {}
        w_pred = result.get('weighted_prediction', {}).get('result', {}) if isinstance(result, dict) else {}
        
        import sys
        ml_pred_debug = repr(ml_pred)
        w_pred_debug = repr(w_pred)
        
        # DEBUG: Print the values we're about to use
        import sys
        
        for label, key, col in [('1 (Home)', '1', C.CYAN), ('X (Draw)', 'X', C.YELLOW), ('2 (Away)', '2', C.MAGENTA)]:
            ml_pct = ml_pred.get('probabilities', {}).get(key, 0) * 100
            w_pct = w_pred.get('probabilities', {}).get(key, 0) * 100
            pct = rp['probabilities'].get(key, 0) * 100
            ml_bar = _bar(ml_pct)
            w_bar = _bar(w_pct)
            odds_c = rp['computed_odds'].get(key, 0)
            odds_m = md.get('odds', {}).get(key) or 0
            marker = " ◀" if key == rp['prediction'] else ""
            match = " ✓" if ml_pred.get('prediction') == key and w_pred.get('prediction') == key else ""
            
            print(f"    {col}{label:<12}{C.RESET}")
            print(f"      ML:{ml_bar} {ml_pct:5.1f}%")
            print(f"      W: {w_bar} {w_pct:5.1f}%")
            print(f"      Fair:{odds_c:.2f}  Mkt:{odds_m:.2f}{C.GREEN}{marker}{match}{C.RESET}")
        
        # Display ML's own odds (computed from raw statistics, not market)
        ml_own_odds = pred.get('ml_own_odds', {})
        if ml_own_odds:
            print()
            print(f"  {C.BOLD}🤖 ML's Own Odds (from raw stats):{C.RESET}")
            r_odds = ml_own_odds.get('result', {})
            ou_odds = ml_own_odds.get('over_under', {})
            expected = ml_own_odds.get('expected_goals', 0)
            
            if r_odds:
                print(f"    1: {r_odds.get('odds_home', 0):.2f}  X: {r_odds.get('odds_draw', 0):.2f}  2: {r_odds.get('odds_away', 0):.2f}")
            if ou_odds:
                print(f"    Over: {ou_odds.get('odds_over', 0):.2f}  Under: {ou_odds.get('odds_under', 0):.2f}")
            if expected:
                print(f"    Expected Goals: {expected:.2f}")
        
        # O/U with thresholds
        over_prob = op['probabilities'].get('Over', 0) * 100
        under_prob = op['probabilities'].get('Under', 0) * 100
        
        # Only recommend if above threshold
        ou_prediction = None
        if over_prob >= 55:
            ou_prediction = 'Over'
        elif under_prob >= 60:
            ou_prediction = 'Under'
        else:
            # Recommend the higher one even if below threshold
            ou_prediction = 'Over' if over_prob > under_prob else 'Under'
        
        print()
        print(f"  {C.BOLD}Over/Under 2.5:{C.RESET}  ", end="")
        if ou_prediction == 'Over':
            ou_col = C.GREEN if over_prob >= 55 else C.YELLOW
            print(f"{C.BOLD}{ou_col}{ou_prediction} 2.5{C.RESET} ({over_prob:.0f}%)")
        else:
            ou_col = C.GREEN if under_prob >= 60 else C.YELLOW
            print(f"{C.BOLD}{ou_col}{ou_prediction} 2.5{C.RESET} ({under_prob:.0f}%)")
        print()
        
        # Get ML and Weighted O/U predictions
        ml_ou = result.get('ml_prediction', {}).get('over_under', {}) if isinstance(result, dict) else {}
        w_ou = result.get('weighted_prediction', {}).get('over_under', {}) if isinstance(result, dict) else {}
        
        for label, key, col in [('Over 2.5', 'Over', C.GREEN), ('Under 2.5', 'Under', C.RED)]:
            ml_pct = ml_ou.get('probabilities', {}).get(key, 0) * 100
            w_pct = w_ou.get('probabilities', {}).get(key, 0) * 100
            pct = op['probabilities'].get(key, 0) * 100
            ml_bar = _bar(ml_pct)
            w_bar = _bar(w_pct)
            odds_c = op['computed_odds'].get(key, 0)
            odds_m = md.get('odds', {}).get(key.lower()) or 0
            thresh = " ◀" if key == ou_prediction else ""
            if key == ou_prediction and ((key == 'Over' and over_prob < 55) or (key == 'Under' and under_prob < 60)):
                thresh = " ◀ (below)"
            match = " ✓" if ml_ou.get('prediction') == key and w_ou.get('prediction') == key else ""
            
            print(f"    {col}{label:<12}{C.RESET}")
            print(f"      ML:{ml_bar} {ml_pct:5.1f}%")
            print(f"      W: {w_bar} {w_pct:5.1f}%")
            print(f"      Fair:{odds_c:.2f}  Mkt:{odds_m:.2f}{C.GREEN}{thresh}{match}{C.RESET}")
        
        # ── INJURY REPORT ──
        injuries_data = result.get('injuries', {}) if isinstance(result, dict) else {}
        if injuries_data:
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}🏥 INJURY REPORT{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            
            for team, players in injuries_data.items():
                if not players:
                    continue
                
                # Check if this is home or away team
                team_lower = team.lower()
                is_home = any(part in team_lower for part in home.lower().split())
                is_away = any(part in team_lower for part in away.lower().split())
                
                if not is_home and not is_away:
                    continue
                
                team_label = home if is_home else away
                emoji = "🏠" if is_home else "✈️"
                
                key_out = [p for p in players if 'will not play' in p.get('status', '').lower()]
                
                if key_out:
                    print(f"\n  {emoji} {C.RED}{team_label} - Key players OUT:{C.RESET}")
                    for p in key_out[:3]:
                        name = p.get('name', 'Unknown')
                        status = p.get('status', '')
                        print(f"    🔴 {name} - {status}")
                elif players:
                    count = len(players)
                    print(f"\n  {emoji} {team_label} - {count} injured player(s)")
                    for p in players[:2]:
                        name = p.get('name', 'Unknown')
                        status = p.get('status', '')
                        print(f"    ⚠️ {name} - {status}")
        
        # Narrative Summary
        print()
        print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
        print(f"  {C.BOLD}📝 PREDICTION SUMMARY{C.RESET}")
        print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
        
        # Match header
        print(f"  {C.CYAN}{home}{C.RESET} vs {C.MAGENTA}{away}{C.RESET}")
        league = md.get('match_info', {}).get('league', 'Unknown League')
        print(f"  {C.DIM}League: {league}{C.RESET}")
        print()
        
        # Get both model predictions
        ml_pred = result.get('ml_prediction', {})
        w_pred = result.get('weighted_prediction', {})
        
        # Show both model predictions clearly
        print(f"  {C.BOLD}┌─────────────────────────────────────────────────────────────┐{C.RESET}")
        print(f"  {C.BOLD}│              ML MODEL              │      WEIGHTED MODEL           │{C.RESET}")
        print(f"  {C.BOLD}├─────────────────────────────────────────────────────────────┤{C.RESET}")
        
        # Result
        ml_r = rmap.get(ml_pred.get('result', {}).get('prediction', '?'), '?')
        w_r = rmap.get(w_pred.get('result', {}).get('prediction', '?'), '?')
        ml_rc = ml_pred.get('result', {}).get('confidence', 0) * 100
        w_rc = w_pred.get('result', {}).get('confidence', 0) * 100
        r_agree = "✓" if ml_r == w_r else "⚠"
        print(f"  {C.BOLD}│{C.RESET} {ml_r:>12} ({ml_rc:>5.0f}%)             │ {w_r:>12} ({w_rc:>5.0f}%)             {r_agree} {C.BOLD}│{C.RESET}")
        
        # O/U
        ml_o = ml_pred.get('over_under', {}).get('prediction', '?')
        w_o = w_pred.get('over_under', {}).get('prediction', '?')
        ml_oc = ml_pred.get('over_under', {}).get('confidence', 0) * 100
        w_oc = w_pred.get('over_under', {}).get('confidence', 0) * 100
        o_agree = "✓" if ml_o == w_o else "⚠"
        print(f"  {C.BOLD}│{C.RESET} {ml_o:>12} ({ml_oc:>5.0f}%)             │ {w_o:>12} ({w_oc:>5.0f}%)             {o_agree} {C.BOLD}│{C.RESET}")
        print(f"  {C.BOLD}└─────────────────────────────────────────────────────────────┘{C.RESET}")
        print()
        
        # Final recommendation based on what the system chose
        result_agreement = convergence.get('result_match', True) if convergence else True
        ou_agreement = convergence.get('ou_match', True) if convergence else True
        
        print(f"  {C.BOLD}🎯 FINAL RECOMMENDATION:{C.RESET}")
        
        # Sort recommendations by confidence
        recommendations = []
        
        # Check model agreement
        result_agreement = convergence.get('result_match', True) if convergence else True
        ou_agreement = convergence.get('ou_match', True) if convergence else True
        
        # Add result recommendation (reduce confidence if models disagree)
        result_conf = rp.get('confidence', 0) * 100
        if not result_agreement:
            result_conf = result_conf * 0.6  # Reduce confidence when models disagree
        result_label = rmap.get(rp['prediction'], '?')
        result_conf_label = 'HIGH' if result_conf > 55 else ('MEDIUM' if result_conf > 40 else 'LOW')
        recommendations.append({
            'type': 'Match Result',
            'label': result_label,
            'confidence': result_conf,
            'confidence_label': result_conf_label,
            'prediction': rp['prediction']
        })
        
        # Add O/U recommendation (reduce confidence if models disagree)
        ou_conf = op.get('confidence', 0) * 100
        if not ou_agreement:
            ou_conf = ou_conf * 0.6  # Reduce confidence when models disagree
        ou_conf_label = 'HIGH' if ou_conf > 55 else ('MEDIUM' if ou_conf > 40 else 'LOW')
        recommendations.append({
            'type': 'Over/Under 2.5',
            'label': f"{ou_prediction} 2.5 Goals",
            'confidence': ou_conf,
            'confidence_label': ou_conf_label,
            'prediction': ou_prediction
        })
        
        # Sort by confidence (highest first)
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Print recommendations in order
        for i, rec in enumerate(recommendations):
            emoji = "🎯" if i == 0 else "  "
            conf_label = rec.get('confidence_label', "HIGH" if rec['confidence'] >= 60 else ("MEDIUM" if rec['confidence'] >= 40 else "LOW"))
            conf_col = C.GREEN if conf_label == "HIGH" else (C.YELLOW if conf_label == "MEDIUM" else C.RED)
            print(f"  {emoji} {rec['type']}: {C.BOLD}{rec['label']}{C.RESET} ({conf_col}{rec['confidence']:.0f}% {conf_label}{C.RESET})")
        
        # If models disagree, explain why
        if not result_agreement or not ou_agreement:
            print()
            print(f"  {C.BOLD}{C.YELLOW}⚠️ WHY MODELS DIVERGE:{C.RESET}")
            
            if not result_agreement:
                # Get ML and Weighted predictions
                ml_r = ml_pred.get('result', {}).get('prediction', '?')
                w_r = w_pred.get('result', {}).get('prediction', '?')
                ml_r_prob = ml_pred.get('result', {}).get('probabilities', {}).get(ml_r, 0) * 100
                w_r_prob = w_pred.get('result', {}).get('probabilities', {}).get(w_r, 0) * 100
                
                if ml_r == '1':
                    print(f"    • ML: Home win ({ml_r_prob:.0f}%) based on form & stats")
                elif ml_r == 'X':
                    print(f"    • ML: Draw ({ml_r_prob:.0f}%) based on recent matches")
                else:
                    print(f"    • ML: Away win ({ml_r_prob:.0f}%) based on league performance")
                
                if w_r == '1':
                    print(f"    • Weighted: Home win ({w_r_prob:.0f}%) based on odds value")
                elif w_r == 'X':
                    print(f"    • Weighted: Draw ({w_r_prob:.0f}%) based on head-to-head")
                else:
                    print(f"    • Weighted: Away win ({w_r_prob:.0f}%) based on market odds")
            
            if not ou_agreement:
                ml_o = ml_pred.get('over_under', {}).get('prediction', '?')
                w_o = w_pred.get('over_under', {}).get('prediction', '?')
                print(f"    • O/U: ML={ml_o}, Weighted={w_o} (different goal expectations)")
        
        # Value bets in summary
        if analysis.get('value_bets'):
            print()
            print(f"  {C.GREEN}💰 Value Bets:{C.RESET}")
            for vb in analysis['value_bets'][:2]:
                lbl = {'1': f'{home} Win', 'X': 'Draw', '2': f'{away} Win'}.get(vb['outcome'], vb['outcome'])
                print(f"    • {lbl}: {C.BOLD}{vb['market']:.2f}{C.RESET} (fair: {vb['fair']:.2f}, +{vb['value_pct']:.0f}% value)")
        
        # Generate detailed narrative for the top recommendation
        narrative_lines = []
        
        # Get top recommendation
        top_rec = recommendations[0] if recommendations else {}
        
        # Get model predictions for summary
        ml_result = pred.get('ml_prediction', {}).get('result', {}).get('prediction', 'N/A')
        weighted_result = pred.get('weighted_prediction', {}).get('result', {}).get('prediction', 'N/A')
        ml_ou = pred.get('ml_prediction', {}).get('over_under', {}).get('prediction', 'N/A')
        weighted_ou = pred.get('weighted_prediction', {}).get('over_under', {}).get('prediction', 'N/A')
        
        # Model agreement status
        result_match = pred.get('convergence', {}).get('result_match', True)
        ou_match = pred.get('convergence', {}).get('ou_match', True)
        
        # Show prediction summary with model breakdown
        narrative_lines.append("")
        narrative_lines.append(f"  {C.BOLD}┌────────────────────────────────────────────────────────┐{C.RESET}")
        narrative_lines.append(f"  {C.BOLD}│           PREDICTION SUMMARY                     │{C.RESET}")
        narrative_lines.append(f"  {C.BOLD}├────────────────────────────────────────────────────────┤{C.RESET}")
        
        # Result prediction
        res_agree_emoji = C.GREEN + "✓" + C.RESET if result_match else C.YELLOW + "⚠" + C.RESET
        narrative_lines.append(f"  {C.BOLD}│{C.RESET} Match Result: {home} vs {away}")
        narrative_lines.append(f"  {C.BOLD}│{C.RESET}   ML Model:      {ml_result}")
        narrative_lines.append(f"  {C.BOLD}│{C.RESET}   Weighted:      {weighted_result}")
        narrative_lines.append(f"  {C.BOLD}│{C.RESET}   Agreement:     {res_agree_emoji} {"AGREE" if result_match else "DISAGREE"}")
        
        # O/U prediction
        ou_agree_emoji = C.GREEN + "✓" + C.RESET if ou_match else C.YELLOW + "⚠" + C.RESET
        narrative_lines.append(f"  {C.BOLD}│{C.RESET}")
        narrative_lines.append(f"  {C.BOLD}│{C.RESET}   Over/Under:   {ml_ou} vs {weighted_ou}")
        narrative_lines.append(f"  {C.BOLD}│{C.RESET}   Agreement:     {ou_agree_emoji} {"AGREE" if ou_match else "DISAGREE"}")
        
        # Final recommendation
        narrative_lines.append(f"  {C.BOLD}├────────────────────────────────────────────────────────┤{C.RESET}")
        final_result = top_rec.get('prediction', top_rec.get('label', 'N/A'))
        final_conf = top_rec.get('confidence', 0)
        final_type = top_rec.get('type', '')
        narrative_lines.append(f"  {C.BOLD}│{C.RESET} {final_type}: {C.BOLD}{final_result}{C.RESET} ({final_conf:.0f}%)")
        narrative_lines.append(f"  {C.BOLD}└────────────────────────────────────────────────────────┘{C.RESET}")
        
        # ── KEY FACTORS (Narrative) ──
        if analysis.get('key_factors'):
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}🔑 KEY FACTORS{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            for factor in analysis['key_factors'][:5]:  # Show top 5 factors
                print(f"  {factor}")
        
        # Confidence
        print()
        conf = analysis['confidence']
        conf_col = C.GREEN if conf == 'HIGH' else (C.YELLOW if conf == 'MEDIUM' else C.RED)
        print(f"  {C.BOLD}Confidence Level:{C.RESET}  {conf_col}{C.BOLD}{conf}{C.RESET}")
        
        # Convergence summary in prediction summary
        if 'convergence' in result and isinstance(result, dict):
            conv = result['convergence']
            r_match = conv.get('result_match', False)
            o_match = conv.get('ou_match', False)
            
            ml_r = conv.get('ml_result', '?')
            w_r = conv.get('weighted_result', '?')
            ml_o = conv.get('ml_ou', '?')
            w_o = conv.get('weighted_ou', '?')
            
            # Calculate convergence degrees
            ml_probs = result.get('ml_prediction', {}).get('result', {}).get('probabilities', {})
            w_probs = result.get('weighted_prediction', {}).get('result', {}).get('probabilities', {})
            
            if ml_probs and w_probs:
                prob_diff = sum(abs(ml_probs.get(k, 0) - w_probs.get(k, 0)) for k in ['1', 'X', '2']) / 3 * 100
                r_deg = max(0, 100 - prob_diff)
            else:
                r_deg = 0
            
            ml_ou_probs = result.get('ml_prediction', {}).get('over_under', {}).get('probabilities', {})
            w_ou_probs = result.get('weighted_prediction', {}).get('over_under', {}).get('probabilities', {})
            
            if ml_ou_probs and w_ou_probs:
                ou_diff = sum(abs(ml_ou_probs.get(k, 0) - w_ou_probs.get(k, 0)) for k in ['Over', 'Under']) / 2 * 100
                o_deg = max(0, 100 - ou_diff)
            else:
                o_deg = 0
            
            print(f"  {C.BOLD}Model Agreement:{C.RESET}")
            if r_match:
                print(f"    {C.GREEN}✓{C.RESET} Both models predict: {ml_r}")
            else:
                print(f"    {C.YELLOW}⚠{C.RESET} ML predicts: {ml_r}, Weighted predicts: {w_r}")
            
        # Show prediction method
        pred_method = pred.get('prediction_method', 'unknown')
        league = md.get('match_info', {}).get('league', 'Unknown League')
        country = md.get('match_info', {}).get('country', '')
        league_code = md.get('match_info', {}).get('league_code', '')
        
        # Check if ML model is actually trained (either main or league-specific or global)
        ml_trained = (
            (self.predictor.result_model is not None and self.predictor.ou_model is not None and self.predictor.scaler is not None) or
            len(self.predictor.league_models) > 0
        )
        
        # Get league-specific model info
        league_model_info = None
        if league_code:
            # Check if we have a league-specific model
            for model_key in self.predictor.league_models:
                if model_key.startswith(league_code + '_') or model_key == league_code:
                    league_model_info = model_key
                    break
        
        # Determine which model was actually used
        if pred_method in ['ml', 'league_ml']:
            if league_model_info:
                print(f"  {C.GREEN}✓ League ML Model{C.RESET} - Using league-specific model for {league_code}")
            elif 'Global_Model' in self.predictor.league_models:
                print(f"  {C.GREEN}✓ Global ML Model{C.RESET} - Using model trained on all leagues (77.5% accuracy)")
            elif ml_trained:
                training_count = pred.get('training_examples', len(self.storage.get_league_training_data(league)))
                print(f"  {C.GREEN}✓ ML Model Trained{C.RESET} - Using learned patterns from {training_count} matches")
            else:
                print(f"  {C.GREEN}✓ ML Model{C.RESET} - prediction_method: {pred_method}")
        elif pred_method == 'weighted_factors':
            print(f"  {C.YELLOW}⚠ Weighted Factors Model{C.RESET} - Using statistical analysis (no ML model available)")
        elif pred_method == 'statistical':
            print(f"  {C.YELLOW}⚠ Fallback Statistical Model{C.RESET} - No ML model trained, using odds/form analysis for {league}")
        else:
            print(f"  {C.DIM}ℹ Model: {pred_method}{C.RESET}")
        
        # Show league model info if available
        if league_model_info:
            # Load metadata for this league model
            import os
            import json
            model_dir = os.path.join('models', league_model_info)
            metadata_path = os.path.join(model_dir, 'metadata.json')
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    example_count = metadata.get('example_count', 0)
                    trained_at = metadata.get('trained_at', 'Unknown')
                    league_info = metadata.get('league_info', {})
                    print(f"  {C.CYAN}📊 League Model:{C.RESET} {league_info.get('country', country)} {league_info.get('league', league)}")
                    print(f"  {C.DIM}  Trained on {example_count} matches on {trained_at[:10]}{C.RESET}")
                except:
                    pass
        
        # Show accuracy if available
        if pred.get('result_accuracy'):
            print(f"  {C.DIM}  Model accuracy: {pred['result_accuracy']:.1%}{C.RESET}")
        

        # Edit league name option
        import re
        url = result.get('url', '')
        # Use league_code from match_info (from leagues_db lookup)
        info = result.get('match_info', {})
        league_code = info.get('league_code')
        current_league = info.get('league', 'Unknown')
        
        if league_code:
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}✏️  EDIT LEAGUE NAME{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.DIM}Current league: {current_league}{C.RESET}")
            print(f"  {C.DIM}League code: {league_code}{C.RESET}")
            new_league = input(f"  {C.BOLD}Enter correct league name (or press Enter to skip): {C.RESET}").strip()
            
            if new_league:
                from football_scraper import ForebetScraper
                scraper = ForebetScraper()
                if scraper.update_league_name(league_code, new_league):
                    updated = self.storage.update_league_name(current_league, new_league)
                    print(f"  {C.GREEN}✓ Updated league from '{current_league}' to '{new_league}'{C.RESET}")
                    print(f"  {C.DIM}  Updated {updated} entries across all data files{C.RESET}")
                else:
                    print(f"  {C.RED}✗ Failed to update league name{C.RESET}")


# ======================================================================
# CLI
# =====================================================================═

def main():
    parser = argparse.ArgumentParser(description='Football Match Prediction System')
    parser.add_argument('command', choices=['predict', 'result', 'train', 'stats', 'batch', 'update-league', 'export'])
    parser.add_argument('--url', help='Forebet match URL or path to file with URLs')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--json', action='store_true', help='Output raw JSON')
    parser.add_argument('--code', help='League code (for update-league command)')
    parser.add_argument('--name', help='League name (for update-league command)')
    parser.add_argument('--output', '-o', help='Output file for export (default: predictions.xlsx)')
    parser.add_argument('--today', action='store_true', help='Export only today\'s predictions')
    args = parser.parse_args()
    
    # Handle update-league command
    if args.command == 'update-league':
        code = args.code
        name = args.name
        if not code or not name:
            print("Error: --code and --name required for update-league")
            return
        
        # Update unknown_leagues.json
        try:
            with open('data/unknown_leagues.json', 'r') as f:
                unknown = json.load(f)
        except:
            unknown = {}
        
        old_name = unknown.get(code, f"League {code}")
        unknown[code] = name
        
        with open('data/unknown_leagues.json', 'w') as f:
            json.dump(unknown, f, indent=2)
        
        print(f"✓ Updated: {code} = {name}")
        
        # Update training data if league name changed
        if old_name != name:
            updated = system.storage.update_league_name(old_name, name)
            if updated:
                print(f"  Updated {updated} training examples from '{old_name}' to '{name}'")
        return

    system = FootballPredictionSystem()

    def process_url(url: str, is_single: bool = True):
        """Process a single URL."""
        if args.command == 'predict':
            result = system.predict_match(url, save_data=not args.no_save)
            if 'error' in result:
                print(f"✗ Error processing {url}: {result['error']}")
                return False
            if args.json:
                out = {k: v for k, v in result.items() if k != 'match_data'}
                print(json.dumps(out, indent=2, default=str))
            else:
                system.display_prediction(result)
            # Save prediction to predictions directory
            os.makedirs(PREDICTIONS_DIR, exist_ok=True)
            fname = os.path.join(PREDICTIONS_DIR, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(fname, 'w') as f:
                # Include match_info and teams from match_data for export
                match_data = result.get('match_data', {})
                out = {
                    'url': url,
                    'timestamp': datetime.now().isoformat(),
                    'match_info': {
                        'home_team': match_data.get('teams', {}).get('home', ''),
                        'away_team': match_data.get('teams', {}).get('away', ''),
                        'date': match_data.get('match_info', {}).get('date', ''),
                        'time': match_data.get('match_info', {}).get('time', ''),
                        'league': match_data.get('match_info', {}).get('league', ''),
                    },
                    'teams': match_data.get('teams', {}),
                    'prediction': result.get('prediction', {}),
                    'ml_prediction': result.get('ml_prediction', {}),
                    'weighted_prediction': result.get('weighted_prediction', {}),
                }
                json.dump(out, f, indent=2, default=str)
            # Also save to predictions.json for accuracy tracking
            try:
                system.storage.save_prediction(url, result.get('prediction', {}))
                print("✓ Prediction saved for accuracy tracking")
            except Exception as e:
                print(f"Warning: Could not save prediction for tracking: {e}")
            return True
        
        elif args.command == 'result':
            ok = system.add_match_result(url)
            if ok:
                print(f"✓ Result saved: {url}")
            else:
                print(f"✗ Failed: {url}")
            return ok
        
        return False

    def load_urls_from_file(filepath: str) -> list:
        """Load URLs from a text file (one per line)."""
        urls = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        urls.append(line)
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
        except Exception as e:
            print(f"Error reading file: {e}")
        return urls

    def add_url_to_queue(url: str, queue_file: str = "results.txt"):
        """Add URL to the queue for later learning."""
        try:
            # Check if URL already in file
            if os.path.exists(queue_file):
                with open(queue_file, 'r') as f:
                    existing = f.read()
                if url in existing:
                    return False  # Already queued
            
            # Add to file
            with open(queue_file, 'a') as f:
                f.write(f"{url}\n")
            return True
        except Exception as e:
            print(f"Error adding URL to queue: {e}")
            return False

    @staticmethod
    def clear_queue(queue_file: str = "results.txt"):
        """Clear the URL queue file contents but keep the file."""
        try:
            with open(queue_file, 'w') as f:
                f.write("")  # Write empty string to clear contents
        except Exception as e:
            print(f"Error clearing queue: {e}")

    def process_url(url: str, is_single: bool = True):
        """Process a single URL."""
        if args.command == 'predict':
            result = system.predict_match(url, save_data=not args.no_save)
            if 'error' in result:
                print(f"✗ Error processing {url}: {result['error']}")
                return False
            
            # Auto-add to queue for learning
            if not args.no_save:
                if add_url_to_queue(url):
                    print(f"\n✓ Added to learning queue (results.txt)")
            
            if args.json:
                out = {k: v for k, v in result.items() if k != 'match_data'}
                print(json.dumps(out, indent=2, default=str))
            else:
                system.display_prediction(result)
            # Save prediction to predictions directory
            os.makedirs(PREDICTIONS_DIR, exist_ok=True)
            fname = os.path.join(PREDICTIONS_DIR, f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(fname, 'w') as f:
                # Include match_info and teams from match_data for export
                match_data = result.get('match_data', {})
                out = {
                    'url': url,
                    'timestamp': datetime.now().isoformat(),
                    'match_info': {
                        'home_team': match_data.get('teams', {}).get('home', ''),
                        'away_team': match_data.get('teams', {}).get('away', ''),
                        'date': match_data.get('match_info', {}).get('date', ''),
                        'time': match_data.get('match_info', {}).get('time', ''),
                        'league': match_data.get('match_info', {}).get('league', ''),
                    },
                    'teams': match_data.get('teams', {}),
                    'prediction': result.get('prediction', {}),
                    'ml_prediction': result.get('ml_prediction', {}),
                    'weighted_prediction': result.get('weighted_prediction', {}),
                }
                json.dump(out, f, indent=2, default=str)
            # Also save to predictions.json for accuracy tracking
            try:
                system.storage.save_prediction(url, result.get('prediction', {}))
                print("✓ Prediction saved for accuracy tracking")
            except Exception as e:
                print(f"Warning: Could not save prediction for tracking: {e}")
            return True
        
        elif args.command == 'result':
            ok = system.add_match_result(url)
            if ok:
                print(f"✓ Result saved: {url}")
            else:
                print(f"✗ Failed: {url}")
            return ok
        
        return False

    if args.command in ['predict', 'result']:
        if not args.url:
            print("Error: --url required")
            return
        
        # Check if --url is a file
        if os.path.isfile(args.url):
            urls = load_urls_from_file(args.url)
            if not urls:
                print("No URLs found in file")
                return
            
            print(f"\n{'='*60}")
            print(f"Processing {len(urls)} URLs from {args.url}")
            print(f"{'='*60}\n")
            
            success = 0
            for i, url in enumerate(urls, 1):
                print(f"\n[{i}/{len(urls)}] Processing: {url}")
                if process_url(url, is_single=False):
                    success += 1
            
            print(f"\n{'='*60}")
            print(f"Completed: {success}/{len(urls)} URLs processed successfully")
            print(f"{'='*60}")
        else:
            # Single URL processing
            process_url(args.url, is_single=True)

    elif args.command == 'export':
        # Export predictions to Excel
        import glob
        
        # Find all prediction files
        pattern = os.path.join(PREDICTIONS_DIR, "prediction_*.json")
        files = glob.glob(pattern)
        
        if not files:
            print("No predictions found to export")
            return
        
        # Load all predictions
        all_predictions = []
        for f in files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    # Extract key fields - handle both old and new formats
                    # Try 'prediction' first, then 'our_prediction'
                    pred = data.get('prediction', {}) or data.get('our_prediction', {})
                    match_info = data.get('match_info', {})
                    teams = data.get('teams', {})
                    
                    # Get home/away from either match_info or teams
                    home_team = match_info.get('home_team', '') or teams.get('home', '')
                    away_team = match_info.get('away_team', '') or teams.get('away', '')
                    date = match_info.get('date', '')
                    time = match_info.get('time', '')
                    league = match_info.get('league', '')
                    
                    # Get result prediction
                    result_pred = pred.get('result', {})
                    ou_pred = pred.get('over_under', {})
                    
                    # Get probabilities
                    result_probs = result_pred.get('probabilities', {})
                    ou_probs = ou_pred.get('probabilities', {})
                    
                    # Get ML and weighted predictions
                    ml_pred = data.get('ml_prediction', {})
                    w_pred = data.get('weighted_prediction', {})
                    
                    all_predictions.append({
                        'Date': date,
                        'Time': time,
                        'Home Team': home_team,
                        'Away Team': away_team,
                        'League': league,
                        'Result Prediction': result_pred.get('prediction', ''),
                        'Result Confidence': f"{result_pred.get('confidence', 0)*100:.0f}%",
                        'Home Win Prob': f"{result_probs.get('1', 0)*100:.1f}%",
                        'Draw Prob': f"{result_probs.get('X', 0)*100:.1f}%",
                        'Away Win Prob': f"{result_probs.get('2', 0)*100:.1f}%",
                        'O/U Prediction': ou_pred.get('prediction', ''),
                        'O/U Confidence': f"{ou_pred.get('confidence', 0)*100:.0f}%",
                        'Over 2.5 Prob': f"{ou_probs.get('Over', 0)*100:.1f}%",
                        'Under 2.5 Prob': f"{ou_probs.get('Under', 0)*100:.1f}%",
                        'ML Result': ml_pred.get('result', {}).get('prediction', ''),
                        'ML O/U': ml_pred.get('over_under', {}).get('prediction', ''),
                        'Weighted Result': w_pred.get('result', {}).get('prediction', ''),
                        'Weighted O/U': w_pred.get('over_under', {}).get('prediction', ''),
                        'URL': data.get('url', ''),
                    })
            except Exception as e:
                print(f"Warning: Could not load {f}: {e}")
        
        if not all_predictions:
            print("No valid predictions found")
            return
        
        # Sort by date/time
        all_predictions.sort(key=lambda x: (x.get('Date', ''), x.get('Time', '')))
        
        # Print summary table
        print(f"\n{'='*100}")
        print(f"PREDICTIONS SUMMARY ({len(all_predictions)} matches)")
        print(f"{'='*100}\n")
        
        # Header
        print(f"{'Date':<12} {'Home':<20} {'Away':<20} {'Result':<15} {'O/U':<15}")
        print(f"{'-'*12} {'-'*20} {'-'*20} {'-'*15} {'-'*15}")
        
        for p in all_predictions:
            date = p['Date'][:10] if p['Date'] else ''
            home = p['Home Team'][:18] if len(p['Home Team']) > 18 else p['Home Team']
            away = p['Away Team'][:18] if len(p['Away Team']) > 18 else p['Away Team']
            result = f"{p['Result Prediction']} ({p['Result Confidence']})"
            ou = f"{p['O/U Prediction']} ({p['O/U Confidence']})"
            print(f"{date:<12} {home:<20} {away:<20} {result:<15} {ou:<15}")
        
        # Export to Excel
        output_file = args.output or 'predictions.xlsx'
        try:
            import pandas as pd
            df = pd.DataFrame(all_predictions)
            df.to_excel(output_file, index=False)
            print(f"\n{'='*100}")
            print(f"Exported {len(all_predictions)} predictions to {output_file}")
            print(f"{'='*100}")
        except ImportError:
            # Fallback to CSV if pandas not available
            output_file = output_file.replace('.xlsx', '.csv')
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_predictions[0].keys())
                writer.writeheader()
                writer.writerows(all_predictions)
            print(f"\n{'='*100}")
            print(f"Exported {len(all_predictions)} predictions to {output_file} (CSV - pandas not available)")
            print(f"{'='*100}")
        else:
            # Single URL
            process_url(args.url, is_single=True)

    elif args.command == 'train':
        metrics = system.train_model()
        print(json.dumps(metrics, indent=2))
        
        # Clear the queue after training
        print(f"\n✓ Learning complete - clearing results.txt")
        clear_queue()

    elif args.command == 'stats':
        print(json.dumps(system.get_statistics(), indent=2))

    elif args.command == 'batch':
        # Batch prediction mode - requires a file with URLs
        if not args.url:
            print("Error: --url required (file with URLs for batch prediction)")
            return
        
        if not os.path.isfile(args.url):
            print(f"Error: File not found: {args.url}")
            return
        
        urls = load_urls_from_file(args.url)
        if not urls:
            print("No URLs found in file")
            return
        
        print(f"\n{'='*60}")
        print(f"Batch Prediction: {len(urls)} matches")
        print(f"{'='*60}\n")
        
        predictions = []
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] {url}")
            result = system.predict_match(url, save_data=True)
            if 'error' not in result:
                teams = result.get('match_data', {}).get('teams', {})
                pred = result.get('prediction', {}).get('result', {}).get('prediction', '?')
                conf = result.get('prediction', {}).get('result', {}).get('confidence', 0)
                print(f"  → {teams.get('home', '?')} vs {teams.get('away', '?')}: {pred} ({conf*100:.0f}%)")
                predictions.append({
                    'url': url,
                    'home': teams.get('home'),
                    'away': teams.get('away'),
                    'prediction': pred,
                    'confidence': conf,
                })
            else:
                print(f"  → Error: {result.get('error')}")
        
        # Save batch predictions to predictions directory
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        fname = os.path.join(PREDICTIONS_DIR, f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(fname, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        print(f"\n✓ Saved {len(predictions)} predictions to {fname}")


if __name__ == "__main__":
    main()
