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
        match_data = self.scraper.scrape_match(url)
        if not match_data:
            return {'error': 'Failed to scrape match data', 'url': url}

        if save_data:
            # Use save_match_with_result to also update training data if result available
            self.storage.save_match_with_result(match_data)

        features = FootballPredictor.extract_features(match_data)
        
        # Extract league for league-specific prediction
        league = match_data.get('info', {}).get('league')
        
        # Get ML prediction
        ml_prediction = self.predictor.predict(features, league)
        
        # Generate weighted prediction
        weighted_prediction = self.weighted_predictor.predict(match_data, features)
        
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

        return {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'match_data': match_data,
            'features': features,
            'prediction': primary_prediction,
            'ml_prediction': ml_prediction,
            'weighted_prediction': weighted_prediction,
            'convergence': convergence,
            'analysis': self._analyse(match_data, features, primary_prediction),
        }

    def add_match_result(self, url: str) -> bool:
        result = self.scraper.extract_actual_result(url)
        if not result or not result.get('result'):
            print("Could not extract result – match may not have been played yet.")
            return False
        return self.storage.save_match_result(url, result)

    def train_model(self) -> Dict:
        # Use get_league_training_data() to get flat list of all examples
        td = self.storage.get_league_training_data()
        if len(td) < 10:
            return {'error': 'Insufficient training data', 'available': len(td), 'required': 10}
        return self.predictor.train(td)

    def get_statistics(self) -> Dict:
        ss = self.storage.get_statistics()
        return {**ss, 'model_trained': self.predictor.result_model is not None}

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyse(self, md: Dict, feats: Dict, pred: Dict) -> Dict:
        analysis: Dict = {'key_factors': [], 'value_bets': [], 'confidence': 'medium'}

        # Form
        hf = md.get('form', {}).get('home', [])
        af = md.get('form', {}).get('away', [])
        hw = hf.count('W')
        aw = af.count('W')
        if hw >= 4:
            analysis['key_factors'].append(f"🔥 Home team in excellent form ({hw} wins in last {len(hf)})")
        elif hw <= 1:
            analysis['key_factors'].append(f"⚠️  Home team in poor form ({hw} win in last {len(hf)})")
        if aw >= 4:
            analysis['key_factors'].append(f"🔥 Away team in excellent form ({aw} wins in last {len(af)})")
        elif aw <= 1:
            analysis['key_factors'].append(f"⚠️  Away team in poor form ({aw} win in last {len(af)})")

        # Home/away record
        hwr = feats.get('home_home_win_rate')
        if hwr and hwr >= 70:
            analysis['key_factors'].append(f"🏟️  Home team dominant at home ({hwr:.0f}% win rate)")
        awr = feats.get('away_away_win_rate')
        if awr and awr <= 20:
            analysis['key_factors'].append(f"📉 Away team struggles away ({awr:.0f}% win rate)")

        # Goals
        etg = feats.get('expected_total_goals', 2.5)
        if etg > 3.0:
            analysis['key_factors'].append(f"⚽ High-scoring match expected (avg {etg:.1f} goals)")
        elif etg < 2.0:
            analysis['key_factors'].append(f"🛡️  Low-scoring match expected (avg {etg:.1f} goals)")

        # Standings
        hp = feats.get('home_position')
        ap = feats.get('away_position')
        if hp and ap and hp < ap - 5:
            analysis['key_factors'].append(f"📊 Home team significantly higher in table (#{hp} vs #{ap})")
        elif hp and ap and ap < hp - 5:
            analysis['key_factors'].append(f"📊 Away team significantly higher in table (#{ap} vs #{hp})")

        # H2H
        h2h = md.get('head_to_head', {}).get('summary', {})
        if h2h.get('home_win_pct', 0) >= 60:
            analysis['key_factors'].append(f"📜 Home team dominates H2H ({h2h['home_win_pct']}% wins)")
        elif h2h.get('away_win_pct', 0) >= 60:
            analysis['key_factors'].append(f"📜 Away team dominates H2H ({h2h['away_win_pct']}% wins)")

        # Trends
        for t in md.get('trends', []):
            if t.get('percentage', 0) >= 90:
                analysis['key_factors'].append(f"📈 {t['description']} ({t['record']} = {t['percentage']}%)")

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

        # Confidence
        rc = pred['result']['confidence']
        oc = pred['over_under']['confidence']
        avg = (rc + oc) / 2
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
            print(f"  {C.CYAN}{home:<18}{C.RESET} {hg.get('scored','?'):>8} {hg.get('conceded','?'):>10} "
                  f"{hg.get('scored_avg','?'):>9} {hg.get('conceded_avg','?'):>9}")
            print(f"  {C.MAGENTA}{away:<18}{C.RESET} {ag.get('scored','?'):>8} {ag.get('conceded','?'):>10} "
                  f"{ag.get('scored_avg','?'):>9} {ag.get('conceded_avg','?'):>9}")

        # ── Home / Away Performance ──
        hm = md.get('home_matches', [])
        am = md.get('away_matches', [])
        if hm or am:
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}🏟️  HOME / AWAY PERFORMANCE{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            if hm:
                hwr = feats.get('home_home_win_rate', 0)
                print(f"  {C.CYAN}{home} at HOME{C.RESET} (win rate: {C.BOLD}{hwr:.0f}%{C.RESET})")
                for m in hm[:6]:
                    sc = f"{m['home_score']}-{m['away_score']}"
                    ht = f"({m['ht_home']}-{m['ht_away']})"
                    print(f"    {C.DIM}{m['date']}{C.RESET}  {m['home_team']} {C.BOLD}{sc}{C.RESET} {ht} {m['away_team']} {C.DIM}{m.get('competition','')}{C.RESET}")
            if am:
                awr = feats.get('away_away_win_rate', 0)
                print(f"  {C.MAGENTA}{away} AWAY{C.RESET} (win rate: {C.BOLD}{awr:.0f}%{C.RESET})")
                for m in am[:6]:
                    sc = f"{m['home_score']}-{m['away_score']}"
                    ht = f"({m['ht_home']}-{m['ht_away']})"
                    print(f"    {C.DIM}{m['date']}{C.RESET}  {m['home_team']} {C.BOLD}{sc}{C.RESET} {ht} {m['away_team']} {C.DIM}{m.get('competition','')}{C.RESET}")

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
                return arr[0] if isinstance(arr, list) and arr else '?'
            hgames = gs.get('home', {}).get('games') or 1
            agames = gs.get('away', {}).get('games') or 1
            print(f"  {'':18} {C.CYAN}{home:>15}{C.RESET}  {C.MAGENTA}{away:>15}{C.RESET}")
            print(f"  {'Shots/game':18} {_s(hs_stats.get('shots_total',[])):>10}/{hgames:<4}  "
                  f"{_s(as_stats.get('shots_total',[])):>10}/{agames:<4}")
            print(f"  {'On target':18} {_s(hs_stats.get('shots_on_target',[])):>15}  "
                  f"{_s(as_stats.get('shots_on_target',[])):>15}")
            bp_h = hs_stats.get('ball_poss', [])
            bp_a = as_stats.get('ball_poss', [])
            if bp_h or bp_a:
                print(f"  {'Possession':18} {C.BOLD}{_s(bp_h)}%{C.RESET:>13}  {C.BOLD}{_s(bp_a)}%{C.RESET:>13}")
            pa_h = hs_stats.get('passes_accurate', [0])
            pt_h = hs_stats.get('passes_total', [1])
            pa_a = as_stats.get('passes_accurate', [0])
            pt_a = as_stats.get('passes_total', [1])
            if isinstance(pt_h, list) and pt_h[0]:
                acc_h = _s(pa_h) / _s(pt_h) * 100 if _s(pt_h) else 0
                acc_a = _s(pa_a) / _s(pt_a) * 100 if _s(pt_a) else 0
                print(f"  {'Pass accuracy':18} {acc_h:>14.0f}%  {acc_a:>14.0f}%")

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
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}🔑 KEY FACTORS (WEIGHTED ANALYSIS){C.RESET}")
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
        
        # ── CONVERGENCE ANALYSIS ──
        if 'convergence' in result and isinstance(result, dict):
            conv = result['convergence']
            print()
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            print(f"  {C.BOLD}🔗 CONVERGENCE ANALYSIS{C.RESET}")
            print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
            
            ml_r = conv.get('ml_result', '?')
            w_r = conv.get('weighted_result', '?')
            ml_conf = result.get('ml_prediction', {}).get('result', {}).get('confidence', 0) * 100
            w_conf = result.get('weighted_prediction', {}).get('result', {}).get('confidence', 0) * 100
            
            # Result convergence
            r_match = conv.get('result_match', False)
            r_emoji = "✅ AGREE" if r_match else "⚠️ DISAGREE"
            r_col = C.GREEN if r_match else C.YELLOW
            
            # Calculate probability difference for degree of convergence
            ml_probs = result.get('ml_prediction', {}).get('result', {}).get('probabilities', {})
            w_probs = result.get('weighted_prediction', {}).get('result', {}).get('probabilities', {})
            
            if ml_probs and w_probs:
                prob_diff = sum(abs(ml_probs.get(k, 0) - w_probs.get(k, 0)) for k in ['1', 'X', '2']) / 3 * 100
                convergence_degree = max(0, 100 - prob_diff)
            else:
                convergence_degree = 0
            
            print()
            print(f"  {C.CYAN}ML Model:{C.RESET}      {ml_r} ({ml_conf:.0f}% confidence)")
            print(f"  {C.YELLOW}Weighted Model:{C.RESET} {w_r} ({w_conf:.0f}% confidence)")
            print()
            print(f"  {C.BOLD}Agreement:{C.RESET} {r_emoji}")
            print(f"  {C.BOLD}Convergence Degree:{C.RESET} {convergence_degree:.0f}% ({_bar(convergence_degree)})")
            
            # O/U convergence
            ml_o = conv.get('ml_ou', '?')
            w_o = conv.get('weighted_ou', '?')
            o_match = conv.get('ou_match', False)
            o_emoji = "✅ AGREE" if o_match else "⚠️ DISAGREE"
            
            ml_ou_conf = result.get('ml_prediction', {}).get('over_under', {}).get('confidence', 0) * 100
            w_ou_conf = result.get('weighted_prediction', {}).get('over_under', {}).get('confidence', 0) * 100
            
            ml_ou_probs = result.get('ml_prediction', {}).get('over_under', {}).get('probabilities', {})
            w_ou_probs = result.get('weighted_prediction', {}).get('over_under', {}).get('probabilities', {})
            
            if ml_ou_probs and w_ou_probs:
                ou_diff = sum(abs(ml_ou_probs.get(k, 0) - w_ou_probs.get(k, 0)) for k in ['Over', 'Under']) / 2 * 100
                ou_convergence_degree = max(0, 100 - ou_diff)
            else:
                ou_convergence_degree = 0
            
            print()
            print(f"  {C.CYAN}ML Model O/U:{C.RESET}  {ml_o} ({ml_ou_conf:.0f}%)")
            print(f"  {C.YELLOW}Weighted O/U:{C.RESET}  {w_o} ({w_ou_conf:.0f}%)")
            print(f"  {C.BOLD}O/U Agreement:{C.RESET} {o_emoji} ({ou_convergence_degree:.0f}%)")
            
            # Overall confidence based on convergence
            c = sum([r_match, o_match])
            conf = "HIGH" if c == 2 else ("MEDIUM" if c == 1 else "LOW")
            col = C.GREEN if conf == "HIGH" else (C.YELLOW if conf == "MEDIUM" else C.RED)
            print()
            print(f"  {C.BOLD}Overall Prediction Confidence:{C.RESET} {col}{C.BOLD}{conf}{C.RESET}")
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
        
        # Narrative Summary
        print()
        print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
        print(f"  {C.BOLD}📝 PREDICTION SUMMARY{C.RESET}")
        print(f"  {C.BOLD}{'─' * (w - 4)}{C.RESET}")
        
        # Generate narrative
        narrative_lines = []
        
        # Result narrative
        if rp['prediction'] == '1':
            narrative_lines.append(f"{home} is favored to win based on the following key factors:")
        elif rp['prediction'] == '2':
            narrative_lines.append(f"{away} is favored to win based on the following key factors:")
        else:
            narrative_lines.append(f"A draw is predicted due to the following factors:")
        
        # Add key factors
        if pred.get('key_factors', {}).get('match_result'):
            for f in pred['key_factors']['match_result'][:3]:
                direction_name = home if f['direction'] == 'home' else away
                factor_descriptions = {
                    'league_position': 'superior league position',
                    'odds_analysis': 'favorable market odds',
                    'recent_form': 'strong recent form',
                    'top5_performance': 'good performance against top teams',
                    'goals_stats': 'better goals statistics',
                    'h2h': 'favorable head-to-head record',
                    'shots_possession': 'strong shots and possession metrics',
                }
                desc = factor_descriptions.get(f['factor'], f['factor'])
                narrative_lines.append(f"  • {direction_name}'s {desc} ({f['impact']}% impact)")
        
        # O/U narrative
        narrative_lines.append("")
        if ou_prediction == 'Over':
            narrative_lines.append(f"Over 2.5 goals is recommended ({over_prob:.0f}%) due to:")
        else:
            narrative_lines.append(f"Under 2.5 goals is recommended ({under_prob:.0f}%) due to:")
        
        # Add O/U factors
        if pred.get('key_factors', {}).get('over_under'):
            for f in pred['key_factors']['over_under']:
                narrative_lines.append(f"  • {f['factor']}: {f['value']}")
        
        # Print narrative
        for line in narrative_lines:
            print(f"  {line}")
        
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
                print(f"    {C.GREEN}✓{C.RESET} Both models predict: {ml_r} ({r_deg:.0f}% convergence)")
            else:
                print(f"    {C.YELLOW}⚠{C.RESET} ML predicts: {ml_r}, Weighted predicts: {w_r} ({r_deg:.0f}% convergence)")
            
            if o_match:
                print(f"    {C.GREEN}✓{C.RESET} Both models predict: {ml_o} O/U ({o_deg:.0f}% convergence)")
            else:
                print(f"    {C.YELLOW}⚠{C.RESET} ML predicts: {ml_o}, Weighted predicts: {w_o} ({o_deg:.0f}% convergence)")
        
        # Show prediction method
        pred_method = pred.get('prediction_method', 'unknown')
        league = md.get('match_info', {}).get('league', 'Unknown League')
        
        # Check if ML model is actually trained
        ml_trained = self.predictor.result_model is not None and self.predictor.ou_model is not None and self.predictor.scaler is not None
        
        if pred_method in ['ml', 'league_ml'] and ml_trained:
            training_count = pred.get('training_examples', len(self.storage.get_league_training_data(league)))
            print(f"  {C.GREEN}✓ ML Model Trained{C.RESET} - Using learned patterns from {training_count} matches in {league}")
        elif pred_method == 'weighted_factors':
            print(f"  {C.YELLOW}⚠ Statistical Model{C.RESET} - Using weighted factors analysis for {league}")
        elif pred_method == 'statistical':
            print(f"  {C.YELLOW}⚠ Fallback Statistical Model{C.RESET} - No ML model trained, using odds/form analysis for {league}")
        else:
            print(f"  {C.DIM}ℹ Model: {pred_method}{C.RESET}")
        
        # Show accuracy if available
        if pred.get('result_accuracy'):
            print(f"  {C.DIM}  Model accuracy: {pred['result_accuracy']:.1%}{C.RESET}")
        
        print()
        print()


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Football Match Prediction System')
    parser.add_argument('command', choices=['predict', 'result', 'train', 'stats', 'batch'])
    parser.add_argument('--url', help='Forebet match URL or path to file with URLs')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--json', action='store_true', help='Output raw JSON')
    args = parser.parse_args()

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
                out = {k: v for k, v in result.items() if k != 'match_data'}
                json.dump(out, f, indent=2, default=str)
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
                out = {k: v for k, v in result.items() if k != 'match_data'}
                json.dump(out, f, indent=2, default=str)
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
