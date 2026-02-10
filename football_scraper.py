#!/usr/bin/env python3
"""
Football Match Data Scraper for Forebet
Extracts comprehensive match data including team stats, form, odds, standings,
head-to-head, home/away performance, shots, passes, trends, and more.
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import os
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta


# League code mapping (5-digit codes extracted from URL)
# URL format: https://www.forebet.com/en/football/matches/team-name-2315994
# The suffix is 7 digits, first 5 (23159) is the unique league code
LEAGUE_CODES = {
    '23159': 'England Premier League',
    '23351': 'England Championship',
    '23161': 'Spain La Liga',
    '23531': 'Spain Segunda Division',
    '23441': 'Italy Serie A',
    '23551': 'Italy Serie B',
    '23331': 'France Ligue 2',
    '23341': 'Netherlands Eerste Divisie',
    '24111': 'Portugal Liga Portugal',
    '23171': 'Scotland Premiership',
    '24171': 'Argentina Liga Profesional',
    '24151': 'Colombia Primera A',
    '23579': 'Cyprus First Division',
    '23261': 'England League One',
    '23301': 'England League Two',
    '23391': 'England National League',
    '23401': 'England National League North',
    '23411': 'England National League South',
    '24091': 'England Isthmian League',
    '23481': 'England Premier League 2',
    '23431': 'England SPL Premier Division',
}

# File to store discovered but unnamed league codes
UNKNOWN_LEAGUES_FILE = "unknown_leagues.json"


def load_unknown_leagues() -> Dict[str, str]:
    """Load unknown league codes from file."""
    if os.path.exists(UNKNOWN_LEAGUES_FILE):
        try:
            with open(UNKNOWN_LEAGUES_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_unknown_league(league_code: str, url: str):
    """Save a discovered league code that needs naming."""
    unknown = load_unknown_leagues()
    if league_code not in unknown:
        unknown[league_code] = {
            'first_seen_url': url,
            'timestamp': datetime.now().isoformat(),
            'suggested_name': None
        }
        with open(UNKNOWN_LEAGUES_FILE, 'w') as f:
            json.dump(unknown, f, indent=2)
        print(f"  [INFO] New league code discovered: {league_code}")
        print(f"  [INFO] Added to {UNKNOWN_LEAGUES_FILE} - please add a name for it")


class ForebetScraper:
    """Scraper for extracting football match data from Forebet"""

    def __init__(self):
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_match(self, url: str) -> Optional[Dict]:
        """Scrape all available match data from a Forebet URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if match has been played and extract result
            result = self._extract_result_from_soup(soup)
            
            # Extract league from URL
            league, league_code = self._extract_league_from_url(url)
            
            # Track unknown league codes
            if league_code and league is None:
                save_unknown_league(league_code, url)
            
            match_data = {
                'url': url,
                'timestamp': time.time(),
                'teams': self._extract_teams(soup),
                'match_info': self._extract_match_info(soup, league),
                'standings': self._extract_standings(soup),
                'league_table': self._extract_league_table(soup),
                'form': self._extract_form(soup),
                'last_6_matches': self._extract_last_6_matches(soup),
                'home_matches': self._extract_home_matches(soup),
                'away_matches': self._extract_away_matches(soup),
                'h2h_matches': self._extract_h2h_matches(soup),
                'odds': self._extract_odds(soup),
                'predictions': self._extract_predictions(soup),
            }
            
            if result:
                match_data['actual_result'] = result
                
            return match_data
            
        except Exception as e:
            print(f"Error scraping match: {e}")
            return None

    # ------------------------------------------------------------------
    # League extraction
    # ------------------------------------------------------------------

    def _extract_league_from_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract league name and code from URL.
        
        Returns:
            Tuple of (league_name, league_code)
        """
        # URL format: https://www.forebet.com/en/football/matches/team-name-2315994
        # The suffix is 7 digits, first 5 is the unique league code
        match = re.search(r'-(\d+)$', url)
        if match:
            full_code = match.group(1)
            if len(full_code) >= 5:
                league_code = full_code[:5]
                return LEAGUE_CODES.get(league_code), league_code
        return None, None

    # ------------------------------------------------------------------
    # Extraction methods
    # ------------------------------------------------------------------

    def _extract_result_from_soup(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract actual match result if available."""
        try:
            score_elem = soup.find('div', class_='event_score')
            if not score_elem:
                return None
                
            score_text = score_elem.get_text(strip=True)
            scores = re.findall(r'(\d+)', score_text)
            
            if len(scores) >= 2:
                home_score = int(scores[0])
                away_score = int(scores[1])
                
                # Determine result
                if home_score > away_score:
                    result = '1'
                elif away_score > home_score:
                    result = '2'
                else:
                    result = 'X'
                
                return {
                    'home_score': home_score,
                    'away_score': away_score,
                    'result': result,
                    'total_goals': home_score + away_score,
                    'over_under_2_5': 'Over' if home_score + away_score > 2.5 else 'Under'
                }
            return None
        except Exception as e:
            print(f"Error extracting result: {e}")
            return None

    def _extract_teams(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract home and away team names."""
        teams = {'home': '???', 'away': '???'}
        try:
            home_elem = soup.find('div', class_='home')
            away_elem = soup.find('div', class_='away')
            
            if home_elem:
                home_name = home_elem.find('span', class_='tname')
                if home_name:
                    teams['home'] = home_name.get_text(strip=True)
                else:
                    teams['home'] = home_elem.get_text(strip=True)
                    
            if away_elem:
                away_name = away_elem.find('span', class_='tname')
                if away_name:
                    teams['away'] = away_name.get_text(strip=True)
                else:
                    teams['away'] = away_elem.get_text(strip=True)
        except Exception as e:
            print(f"Error extracting teams: {e}")
        return teams

    def _extract_match_info(self, soup: BeautifulSoup, league: str = None) -> Dict:
        """Extract match date, time, league, venue."""
        info = {'date': None, 'time': None, 'league': league, 'venue': None}
        try:
            date_elem = soup.find('span', class_='date')
            if date_elem:
                text = date_elem.get_text(strip=True)
                d = re.search(r'(\d{2}/\d{2}/\d{4})', text)
                t = re.search(r'(\d{2}:\d{2})', text)
                if d: info['date'] = d.group(1)
                if t: info['time'] = t.group(1)
                
            venue_elem = soup.find('span', class_='venue')
            if venue_elem:
                info['venue'] = venue_elem.get_text(strip=True)
                
        except Exception as e:
            print(f"Error extracting match info: {e}")
        return info

    def _extract_standings(self, soup: BeautifulSoup) -> Dict:
        standings = {'home': None, 'away': None, 'home_points': None, 'away_points': None}
        try:
            tc = soup.find('div', class_='teamtablesp_container')
            if tc:
                positions = re.findall(r'(\d+)(?:st|nd|rd|th)\s+place', tc.get_text())
                if len(positions) >= 2:
                    standings['home'] = int(positions[0])
                    standings['away'] = int(positions[1])
        except Exception as e:
            print(f"Error extracting standings: {e}")
        return standings

    def _extract_league_table(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract the full league standings table."""
        table = []
        try:
            pos_cells = soup.find_all('td', class_='std_pos')
            for pc in pos_cells:
                try:
                    row = pc.find_parent('tr')
                    cells = row.find_all('td')
                    if len(cells) >= 8:
                        team = cells[1].get_text(strip=True)
                        table.append({
                            'position': int(pc.get_text(strip=True)),
                            'team': team,
                            'played': int(cells[2].get_text(strip=True)),
                            'won': int(cells[3].get_text(strip=True)),
                            'drawn': int(cells[4].get_text(strip=True)),
                            'lost': int(cells[5].get_text(strip=True)),
                            'gf': int(cells[6].get_text(strip=True)),
                            'ga': int(cells[7].get_text(strip=True)),
                            'gd': int(cells[8].get_text(strip=True)) if len(cells) > 8 else int(cells[6].get_text(strip=True)) - int(cells[7].get_text(strip=True)),
                            'points': int(cells[9].get_text(strip=True)) if len(cells) > 9 else 0,
                        })
                except (ValueError, IndexError):
                    continue
        except Exception as e:
            print(f"Error extracting league table: {e}")
        return table

    def _extract_form(self, soup: BeautifulSoup) -> Dict:
        """Extract recent form for both teams (last 6 matches)."""
        form = {'home': [], 'away': []}
        try:
            home_form = soup.find('div', class_='home')
            if home_form:
                form_elem = home_form.find('div', class_='form')
                if form_elem:
                    results = form_elem.find_all('span', class_='p1')
                    form['home'] = [r.get_text(strip=True) for r in results[:6]]
                    
            away_form = soup.find('div', class_='away')
            if away_form:
                form_elem = away_form.find('div', class_='form')
                if form_elem:
                    results = form_elem.find_all('span', class_='p1')
                    form['away'] = [r.get_text(strip=True) for r in results[:6]]
        except Exception as e:
            print(f"Error extracting form: {e}")
        return form

    def _extract_last_6_matches(self, soup: BeautifulSoup) -> Dict:
        """Extract last 6 matches for both teams."""
        l6 = {'home': [], 'away': []}
        try:
            tc = soup.find('div', class_='last-matches')
            if tc:
                home_matches = tc.find('div', class_='home')
                away_matches = tc.find('div', class_='away')
                
                if home_matches:
                    for match in home_matches.find_all('tr')[1:7]:
                        cells = match.find_all('td')
                        if len(cells) >= 4:
                            l6['home'].append({
                                'date': cells[0].get_text(strip=True),
                                'home_team': cells[1].get_text(strip=True),
                                'away_team': cells[2].get_text(strip=True),
                                'home_score': int(cells[3].get_text(strip=True)),
                                'away_score': int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0,
                                'competition': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                            })
                            
                if away_matches:
                    for match in away_matches.find_all('tr')[1:7]:
                        cells = match.find_all('td')
                        if len(cells) >= 4:
                            l6['away'].append({
                                'date': cells[0].get_text(strip=True),
                                'home_team': cells[1].get_text(strip=True),
                                'away_team': cells[2].get_text(strip=True),
                                'home_score': int(cells[3].get_text(strip=True)),
                                'away_score': int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0,
                                'competition': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                            })
        except Exception as e:
            print(f"Error extracting last 6 matches: {e}")
        return l6

    def _extract_home_matches(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract home team's last 6 home matches."""
        return self._extract_team_matches(soup, 'home')

    def _extract_away_matches(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract away team's last 6 away matches."""
        return self._extract_team_matches(soup, 'away')

    def _extract_team_matches(self, soup: BeautifulSoup, team_type: str) -> List[Dict]:
        """Extract matches for a specific team type (home/away)."""
        matches = []
        try:
            tc = soup.find('div', class_=f'{team_type}-matches')
            if tc:
                for row in tc.find_all('tr')[1:7]:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        match = {
                            'date': cells[0].get_text(strip=True),
                            'home_team': cells[1].get_text(strip=True),
                            'away_team': cells[2].get_text(strip=True),
                            'home_score': int(cells[3].get_text(strip=True)),
                            'away_score': int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0,
                            'competition': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                        }
                        matches.append(match)
        except Exception as e:
            print(f"Error extracting {team_type} matches: {e}")
        return matches

    def _extract_h2h_matches(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract head-to-head matches."""
        h2h = []
        try:
            tc = soup.find('div', class_='h2h-matches')
            if tc:
                for row in tc.find_all('tr')[1:11]:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        match = {
                            'date': cells[0].get_text(strip=True),
                            'home_team': cells[1].get_text(strip=True),
                            'away_team': cells[2].get_text(strip=True),
                            'home_score': int(cells[3].get_text(strip=True)),
                            'away_score': int(cells[4].get_text(strip=True)) if len(cells) > 4 else 0,
                            'competition': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                        }
                        h2h.append(match)
        except Exception as e:
            print(f"Error extracting H2H matches: {e}")
        return h2h

    def _extract_odds(self, soup: BeautifulSoup) -> Dict:
        """Extract betting odds."""
        odds = {}
        try:
            tc = soup.find('div', class_='odds')
            if tc:
                home_odds = tc.find('div', class_='home')
                draw_odds = tc.find('div', class_='draw')
                away_odds = tc.find('div', class_='away')
                
                if home_odds:
                    odds['home'] = float(home_odds.get_text(strip=True))
                if draw_odds:
                    odds['draw'] = float(draw_odds.get_text(strip=True))
                if away_odds:
                    odds['away'] = float(away_odds.get_text(strip=True))
                    
                # Also try to find over/under odds
                ou = soup.find('div', class_='ou')
                if ou:
                    over_elem = ou.find('div', class_='over')
                    under_elem = ou.find('div', class_='under')
                    if over_elem:
                        odds['over'] = float(over_elem.get_text(strip=True))
                    if under_elem:
                        odds['under'] = float(under_elem.get_text(strip=True))
        except Exception as e:
            print(f"Error extracting odds: {e}")
        return odds

    def _extract_predictions(self, soup: BeautifulSoup) -> Dict:
        """Extract Forebet's predictions."""
        pred = {'home': 0, 'draw': 0, 'away': 0}
        try:
            tc = soup.find('div', class_='prediction')
            if tc:
                probs = tc.find_all('div', class_='proc')
                if len(probs) >= 3:
                    pred['home'] = float(probs[0].get_text(strip=True).replace('%', ''))
                    pred['draw'] = float(probs[1].get_text(strip=True).replace('%', ''))
                    pred['away'] = float(probs[2].get_text(strip=True).replace('%', ''))
        except Exception as e:
            print(f"Error extracting predictions: {e}")
        return pred
    
    def extract_actual_result(self, url: str) -> Optional[Dict]:
        """Extract actual match result from a URL (for completed matches)."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return self._extract_result_from_soup(soup)
        except Exception as e:
            print(f"Error extracting actual result: {e}")
            return None
