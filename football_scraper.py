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
    '23444': 'Italy Serie A',
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
    '24236': 'Bahrain Premier League',
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


def save_unknown_league(league_code: str, url: str) -> Optional[str]:
    """Save a discovered league code and ask for user input to name it.
    
    Returns the league name if provided, None otherwise.
    """
    unknown = load_unknown_leagues()
    
    # Check if already named
    if league_code in LEAGUE_CODES:
        return LEAGUE_CODES[league_code]
    
    # Check if we already have a suggested name
    if league_code in unknown:
        existing = unknown[league_code]
        if existing.get('suggested_name'):
            return existing['suggested_name']
    
    # Ask user for league name
    print(f"\n  {'='*60}")
    print(f"  [?] New League Code Discovered: {league_code}")
    print(f"  {'='*60}")
    print(f"  URL: {url}")
    
    # Try to guess the league name from the URL or teams
    teams_match = re.search(r'/matches/([^/]+)-', url)
    if teams_match:
        teams_str = teams_match.group(1).replace('-', ' ').title()
        print(f"  Teams in URL: {teams_str}")
    
    print(f"\n  Known leagues: {list(LEAGUE_CODES.values())[:5]}...")
    
    try:
        league_name = input(f"  Enter league name (or press Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        league_name = ""
    
    if league_name:
        # Add to LEAGUE_CODES
        LEAGUE_CODES[league_code] = league_name
        
        # Save to unknown leagues file with the name
        unknown[league_code] = {
            'first_seen_url': url,
            'timestamp': datetime.now().isoformat(),
            'suggested_name': league_name
        }
        with open(UNKNOWN_LEAGUES_FILE, 'w') as f:
            json.dump(unknown, f, indent=2)
        
        print(f"  [✓] Added '{league_name}' to known leagues")
        return league_name
    else:
        # Just save as unknown
        if league_code not in unknown:
            unknown[league_code] = {
                'first_seen_url': url,
                'timestamp': datetime.now().isoformat(),
                'suggested_name': None
            }
            with open(UNKNOWN_LEAGUES_FILE, 'w') as f:
                json.dump(unknown, f, indent=2)
            print(f"  [INFO] League code saved for later naming")
        return None


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
            
            # Extract league from URL (5-digit code)
            league, league_code = self._extract_league_from_url(url)
            
            # Track unknown league codes (interactive if needed)
            if league_code and league is None:
                league = save_unknown_league(league_code, url)
            
            match_data = {
                'url': url,
                'timestamp': time.time(),
                'teams': self._extract_teams(soup),
                'match_info': self._extract_match_info(soup, league),
                'standings': self._extract_standings(soup),
                'league_table': self._extract_league_table(soup),
                'form': self._extract_form(soup),
                'last_6_matches': self._extract_last_6_matches(soup),
                'home_matches': self._extract_home_away_matches(soup, 'home'),
                'away_matches': self._extract_home_away_matches(soup, 'away'),
                'head_to_head': self._extract_head_to_head(soup),
                'odds': self._extract_odds(soup),
                'predictions': self._extract_predictions(soup),
                'goals_stats': self._extract_goals_stats(soup),
                'over_under_stats': self._extract_over_under_stats(soup),
                'both_to_score': self._extract_bts_stats(soup),
                'shots_stats': self._extract_shots_stats(soup),
                'passes_stats': self._extract_passes_stats(soup),
                'attacks_stats': self._extract_attacks_stats(soup),
                'discipline': self._extract_discipline(soup),
                'other_stats': self._extract_other_stats(soup),
                'trends': self._extract_trends(soup),
                'match_intro': self._extract_match_intro(soup),
                'injuries': self._extract_injuries(soup),
                'next_matches': self._extract_next_matches(soup),
                'js_detailed_stats': self._extract_javascript_data(soup),
                'actual_result': result,
            }
            return match_data
            
        except Exception as e:
            print(f"Error scraping match: {e}")
            import traceback
            traceback.print_exc()
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
        """Extract actual result from parsed HTML if match has been played."""
        result: Dict = {
            'home_score': None, 'away_score': None,
            'result': None, 'total_goals': None,
            'over_under_2_5': None,
        }
        
        page_text = soup.get_text()
        
        # Check if "FT" (Full Time) indicator is present
        ft_pattern = re.search(r'(\d+)\s*-\s*(\d+)\s*FT', page_text)
        if not ft_pattern:
            # Check for "Final" indicator
            ft_pattern = re.search(r'(\d+)\s*-\s*(\d+)\s*Final', page_text, re.IGNORECASE)
        
        if ft_pattern:
            result['home_score'] = int(ft_pattern.group(1))
            result['away_score'] = int(ft_pattern.group(2))
        else:
            # Fallback: look for score element
            score_elem = soup.find('span', class_='lscrsp')
            if score_elem:
                text = score_elem.get_text(strip=True)
                m = re.search(r'(\d+)\s*-\s*(\d+)', text)
                if m:
                    result['home_score'] = int(m.group(1))
                    result['away_score'] = int(m.group(2))
        
        if result['home_score'] is not None:
            result['total_goals'] = result['home_score'] + result['away_score']
            result['over_under_2_5'] = 'Over' if result['total_goals'] > 2.5 else 'Under'
            if result['home_score'] > result['away_score']:
                result['result'] = '1'
            elif result['home_score'] < result['away_score']:
                result['result'] = '2'
            else:
                result['result'] = 'X'
            return result
        
        return None

    def is_match_played(self, url: str) -> bool:
        """Check if a match has been played."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            page_text = soup.get_text()
            if 'FT' in page_text:
                ft_pattern = re.search(r'(\d+)\s*-\s*(\d+)\s*FT', page_text)
                if ft_pattern:
                    return True
            
            if 'Final' in page_text:
                final_pattern = re.search(r'(\d+)\s*-\s*(\d+)\s*Final', page_text, re.IGNORECASE)
                if final_pattern:
                    return True
            
            score_elem = soup.find('span', class_='lscrsp')
            if score_elem:
                text = score_elem.get_text(strip=True)
                if re.search(r'(\d+)\s*-\s*(\d+)', text):
                    return True
            
            return False
        except Exception as e:
            print(f"Error checking if match is played: {e}")
            return False

    def extract_actual_result(self, url: str) -> Optional[Dict]:
        """Extract actual match result after the game is played."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            result: Dict = {
                'home_score': None, 'away_score': None,
                'result': None, 'total_goals': None,
                'over_under_2_5': None,
            }

            score_elem = soup.find('span', class_='lscrsp')
            if score_elem:
                text = score_elem.get_text(strip=True)
                m = re.search(r'(\d+)\s*-\s*(\d+)', text)
                if m:
                    result['home_score'] = int(m.group(1))
                    result['away_score'] = int(m.group(2))

            if result['home_score'] is None:
                for elem in soup.find_all(['span', 'div'], class_=re.compile(r'score|result')):
                    text = elem.get_text(strip=True)
                    m = re.search(r'(\d+)\s*-\s*(\d+)', text)
                    if m:
                        result['home_score'] = int(m.group(1))
                        result['away_score'] = int(m.group(2))
                        break

            if result['home_score'] is not None:
                result['total_goals'] = result['home_score'] + result['away_score']
                result['over_under_2_5'] = 'Over' if result['total_goals'] > 2.5 else 'Under'
                if result['home_score'] > result['away_score']:
                    result['result'] = '1'
                elif result['home_score'] < result['away_score']:
                    result['result'] = '2'
                else:
                    result['result'] = 'X'
                return result
            
            return None
        except Exception as e:
            print(f"Error extracting actual result: {e}")
            return None

    def _extract_teams(self, soup: BeautifulSoup) -> Dict:
        """Extract team names."""
        teams = {'home': None, 'away': None}
        try:
            # Try to find team_name spans
            team_elems = soup.find_all('span', class_='team_name')
            if len(team_elems) >= 2:
                teams['home'] = team_elems[0].get_text(strip=True)
                teams['away'] = team_elems[1].get_text(strip=True)
                return teams
            
            # Try to find team_name class
            team_elems = soup.find_all(class_='team_name')
            if len(team_elems) >= 2:
                teams['home'] = team_elems[0].get_text(strip=True)
                teams['away'] = team_elems[1].get_text(strip=True)
                return teams
            
            # Fallback: extract from title or mainBox
            title = soup.title.get_text() if soup.title else ""
            if ' vs ' in title:
                parts = title.split(' vs ')
                teams['home'] = parts[0].strip()
                teams['away'] = parts[1].split(' Prediction')[0].strip() if len(parts) > 1 else None
                return teams
            
            main_box = soup.find('div', class_='mainBox')
            if main_box:
                text = main_box.get_text()
                parts = re.split(r'\s+v\s+|\s+vs\.?\s+', text)
                if len(parts) >= 2:
                    teams['home'] = parts[0].strip()
                    teams['away'] = parts[1].strip()
                    return teams
            
            # Last resort: look in page text for "Team1 vs Team2" pattern
            page_text = soup.get_text()
            m = re.search(r'([A-Za-z][A-Za-z\s&]+?)\s+(?:vs\.?|vs\s+)\s+([A-Za-z][A-Za-z\s&]+?)\s+(?:Prediction|Stats)', page_text)
            if m:
                teams['home'] = m.group(1).strip()
                teams['away'] = m.group(2).strip()
                return teams
                
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

    def _extract_home_away_matches(self, soup: BeautifulSoup, team_type: str) -> List[Dict]:
        """Extract home or away matches for a team."""
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

    def _extract_head_to_head(self, soup: BeautifulSoup) -> Dict:
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
        return {'matches': h2h}

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

    def _extract_goals_stats(self, soup: BeautifulSoup) -> Dict:
        """Extract goals statistics."""
        stats = {'home': {}, 'away': {}}
        try:
            gs = soup.find('div', class_='goals')
            if gs:
                home_elem = gs.find('div', class_='home')
                away_elem = gs.find('div', class_='away')
                
                if home_elem:
                    stats['home']['scored'] = home_elem.find('span', class_='for')
                    stats['home']['conceded'] = home_elem.find('span', class_='again')
                    
                if away_elem:
                    stats['away']['scored'] = away_elem.find('span', class_='for')
                    stats['away']['conceded'] = away_elem.find('span', class_='again')
                    
        except Exception as e:
            print(f"Error extracting goals stats: {e}")
        return stats

    def _extract_over_under_stats(self, soup: BeautifulSoup) -> Dict:
        """Extract Over/Under statistics."""
        return {}

    def _extract_bts_stats(self, soup: BeautifulSoup) -> Dict:
        """Extract Both Teams To Score statistics."""
        return {}

    def _extract_shots_stats(self, soup: BeautifulSoup) -> Dict:
        """Extract shots statistics."""
        return {}

    def _extract_passes_stats(self, soup: BeautifulSoup) -> Dict:
        """Extract passes statistics."""
        return {}

    def _extract_attacks_stats(self, soup: BeautifulSoup) -> Dict:
        """Extract attacks statistics."""
        return {}

    def _extract_discipline(self, soup: BeautifulSoup) -> Dict:
        """Extract discipline statistics."""
        return {}

    def _extract_other_stats(self, soup: BeautifulSoup) -> Dict:
        """Extract other statistics."""
        return {}

    def _extract_trends(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract trends."""
        trends = []
        try:
            tc = soup.find('div', class_='trends')
            if tc:
                for trend in tc.find_all('tr')[1:]:
                    cells = trend.find_all('td')
                    if len(cells) >= 2:
                        trends.append({
                            'description': cells[0].get_text(strip=True),
                            'percentage': cells[1].get_text(strip=True),
                            'record': cells[2].get_text(strip=True) if len(cells) > 2 else '',
                        })
        except Exception as e:
            print(f"Error extracting trends: {e}")
        return trends

    def _extract_match_intro(self, soup: BeautifulSoup) -> str:
        """Extract match introduction."""
        try:
            intro = soup.find('div', class_='intro')
            if intro:
                return intro.get_text(strip=True)
        except Exception as e:
            print(f"Error extracting match intro: {e}")
        return ""

    def _extract_injuries(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract injury information."""
        return []

    def _extract_next_matches(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract next matches for teams."""
        return []

    def _extract_javascript_data(self, soup: BeautifulSoup) -> Dict:
        """Extract data from JavaScript variables."""
        stats = {'home_stats': {}, 'away_stats': {}}
        try:
            for script in soup.find_all('script'):
                text = script.get_text()
                # Extract various stats from JavaScript
                patterns = {
                    'shots_total': r'"shots_total"\s*:\s*\[([^\]]+)\]',
                    'shots_on_target': r'"shots_on_target"\s*:\s*\[([^\]]+)\]',
                    'ball_poss': r'"ball_poss"\s*:\s*\[([^\]]+)\]',
                    'passes_accurate': r'"passes_accurate"\s*:\s*\[([^\]]+)\]',
                    'passes_total': r'"passes_total"\s*:\s*\[([^\]]+)\]',
                    'fouls': r'"fouls"\s*:\s*\[([^\]]+)\]',
                    'yellowcards': r'"yellowcards"\s*:\s*\[([^\]]+)\]',
                    'dan_attacks': r'"dan_attacks"\s*:\s*\[([^\]]+)\]',
                }
                
                for key, pattern in patterns.items():
                    match = re.search(pattern, text)
                    if match:
                        values = match.group(1).split(',')
                        stats['home_stats'][key] = [float(v.strip()) for v in values if v.strip()]
                        # Look for away stats
                        away_match = re.search(text.replace(match.group(0), ''), pattern)
        except Exception as e:
            print(f"Error extracting JS data: {e}")
        return stats
