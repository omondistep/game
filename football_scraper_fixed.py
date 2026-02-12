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
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta
import os
import glob


class ForebetScraper:
    """Scraper for extracting football match data from Forebet"""

    # League code mapping (5-digit codes from URL)
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
        '24236': 'Bulgaria First League',
    }

    def __init__(self):
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        }
        self._league_cache = {}  # Cache for league info
        self._load_leagues_from_historical()

    def _load_leagues_from_historical(self):
        """Load league info from all historical JSON files."""
        historical_files = glob.glob('data/historical_matches_*.json')
        for filepath in historical_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                    for match in matches:
                        # Use url_5digit_code as the key
                        url_code = match.get('url_5digit_code')
                        if url_code and url_code not in self._league_cache:
                            self._league_cache[url_code] = {
                                'league_code': match.get('short_code', ''),
                                'country': match.get('country', ''),
                                'league': match.get('league_name', ''),
                                'league_url_path': match.get('league_url_path', ''),
                                'country_code': match.get('country_code', ''),
                                'url_5digit_code': url_code,
                            }
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        # Also load from leagues_db.json
        try:
            with open('data/leagues_db.json', 'r', encoding='utf-8') as f:
                leagues_db = json.load(f)
                for key, info in leagues_db.items():
                    url_code = info.get('url_5digit_code')
                    if url_code and url_code not in self._league_cache:
                        self._league_cache[url_code] = info
        except:
            pass
    
    def _get_league_info_from_cache(self, url_5digit_code: str) -> Optional[Dict]:
        """Get league info from cache using 5-digit code."""
        return self._league_cache.get(url_5digit_code)
    
    def _extract_5digit_code(self, url: str) -> Optional[str]:
        """Extract 5-digit league code from URL.
        
        URL format: /matches/{team1}-{team2}-{league_code}{match_id}
        Example: /matches/cska-sofia-cska-1948-2423671 -> league_code = 24236
        """
        # Match 5-digit code before match ID at end
        match = re.search(r'-(\\d{5})\\d{1,3}$', url)
        if match:
            return match.group(1)
        return None
    
    def _extract_league_code(self, url: str) -> Optional[str]:
        """Extract league code (first 5 digits) from URL.
        
        URL format: /matches/{team1}-{team2}-{league_code}{match_id}
        Example: /matches/cska-sofia-cska-1948-2423671 -> league_code = 24236
        """
        return self._extract_5digit_code(url)
    
    def _get_league_name(self, code: str, prompt_user: bool = False) -> Optional[str]:
        """Get league name from code, prompting user if unknown and prompt_user=True."""
        if not code:
            return None
        
        # Check known leagues
        if code in self.LEAGUE_CODES:
            return self.LEAGUE_CODES[code]
        
        # Check saved unknown leagues
        try:
            with open('data/unknown_leagues.json', 'r') as f:
                unknown = json.load(f)
                if code in unknown:
                    value = unknown[code]
                    # Handle both old format (string) and new format (dict with suggested_name)
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, dict):
                        return value.get('suggested_name') or value.get('name')
        except:
            pass
        
        # If prompt_user is True, ask user for league name
        if prompt_user:
            try:
                league = input(f"Enter league name for code {code}: ").strip()
                if league:
                    # Save to unknown_leagues.json for future use
                    try:
                        with open('data/unknown_leagues.json', 'r') as f:
                            unknown = json.load(f)
                    except:
                        unknown = {}
                    unknown[code] = league
                    with open('data/unknown_leagues.json', 'w') as f:
                        json.dump(unknown, f, indent=2)
                    return league
            except KeyboardInterrupt:
                return None
        
        # Return placeholder with code so training can proceed without user input
        return f"League {code}"

    def update_league_name(self, code: str, new_name: str) -> bool:
        """Update league name for a given code.
        
        Args:
            code: League code (5-digit code from URL)
            new_name: New league name to use
            
        Returns:
            True if update was successful, False otherwise
        """
        if not code or not new_name:
            return False
        
        # Update in unknown_leagues.json
        try:
            with open('data/unknown_leagues.json', 'r') as f:
                unknown = json.load(f)
        except:
            unknown = {}
        
        # Check if code exists in unknown_leagues or is a known league
        if code in self.LEAGUE_CODES or code in unknown:
            unknown[code] = new_name
            with open('data/unknown_leagues.json', 'w') as f:
                json.dump(unknown, f, indent=2)
            return True
        
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _extract_league_from_page(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract league name from match page HTML.
        
        First checks the cache from historical JSON files, then tries to extract from page.
        Saves new leagues to leagues_db.json.
        """
        # First, check if we have cached league info for this URL's code
        url_5digit_code = self._extract_5digit_code(url)
        
        if url_5digit_code:
            # Check cache first (loaded from historical JSON files)
            if url_5digit_code in self._league_cache:
                league_info = self._league_cache[url_5digit_code]
                return league_info.get('league') or league_info.get('suggested_name')
            
            # Check leagues_db.json
            try:
                with open('data/leagues_db.json', 'r', encoding='utf-8') as f:
                    leagues_db = json.load(f)
                if url_5digit_code in leagues_db:
                    return leagues_db[url_5digit_code].get('league')
            except:
                pass
            
            # Check unknown_leagues.json
            try:
                with open('data/unknown_leagues.json', 'r') as f:
                    unknown = json.load(f)
                if url_5digit_code in unknown:
                    value = unknown[url_5digit_code]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, dict):
                        return value.get('suggested_name') or value.get('name')
            except:
                pass
        
        # Extract from page and save to DB
        league_info = self._extract_league_from_page_html(soup, url, url_5digit_code)
        
        if league_info.get('league') and url_5digit_code:
            self._save_league_to_db(url_5digit_code, league_info)
        
        return league_info.get('league') if league_info else None
    
    def _extract_league_from_page_html(self, soup: BeautifulSoup, url: str, url_5digit_code: str = None) -> Dict:
        """Extract league info from page HTML."""
        result = {
            'league_code': '',
            'country': '',
            'league': '',
            'league_url_path': '',
            'country_code': '',
            'url_5digit_code': url_5digit_code or '',
        }
        
        # Method 1: Try to find league info from onclick attributes
        img_with_onclick = soup.find('img', onclick=True)
        if img_with_onclick:
            onclick = img_with_onclick.get('onclick', '')
            # Pattern: getstag(this,match_id,'Country','League','url_path','country_code')
            stag_match = re.search(r"getstag\(this,\d+,'([^']+)','([^']+)','([^']+)','([^']+)'", onclick)
            if stag_match:
                result['country'] = stag_match.group(1)
                result['league'] = stag_match.group(2)
                result['league_url_path'] = stag_match.group(3)
                result['country_code'] = stag_match.group(4)
                if result['league'] and len(result['league']) > 2:
                    return result
        
        # Method 2: Try shortTag span for short code
        short_tag = soup.find('span', class_='shortTag')
        if short_tag:
            short_code = short_tag.get_text(strip=True)
            if short_code and len(short_code) >= 2:
                result['league_code'] = short_code
                # Try to expand short code
                expanded = self._expand_league_code(short_code, url)
                if expanded:
                    result['league'] = expanded
                else:
                    result['league'] = short_code  # Use short code as fallback
                return result
        
        # Method 3: Extract from meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            content = meta_desc.get('content')
            # Pattern: "...this match of {Country} {League}..."
            match = re.search(r'this match of ([A-Za-z]+(?:\\s+[A-Za-z]+)*)', content)
            if match:
                league_text = match.group(1).strip()
                words_to_skip = ['football', 'predictions', 'statistics', 'match']
                words = [w for w in league_text.split() if w.lower() not in words_to_skip]
                if words:
                    result['league'] = ' '.join(words)
                    return result
        
        return result
    
    def _save_league_to_db(self, url_5digit_code: str, league_info: Dict):
        """Save league info to leagues_db.json."""
        if not url_5digit_code:
            return
        
        try:
            leagues_db = {}
            try:
                with open('data/leagues_db.json', 'r', encoding='utf-8') as f:
                    leagues_db = json.load(f)
            except:
                pass
            
            # Check if already exists
            if url_5digit_code in leagues_db:
                return
            
            leagues_db[url_5digit_code] = {
                'league_code': league_info.get('league_code', ''),
                'country': league_info.get('country', ''),
                'league': league_info.get('league', ''),
                'league_url_path': league_info.get('league_url_path', ''),
                'country_code': league_info.get('country_code', ''),
                'url_5digit_code': url_5digit_code,
                'match_count': 0
            }
            
            with open('data/leagues_db.json', 'w', encoding='utf-8') as f:
                json.dump(leagues_db, f, indent=2, ensure_ascii=False)
            
            print(f"  📚 Auto-saved league: {url_5digit_code} -> {league_info.get('league', 'Unknown')}")
            
        except Exception as e:
            print(f"  ⚠ Error saving league to DB: {e}")
    
    def _expand_league_code(self, short_code: str, url: str) -> Optional[str]:
        """Expand abbreviated league code to full name."""
        # Common short codes mapped to full names
        league_mappings = {
            'RoC': 'Romania Cupa',
            'RoL': 'Romania Liga 1',
            'RoL2': 'Romania Liga 2',
            'EPL': 'England Premier League',
            'ELC': 'England Championship',
            'EL1': 'England League One',
            'EL2': 'England League Two',
            'LL': 'Spain La Liga',
            'LL2': 'Spain Segunda Division',
            'SA': 'Italy Serie A',
            'SB': 'Italy Serie B',
            'BL1': 'Germany Bundesliga',
            'BL2': 'Germany Bundesliga 2',
            'FL1': 'France Ligue 1',
            'FL2': 'France Ligue 2',
            'NL': 'Netherlands Eredivisie',
            'PL': 'Portugal Primeira Liga',
            'SC': 'Scotland Premiership',
            'PK': 'Poland Ekstraklasa',
            'HU1': 'Hungary NB I',
            'CZ1': 'Czech Republic First League',
            'GR1': 'Greece Super League',
            'BE1': 'Belgium First Division A',
            'AT1': 'Austria Bundesliga',
            'CH1': 'Switzerland Super League',
            'SE1': 'Sweden Allsvenskan',
            'NO1': 'Norway Eliteserien',
            'DK1': 'Denmark Superliga',
            'FI1': 'Finland Veikkausliiga',
            'IE1': 'Ireland Premier Division',
            'IL1': 'Israel Premier League',
            'TR1': 'Türkiye Süper Lig',
            'UK1': 'Ukraine Premier League',
            'RS1': 'Serbia Super Liga',
            'BG1': 'Bulgaria First League',
            'SK1': 'Slovakia Super Liga',
            'HR1': 'Croatia HNL',
            'SI1': 'Slovenia Prva Liga',
            'LT1': 'Lithuania A Lyga',
            'LV1': 'Latvia Higher League',
            'EE1': 'Estonia Meistriliiga',
            'MD1': 'Moldova National Division',
            'AL1': 'Albania Super Liga',
            'MK1': 'North Macedonia First League',
            'BA1': 'Bosnia Premier League',
            'ME1': 'Montenegro First League',
            'AD1': 'Andorra Primera Divisio',
            'SM1': 'San Marino Championship',
            'GI1': 'Gibraltar Premier Division',
            'JE1': 'Jersey Premier Division',
            'IM1': 'Isle of Man Premier League',
            'XS1': 'Scotland Lowland League',
            'XW1': 'Scotland Highland League',
            'WA1': 'Wales Premier League',
            'NIR': 'Northern Ireland Premiership',
            'IRC': 'Northern Ireland Cup',
        }
        
        # Check direct mappings first
        if short_code in league_mappings:
            return league_mappings[short_code]
        
        # Try to extract country from URL and combine with short code
        country_mappings = {
            'england': 'England',
            'spain': 'Spain',
            'italy': 'Italy',
            'germany': 'Germany',
            'france': 'France',
            'netherlands': 'Netherlands',
            'portugal': 'Portugal',
            'scotland': 'Scotland',
            'turkey': 'Türkiye',
            'greece': 'Greece',
            'belgium': 'Belgium',
            'austria': 'Austria',
            'switzerland': 'Switzerland',
            'ukraine': 'Ukraine',
            'russia': 'Russia',
            'poland': 'Poland',
            'czech': 'Czech Republic',
            'hungary': 'Hungary',
            'romania': 'Romania',
            'sweden': 'Sweden',
            'norway': 'Norway',
            'denmark': 'Denmark',
            'finland': 'Finland',
            'ireland': 'Ireland',
            'israel': 'Israel',
        }
        
        url_lower = url.lower()
        for country_key, country_name in country_mappings.items():
            if country_key in url_lower:
                # Combine country with short code for league name
                return f"{country_name} {short_code}"
        
        return None
    
    def _rate_limit(self):
        """Implement rate limiting to avoid being blocked."""
        current_time = time.time()
        if hasattr(self, '_last_request_time'):
            time_since_last = current_time - self._last_request_time
            if time_since_last < 2.0:
                time.sleep(2.0 - time_since_last)
        self._last_request_time = current_time
    
    def _make_request(self, url: str, max_retries: int = 3) -> Optional[requests.Response]:
        """Make a request to the URL with rate limiting and retries."""
        self._rate_limit()
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return None
    
    def scrape_match(self, url: str, prompt_user: bool = False) -> Optional[Dict]:
        """
        Scrape comprehensive match data from Forebet.
        
        Args:
            url: Full Forebet match URL
            prompt_user: Whether to prompt user for unknown league names
            
        Returns:
            Dictionary containing match data or None if failed
        """
        response = self._make_request(url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        match_data = {'url': url}
        
        # Extract league info
        league_name = self._extract_league_from_page(soup, url)
        if not league_name:
            league_code = self._extract_league_code(url)
            league_name = self._get_league_name(league_code)
        
        match_data['league'] = league_name or 'Unknown'
        
        # Extract teams
        teams = self._extract_teams(soup)
        match_data['home_team'] = teams.get('home')
        match_data['away_team'] = teams.get('away')
        
        # Extract form
        form = self._extract_form(soup)
        match_data['home_form'] = form.get('home', [])
        match_data['away_form'] = form.get('away', [])
        
        # Extract goals stats
        goals_stats = self._extract_goals_stats(soup)
        match_data['goals_stats'] = goals_stats
        
        # Extract head-to-head
        h2h = self._extract_h2h(soup)
        match_data['h2h'] = h2h
        
        # Extract standings
        standings = self._extract_standings(soup)
        match_data['standings'] = standings
        
        # Extract odds
        odds = self._extract_odds(soup)
        match_data['odds'] = odds
        
        # Extract home/away performance
        home_away = self._extract_home_away(soup)
        match_data['home_away'] = home_away
        
        # Extract match time
        match_time = self._extract_match_time(soup)
        match_data['match_time'] = match_time
        
        return match_data
    
    def _extract_match_time(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract match datetime from page."""
        time_elem = soup.find('div', class_='mstat-date')
        if time_elem:
            date_text = time_elem.get_text(strip=True)
            if date_text:
                return date_text
        return None
    
    def _extract_teams(self, soup: BeautifulSoup) -> Dict:
        """Extract team names from match page."""
        teams = {'home': None, 'away': None}
        
        # Method 1: Try script data (JSON in script tags) - most reliable
        script_data = self._extract_teams_from_script(soup)
        if script_data.get('home') and script_data.get('away'):
            return script_data
        
        # Method 2: Try different selectors
        team_selectors = [
            ('span', {'class': 'tname'}),
            ('div', {'class': 'team-name'}),
            ('a', {'class': 'team-link'}),
            ('h2', {'class': 'team-name'}),
            ('div', {'class': 'trname'}),
            ('span', {'class': 'name'}),
        ]
        
        team_elems = []
        for tag, attrs in team_selectors:
            team_elems = soup.find_all(tag, attrs)
            if len(team_elems) >= 2:
                break
        
        if len(team_elems) >= 2:
            teams['home'] = team_elems[0].get_text(strip=True)
            teams['away'] = team_elems[1].get_text(strip=True)
        
        # Method 3: Try to find team names in page title or breadcrumbs
        if not teams['home'] or not teams['away']:
            page_title = soup.find('title')
            if page_title:
                title_text = page_title.get_text()
                # Pattern: "Team1 vs Team2 - Forebet"
                vs_match = re.search(r'^(.+?)\\s+vs\\s+(.+?)\\s+-', title_text)
                if vs_match:
                    teams['home'] = vs_match.group(1).strip()
                    teams['away'] = vs_match.group(2).strip()
        
        # Method 4: Look for teams in heading elements
        if not teams['home'] or not teams['away']:
            headings = soup.find_all(['h1', 'h2', 'h3'])
            for h in headings:
                text = h.get_text(strip=True)
                if ' vs ' in text:
                    parts = text.split(' vs ')
                    if len(parts) >= 2:
                        teams['home'] = parts[0].strip()
                        teams['away'] = parts[1].strip()
                        break
        
        # Method 5: Look for teams in the match URL
        if not teams['home'] or not teams['away']:
            url_teams = self._extract_teams_from_url(soup)
            if url_teams.get('home'):
                teams['home'] = url_teams['home']
            if url_teams.get('away'):
                teams['away'] = url_teams['away']
        
        return teams
    
    def _extract_teams_from_script(self, soup: BeautifulSoup) -> Dict:
        """Extract team names from script tags containing JSON data."""
        teams = {'home': None, 'away': None}
        scripts = soup.find_all('script')
        
        for script in scripts:
            script_text = script.get_text()
            
            # Look for team data in various formats
            patterns = [
                # Pattern: "teamNames":["Home Team","Away Team"]
                r'"teamNames"\s*:\s*\[([^\]]+)\]',
                # Pattern: homeTeam:"Home Team", awayTeam:"Away Team"
                r'"homeTeam"\s*:\s*"([^"]+)"[^}]*"awayTeam"\s*:\s*"([^"]+)"',
                # Pattern: home:"Home Team", away:"Away Team"
                r'"home"\s*:\s*"([^"]+)"[^}]*"away"\s*:\s*"([^"]+)"',
                # Pattern: teams:["Home","Away"]
                r'"teams"\s*:\s*\[([^\]]+)\]',
            ]
            
            for pattern in patterns:
                try:
                    match = re.search(pattern, script_text)
                    if match:
                        if 'teamNames' in pattern or 'teams' in pattern:
                            # Extract array content
                            content = match.group(1)
                            team_list = re.findall(r'"([^"]+)"', content)
                            if len(team_list) >= 2:
                                teams['home'] = team_list[0]
                                teams['away'] = team_list[1]
                                return teams
                        elif 'homeTeam' in pattern:
                            teams['home'] = match.group(1)
                            teams['away'] = match.group(2)
                            return teams
                        elif '"home"' in pattern:
                            teams['home'] = match.group(1)
                            teams['away'] = match.group(2)
                            return teams
                except re.error:
                    continue
        
        return teams
    
    def _extract_teams_from_url(self, soup: BeautifulSoup) -> Dict:
        """Extract team names from URL path if available in page."""
        teams = {'home': None, 'away': None}
        
        # Try to find URL in page
        url_elem = soup.find('link', rel='canonical')
        if url_elem and url_elem.get('href'):
            url = url_elem.get('href')
            # Pattern: /matches/team1-team2-{id}
            match = re.search(r'/matches/([^/]+)-([^/]+)-\\d+$', url)
            if match:
                team1 = match.group(1).replace('-', ' ').title()
                team2 = match.group(2).replace('-', ' ').title()
                if team1 and team2:
                    teams['home'] = team1
                    teams['away'] = team2
        
        return teams
    
    def _extract_form(self, soup: BeautifulSoup) -> Dict:
        """Extract recent form for both teams."""
        form = {'home': [], 'away': []}
        
        # Try to find form data in script tags
        scripts = soup.find_all('script')
        
        for script in scripts:
            script_text = script.get_text()
            
            # Look for form data patterns
            home_form_match = re.search(r'homeTeam.*?form.*?\\[([^\\]]+)\\]', script_text)
            away_form_match = re.search(r'awayTeam.*?form.*?\\[([^\\]]+)\\]', script_text)
            
            if home_form_match:
                form_str = home_form_match.group(1)
                form['home'] = re.findall(r'[WDL]', form_str)
            
            if away_form_match:
                form_str = away_form_match.group(1)
                form['away'] = re.findall(r'[WDL]', form_str)
        
        # Fallback: extract from HTML tables
        if not form['home'] or not form['away']:
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    for cell in cells:
                        cell_text = cell.get_text(strip=True)
                        if re.match(r'^[WDL]{3,5}$', cell_text):
                            if not form['home']:
                                form['home'] = list(cell_text)
                            elif not form['away']:
                                form['away'] = list(cell_text)
                                break
        
        return form
    
    def _extract_goals_stats(self, soup: BeautifulSoup) -> Dict:
        """Extract goals statistics."""
        stats = {
            'home_avg_scored': None,
            'home_avg_conceded': None,
            'away_avg_scored': None,
            'away_avg_conceded': None,
            'home_last6_scored': [],
            'away_last6_scored': [],
        }
        
        scripts = soup.find_all('script')
        
        for script in scripts:
            script_text = script.get_text()
            
            # Look for goals data
            goals_patterns = [
                r'\"home_avg_scored\"\\s*:\\s*([\\d.]+)',
                r'\"away_avg_scored\"\\s*:\\s*([\\d.]+)',
                r'\"home_last6\"\\s*:\\s*\\[([^\\]]+)\\]',
                r'\"away_last6\"\\s*:\\s*\\[([^\\]]+)\\]',
            ]
            
            for pattern in goals_patterns:
                match = re.search(pattern, script_text)
                if match:
                    if 'home_avg_scored' in pattern:
                        stats['home_avg_scored'] = float(match.group(1))
                    elif 'away_avg_scored' in pattern:
                        stats['away_avg_scored'] = float(match.group(1))
                    elif 'home_last6' in pattern:
                        goals = re.findall(r'\\d+', match.group(1))
                        stats['home_last6_scored'] = [int(g) for g in goals]
                    elif 'away_last6' in pattern:
                        goals = re.findall(r'\\d+', match.group(1))
                        stats['away_last6_scored'] = [int(g) for g in goals]
        
        return stats
    
    def _extract_h2h(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract head-to-head data."""
        h2h = []
        
        scripts = soup.find_all('script')
        
        for script in scripts:
            script_text = script.get_text()
            
            # Look for h2h data
            h2h_match = re.search(r'\"h2h\"\\s*:\\s*\\[([^\\]]+)\\]', script_text)
            if h2h_match:
                h2h_json = '[' + h2h_match.group(1) + ']'
                try:
                    h2h = json.loads(h2h_json.replace("'", '"'))
                except:
                    pass
        
        return h2h
    
    def _extract_standings(self, soup: BeautifulSoup) -> Dict:
        """Extract league standings data."""
        standings = {'home': {}, 'away': {}}
        
        scripts = soup.find_all('script')
        
        for script in scripts:
            script_text = script.get_text()
            
            # Look for standings data
            home_standing = re.search(r'\"home_standing\"\\s*:\\s*(\\d+)', script_text)
            away_standing = re.search(r'\"away_standing\"\\s*:\\s*(\\d+)', script_text)
            
            if home_standing:
                standings['home']['position'] = int(home_standing.group(1))
            if away_standing:
                standings['away']['position'] = int(away_standing.group(1))
        
        return standings
    
    def _extract_odds(self, soup: BeautifulSoup) -> Dict:
        """Extract betting odds."""
        odds = {
            'home_odds': None,
            'draw_odds': None,
            'away_odds': None,
            'over_25_odds': None,
            'under_25_odds': None,
        }
        
        scripts = soup.find_all('script')
        
        for script in scripts:
            script_text = script.get_text()
            
            # Look for odds data
            patterns = [
                r'\"home_odds\"\\s*:\\s*([\\d.]+)',
                r'\"draw_odds\"\\s*:\\s*([\\d.]+)',
                r'\"away_odds\"\\s*:\\s*([\\d.]+)',
                r'\"over_25_odds\"\\s*:\\s*([\\d.]+)',
                r'\"under_25_odds\"\\s*:\\s*([\\d.]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, script_text)
                if match:
                    if 'home_odds' in pattern:
                        odds['home_odds'] = float(match.group(1))
                    elif 'draw_odds' in pattern:
                        odds['draw_odds'] = float(match.group(1))
                    elif 'away_odds' in pattern:
                        odds['away_odds'] = float(match.group(1))
                    elif 'over_25_odds' in pattern:
                        odds['over_25_odds'] = float(match.group(1))
                    elif 'under_25_odds' in pattern:
                        odds['under_25_odds'] = float(match.group(1))
        
        return odds
    
    def _extract_home_away(self, soup: BeautifulSoup) -> Dict:
        """Extract home/away performance data."""
        home_away = {
            'home_home_avg_scored': None,
            'home_home_avg_conceded': None,
            'away_away_avg_scored': None,
            'away_away_avg_conceded': None,
        }
        
        scripts = soup.find_all('script')
        
        for script in scripts:
            script_text = script.get_text()
            
            # Look for home/away data
            patterns = [
                r'\"home_home_avg_scored\"\\s*:\\s*([\\d.]+)',
                r'\"home_home_avg_conceded\"\\s*:\\s*([\\d.]+)',
                r'\"away_away_avg_scored\"\\s*:\\s*([\\d.]+)',
                r'\"away_away_avg_conceded\"\\s*:\\s*([\\d.]+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, script_text)
                if match:
                    if 'home_home_avg_scored' in pattern:
                        home_away['home_home_avg_scored'] = float(match.group(1))
                    elif 'home_home_avg_conceded' in pattern:
                        home_away['home_home_avg_conceded'] = float(match.group(1))
                    elif 'away_away_avg_scored' in pattern:
                        home_away['away_away_avg_scored'] = float(match.group(1))
                    elif 'away_away_avg_conceded' in pattern:
                        home_away['away_away_avg_conceded'] = float(match.group(1))
        
        return home_away
    
    def get_team_history(self, team_name: str) -> List[Dict]:
        """
        Get historical match data for a specific team.
        
        Args:
            team_name: Name of the team
            
        Returns:
            List of dictionaries containing match history
        """
        history = []
        
        # Load from team history files if available
        for filepath in glob.glob('data/team_history_*.json'):
            try:
                with open(filepath, 'r') as f:
                    team_data = json.load(f)
                    if team_data.get('team') == team_name:
                        history = team_data.get('matches', [])
                        break
            except:
                pass
        
        return history
    
    def update_team_history(self, team_name: str, match_data: Dict):
        """
        Update team history with new match data.
        
        Args:
            team_name: Name of the team
            match_data: Match data to add
        """
        history = self.get_team_history(team_name)
        history.append(match_data)
        
        # Save updated history
        team_file = f"data/team_history_{hash(team_name) % 1000}.json"
        try:
            with open(team_file, 'w') as f:
                json.dump({
                    'team': team_name,
                    'matches': history
                }, f, indent=2)
        except:
            pass


def main():
    """Main function to test scraping."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python football_scraper.py <match_url>")
        print("Example: python football_scraper.py https://www.forebet.com/en/football/matches/juventus-lazio-2344437")
        sys.exit(1)
    
    url = sys.argv[1]
    
    scraper = ForebetScraper()
    match_data = scraper.scrape_match(url)
    
    if match_data:
        print("\n" + "=" * 60)
        print("MATCH DATA")
        print("=" * 60)
        print(f"URL: {match_data.get('url')}")
        print(f"League: {match_data.get('league', 'Unknown')}")
        print(f"Home Team: {match_data.get('home_team', 'Unknown')}")
        print(f"Away Team: {match_data.get('away_team', 'Unknown')}")
        print(f"Match Time: {match_data.get('match_time', 'TBD')}")
        
        print("\n" + "=" * 60)
        print("FORM")
        print("=" * 60)
        print(f"Home Form: {''.join(match_data.get('home_form', []))}")
        print(f"Away Form: {''.join(match_data.get('away_form', []))}")
        
        print("\n" + "=" * 60)
        print("ODDS")
        print("=" * 60)
        odds = match_data.get('odds', {})
        print(f"Home Win: {odds.get('home_odds', 'N/A')}")
        print(f"Draw: {odds.get('draw_odds', 'N/A')}")
        print(f"Away Win: {odds.get('away_odds', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("RAW MATCH DATA")
        print("=" * 60)
        print(json.dumps(match_data, indent=2, default=str))
    else:
        print("❌ Failed to extract match data")


if __name__ == '__main__':
    main()
