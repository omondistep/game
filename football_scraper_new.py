#!/usr/bin/env python3
"""
Football Match Data Scraper for Forebet - Updated version
Uses full match_id for league lookups and meta description for league extraction.
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from typing import Dict, List, Optional
import time
import glob


class ForebetScraperNew:
    """Scraper with improved league extraction."""

    def __init__(self):
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        }
        self._league_cache = {}
        self._load_leagues_from_historical()

    def _load_leagues_from_historical(self):
        """Load league info from all historical JSON files."""
        historical_files = glob.glob('data/historical_matches_*.json')
        for filepath in historical_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                    for match in matches:
                        match_id = match.get('match_id')
                        if match_id and match_id not in self._league_cache:
                            self._league_cache[match_id] = {
                                'league_code': match.get('short_code', ''),
                                'country': match.get('country', ''),
                                'league': match.get('league_name', ''),
                                'league_url_path': match.get('league_url_path', ''),
                                'country_code': match.get('country_code', ''),
                                'match_id': match_id,
                            }
            except:
                pass
        
        # Also load from leagues_db.json
        try:
            with open('data/leagues_db.json', 'r', encoding='utf-8') as f:
                leagues_db = json.load(f)
                for key, info in leagues_db.items():
                    match_id = info.get('match_id')
                    if match_id and match_id not in self._league_cache:
                        self._league_cache[match_id] = info
        except:
            pass

    def _extract_match_id(self, url: str) -> Optional[str]:
        """Extract full match_id from URL."""
        match = re.search(r'-(\d{5,7})$', url)
        if match:
            return match.group(1)
        return None

    def _get_league_info(self, url: str) -> Dict:
        """Get league info for a URL."""
        match_id = self._extract_match_id(url)
        
        # Check cache first
        if match_id and match_id in self._league_cache:
            return self._league_cache[match_id]
        
        return {'match_id': match_id}

    def _extract_league_from_page(self, soup: BeautifulSoup, url: str) -> Dict:
        """Extract league info from page HTML."""
        result = {
            'league_code': '',
            'country': '',
            'league': '',
            'league_url_path': '',
            'country_code': '',
            'match_id': self._extract_match_id(url),
        }
        
        # Try meta description first (works for most pages)
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            content = meta_desc.get('content')
            # Pattern: "...this match of {Country} {League} on {Date}"
            match = re.search(r'this match of ([A-Za-z]+(?:\s+[A-Za-z]+)*)', content)
            if match:
                league_text = match.group(1).strip()
                words_to_skip = ['football', 'predictions', 'statistics', 'match', 'on']
                words = [w for w in league_text.split() if w.lower() not in words_to_skip]
                if words:
                    result['league'] = ' '.join(words)
                    # Try to extract country
                    country_match = re.search(r'^([A-Za-z]+)', content)
                    if country_match:
                        result['country'] = country_match.group(1)
        
        # Try shortTag
        short_tag = soup.find('span', class_='shortTag')
        if short_tag:
            short_code = short_tag.get_text(strip=True)
            if short_code and len(short_code) >= 2:
                result['league_code'] = short_code
        
        return result

    def _clean_team_name(self, name: str) -> str:
        """Clean up team name."""
        if not name:
            return name
        name = re.sub(r'\s*Prediction,\s*Stats,\s*H2H.*$', '', name)
        name = re.sub(r'\s*Prediction\s*.*$', '', name)
        name = re.sub(r'\s*Stats\s*.*$', '', name)
        name = re.sub(r'\s*H2H\s*.*$', '', name)
        return name.strip()

    def _extract_teams(self, soup: BeautifulSoup) -> Dict:
        """Extract team names from page."""
        teams = {'home': None, 'away': None}
        
        # Try page title
        page_title = soup.find('title')
        if page_title:
            title_text = page_title.get_text()
            vs_match = re.search(r'^(.+?)\s+vs\s+(.+?)\s+-', title_text)
            if vs_match:
                teams['home'] = self._clean_team_name(vs_match.group(1))
                teams['away'] = self._clean_team_name(vs_match.group(2))
        
        return teams

    def scrape_match(self, url: str) -> Optional[Dict]:
        """Scrape match data from Forebet."""
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"Request failed: {e}")
            return None
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        match_data = {'url': url}
        
        # Get league info
        league_info = self._get_league_info(url)
        if league_info.get('league'):
            match_data['league'] = league_info['league']
        else:
            # Extract from page
            page_league = self._extract_league_from_page(soup, url)
            if page_league.get('league'):
                match_data['league'] = page_league['league']
                # Save to DB
                self._save_league_to_db(page_league)
            else:
                match_data['league'] = 'Unknown'
        
        # Get teams
        teams = self._extract_teams(soup)
        match_data['home_team'] = teams.get('home', 'Unknown')
        match_data['away_team'] = teams.get('away', 'Unknown')
        
        return match_data

    def _save_league_to_db(self, league_info: Dict):
        """Save league info to leagues_db.json."""
        match_id = league_info.get('match_id')
        if not match_id:
            return
        
        try:
            leagues_db = {}
            try:
                with open('data/leagues_db.json', 'r', encoding='utf-8') as f:
                    leagues_db = json.load(f)
            except:
                pass
            
            if match_id in leagues_db:
                return
            
            leagues_db[match_id] = {
                'league_code': league_info.get('league_code', ''),
                'country': league_info.get('country', ''),
                'league': league_info.get('league', ''),
                'league_url_path': league_info.get('league_url_path', ''),
                'country_code': league_info.get('country_code', ''),
                'match_id': match_id,
                'match_count': 0
            }
            
            with open('data/leagues_db.json', 'w', encoding='utf-8') as f:
                json.dump(leagues_db, f, indent=2, ensure_ascii=False)
            
            print(f"  Auto-saved league: {match_id} -> {league_info.get('league', 'Unknown')}")
            
        except Exception as e:
            print(f"  Error saving league: {e}")


def main():
    """Test the scraper."""
    import sys
    
    if len(sys.argv) < 2:
        url = "https://www.forebet.com/en/football/matches/hh-export-cd-walter-ferretti-2420379"
    else:
        url = sys.argv[1]
    
    scraper = ForebetScraperNew()
    match_data = scraper.scrape_match(url)
    
    if match_data:
        print(f"\nMatch Data:")
        print(f"  URL: {match_data.get('url')}")
        print(f"  League: {match_data.get('league', 'Unknown')}")
        print(f"  Home Team: {match_data.get('home_team', 'Unknown')}")
        print(f"  Away Team: {match_data.get('away_team', 'Unknown')}")
    else:
        print("Failed to scrape match data")


if __name__ == '__main__':
    main()
