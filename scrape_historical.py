#!/usr/bin/env python3
"""
Historical data scraper for Forebet.
Scrapes match data and saves to JSON files for model training.
Also saves league info to database.

Usage:
    python scrape_historical.py --start 2026-01-01 --end 2026-02-11
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import os
import unicodedata


class HistoricalForebetScraper:
    """Scraper for extracting historical football match data from Forebet by date."""
    
    BASE_URL = "https://www.forebet.com"
    
    def __init__(self):
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Database for leagues
        self.leagues_db = {}
        self._load_leagues_db()
    
    @staticmethod
    def normalize_team_name(name: str) -> str:
        """Normalize team name by removing accents and special Unicode characters."""
        if not name:
            return name
        # Normalize Unicode characters (NFD decomposition) and remove accents
        normalized = unicodedata.normalize('NFD', name)
        # Remove combining diacritical marks (accents)
        ascii_name = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        # Replace & with 'and' for consistency
        ascii_name = ascii_name.replace('&', 'and')
        # Clean up any remaining special characters
        ascii_name = re.sub(r'[^\w\s-]', '', ascii_name)
        return ascii_name.strip()
    
    def _load_leagues_db(self):
        """Load existing leagues from database."""
        db_path = 'data/leagues_db.json'
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r') as f:
                    self.leagues_db = json.load(f)
                print(f"Loaded {len(self.leagues_db)} leagues from database")
            except:
                self.leagues_db = {}
    
    def _save_league_mapping(self, code_5digit: str, short_code: str, league_name: str, country: str):
        """Save league mapping to data/league_mapping.json."""
        try:
            os.makedirs('data', exist_ok=True)
            mapping = {}
            try:
                with open('data/league_mapping.json', 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
            except:
                pass
            
            # Only save if not exists
            if code_5digit not in mapping:
                mapping[code_5digit] = {
                    'short_code': short_code,
                    'league_name': league_name,
                    'country': country,
                    'source': 'historical_scraper'
                }
                with open('data/league_mapping.json', 'w', encoding='utf-8') as f:
                    json.dump(mapping, f, indent=2)
        except:
            pass
    
    def scrape_historical_matches(self, date_str: str) -> List[Dict]:
        """Scrape historical matches for a given date."""
        url = f"{self.BASE_URL}/en/football-predictions/predictions-1x2/{date_str}"
        print(f"Fetching: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching URL: {e}")
            return []
        
        soup = BeautifulSoup(response.text, 'lxml')
        matches = []
        
        match_containers = soup.find_all('div', class_=lambda x: x and 'rcnt tr_' in x if x else False)
        
        print(f"  Found {len(match_containers)} match containers")
        
        for container in match_containers:
            match_data = self._parse_match_container(container, date_str)
            if match_data:
                matches.append(match_data)
        
        return matches
    
    def _parse_match_container(self, container, default_date: str) -> Optional[Dict]:
        """Parse a single match container."""
        match = {}
        
        team_link = container.find('a', class_='tnmscn')
        if team_link:
            match['url'] = team_link.get('href', '')
            
            home_team = team_link.find('span', class_='homeTeam')
            away_team = team_link.find('span', class_='awayTeam')
            
            if home_team:
                home_name = home_team.find('span', itemprop='name')
                raw_home = home_name.get_text(strip=True) if home_name else ''
                match['home_team'] = self.normalize_team_name(raw_home)
            
            if away_team:
                away_name = away_team.find('span', itemprop='name')
                raw_away = away_name.get_text(strip=True) if away_name else ''
                match['away_team'] = self.normalize_team_name(raw_away)
        
        short_tag = container.find('span', class_='shortTag')
        if short_tag:
            short_code = short_tag.get_text(strip=True)
            match['league_code'] = short_code
        
        img_with_onclick = container.find('img', onclick=True)
        if img_with_onclick:
            onclick = img_with_onclick.get('onclick', '')
            stag_match = re.search(r"getstag\(this,(\d+),'([^']+)','([^']+)','([^']+)','([^']+)'", onclick)
            if stag_match:
                match['match_id'] = stag_match.group(1)
                # Pattern: getstag(this,match_id,'Country','League','url_path','country_code')
                match['country'] = stag_match.group(2)
                match['league_name'] = stag_match.group(3)  # Actual league name
                match['league_url_path'] = stag_match.group(4)
                match['country_code'] = stag_match.group(5)
        
        # Use shortTag as short_code
        if short_tag:
            match['short_code'] = short_tag.get_text(strip=True)
        
        # Extract 5-digit code from URL and add league info
        if match.get('url'):
            # Extract match_id from URL
            url_match_id = re.search(r'-(\d{5,7})$', match['url'])
            if url_match_id:
                match['match_id'] = url_match_id.group(1)
            
            # Extract 5-digit code for league mapping
            url_5digit_match = re.search(r'-(\d{5})\d{1,4}$', match['url'])
            if url_5digit_match:
                match['url_5digit_code'] = url_5digit_match.group(1)
        
        # Use league_name as the standard 'league' field (to match JSON format)
        if match.get('league_name'):
            match['league'] = match['league_name']
        
        # Fallback: Extract league name from URL path if not available
        if not match.get('league') and match.get('league_url_path'):
            # URL path format: football-tips-and-predictions-for-{country}/{league-name}
            url_path = match['league_url_path']
            # Extract league name from the last part of URL
            league_from_url = url_path.split('/')[-1] if '/' in url_path else url_path
            # Convert dash-separated to title case
            league_from_url = league_from_url.replace('-', ' ').title()
            match['league'] = league_from_url
        
        # Auto-save league mapping to data/league_mapping.json
        if match.get('url_5digit_code') and match.get('league'):
            self._save_league_mapping(
                match['url_5digit_code'],
                match.get('league_code', ''),
                match.get('league', ''),
                match.get('country', '')
            )
        
        fprc = container.find('div', class_='fprc')
        if fprc:
            text = fprc.get_text(separator=' ').strip()
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if len(numbers) >= 3:
                try:
                    match['prob_home'] = float(numbers[0])
                    match['prob_draw'] = float(numbers[1])
                    match['prob_away'] = float(numbers[2])
                except ValueError:
                    pass
        
        forecell = container.find('div', class_='predict_no') or container.find('div', class_='predict_y')
        if forecell:
            forepr = forecell.find('span', class_='forepr')
            if forepr:
                pred = forepr.get_text(strip=True)
                rmap = {'1': 'home', 'X': 'draw', '2': 'away'}
                match['prediction'] = rmap.get(pred.lower(), pred)
        
        l_scr = container.find('b', class_='l_scr')
        if l_scr:
            score_text = l_scr.get_text(strip=True)
            if re.match(r'\d+\s*-\s*\d+', score_text):
                score_match = re.match(r'(\d+)\s*-\s*(\d+)', score_text)
                if score_match:
                    match['home_score'] = int(score_match.group(1))
                    match['away_score'] = int(score_match.group(2))
                    
                    if match['home_score'] > match['away_score']:
                        match['actual_result'] = 'home'
                    elif match['home_score'] < match['away_score']:
                        match['actual_result'] = 'away'
                    else:
                        match['actual_result'] = 'draw'
                    
                    match['has_result'] = True
        
        avg_sc = container.find('div', class_='avg_sc')
        if avg_sc:
            try:
                match['predicted_avg_goals'] = float(avg_sc.get_text(strip=True))
            except ValueError:
                pass
        
        odds_elem = container.find('span', class_='lscrsp')
        if odds_elem:
            try:
                match['odds'] = float(odds_elem.get_text(strip=True))
            except ValueError:
                pass
        
        return match if match.get('home_team') and match.get('away_team') else None


def save_to_json(matches: List[Dict], date_str: str):
    """Save detailed match data to JSON."""
    os.makedirs('data', exist_ok=True)
    filename = f"data/historical_matches_{date_str}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2, default=str)
    print(f"  Saved detailed data to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Scrape historical football data from Forebet')
    parser.add_argument('--start', default='2026-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2026-02-11', help='End date (YYYY-MM-DD)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (seconds)')
    
    args = parser.parse_args()
    
    scraper = HistoricalForebetScraper()
    
    start_date = datetime.strptime(args.start, '%Y-%m-%d')
    end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    print("=" * 60)
    print(f"Scraping historical data from {args.start} to {args.end}")
    print("=" * 60)
    
    all_matches = []
    current_date = start_date
    day_count = (end_date - start_date).days + 1
    
    for i in range(day_count):
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\n[{i+1}/{day_count}] Date: {date_str}")
        
        matches = scraper.scrape_historical_matches(date_str)
        all_matches.extend(matches)
        
        with_results = sum(1 for m in matches if m.get('has_result'))
        print(f"  Total: {len(matches)} | With Results: {with_results}")
        
        save_to_json(matches, date_str)
        
        time.sleep(args.delay)
        current_date += timedelta(days=1)
    
    total = len(all_matches)
    with_results = sum(1 for m in all_matches if m.get('has_result'))
    correct = sum(1 for m in all_matches if m.get('has_result') and m.get('prediction') == m.get('actual_result'))
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total matches scraped: {total}")
    print(f"Matches with results: {with_results}")
    print(f"Correct predictions (Forebet): {correct}")
    if with_results > 0:
        print(f"Forebet Accuracy: {correct / with_results * 100:.1f}%")
    
    print("\nData saved to: data/historical_matches_YYYY-MM-DD.json")
    print("\nUse these JSON files to train the model with:")
    print("  python rebuild_data.py")


if __name__ == "__main__":
    main()
