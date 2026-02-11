#!/usr/bin/env python3
"""
Test scraper for Forebet historical match data by date.
URL format: https://www.forebet.com/en/football-predictions/predictions-1x2/YYYY-MM-DD

This scraper extracts:
- Match data (teams, date, league)
- Predictions (1X2 probabilities, predicted outcome)
- Actual results (for historical matches)
- Odds
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time


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
    
    def scrape_historical_matches(self, date_str: str) -> List[Dict]:
        """
        Scrape historical matches for a given date.
        
        Args:
            date_str: Date in YYYY-MM-DD format (e.g., "2026-02-01")
        
        Returns:
            List of match dictionaries with available data
        """
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
        
        # Find all match containers (rcnt class with tr_0 or tr_1)
        match_containers = soup.find_all('div', class_=lambda x: x and 'rcnt tr_' in x if x else False)
        
        print(f"Found {len(match_containers)} match containers")
        
        for container in match_containers:
            match_data = self._parse_match_container(container, date_str)
            if match_data:
                matches.append(match_data)
        
        return matches
    
    def _parse_match_container(self, container, default_date: str) -> Optional[Dict]:
        """Parse a single match container."""
        match = {}
        
        # Extract match URL and teams
        team_link = container.find('a', class_='tnmscn')
        if team_link:
            match['url'] = team_link.get('href', '')
            
            home_team = team_link.find('span', class_='homeTeam')
            away_team = team_link.find('span', class_='awayTeam')
            
            if home_team:
                home_name = home_team.find('span', itemprop='name')
                match['home_team'] = home_name.get_text(strip=True) if home_name else ''
            
            if away_team:
                away_name = away_team.find('span', itemprop='name')
                match['away_team'] = away_name.get_text(strip=True) if away_name else ''
            
            # Extract date/time
            time_elem = team_link.find('time', itemprop='startDate')
            if time_elem:
                datetime_str = time_elem.get('datetime', '')
                match['datetime'] = datetime_str
                # Parse to standard format
                if datetime_str:
                    try:
                        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                        match['date'] = dt.strftime('%Y-%m-%d')
                        match['time'] = dt.strftime('%H:%M')
                    except:
                        match['date'] = default_date
            else:
                # Try to get from date_bah span
                date_bah = container.find('span', class_='date_bah')
                if date_bah:
                    date_text = date_bah.get_text(strip=True)
                    match['date_str'] = date_text  # Keep original format
        
        # Extract league from shortTag
        short_tag = container.find('span', class_='shortTag')
        if short_tag:
            match['league_code'] = short_tag.get_text(strip=True)
        
        # Extract league info from onclick on the img element
        img_with_onclick = container.find('img', onclick=True)
        if img_with_onclick:
            onclick = img_with_onclick.get('onclick', '')
            # Format: getstag(this,match_id,'Country','League','url_path','country_code')
            stag_match = re.search(r"getstag\(this,(\d+),'([^']+)','([^']+)','([^']+)','([^']+)'", onclick)
            if stag_match:
                match['match_id'] = stag_match.group(1)
                match['country'] = stag_match.group(2)
                match['league'] = stag_match.group(3)
                match['league_url_path'] = stag_match.group(4)
                match['country_code'] = stag_match.group(5)
        
        # Also try to get from fav_icon onclick as backup
        fav_icon = container.find('div', class_='nofav fav_icon')
        if fav_icon:
            fav_id = fav_icon.get('id')
            if fav_id and not match.get('match_id'):
                match['match_id'] = fav_id
        
        # Extract prediction probabilities (fprc contains 3 values: 1, X, 2)
        fprc = container.find('div', class_='fprc')
        if fprc:
            # Get all text content and parse numbers
            text = fprc.get_text(separator=' ').strip()
            # Find all numbers in the text
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            if len(numbers) >= 3:
                try:
                    match['prob_home'] = float(numbers[0])
                    match['prob_draw'] = float(numbers[1])
                    match['prob_away'] = float(numbers[2])
                except ValueError:
                    pass
        
        # Extract predicted outcome (forepr contains 1, X, or 2)
        forecell = container.find('div', class_='predict_no') or container.find('div', class_='predict_y')
        if forecell:
            forepr = forecell.find('span', class_='forepr')
            if forepr:
                pred = forepr.get_text(strip=True)
                rmap = {'1': 'home', 'X': 'draw', '2': 'away'}
                match['prediction'] = rmap.get(pred.lower(), pred)
        
        # Extract actual score (l_scr contains actual score like "0 - 2")
        l_scr = container.find('b', class_='l_scr')
        if l_scr:
            score_text = l_scr.get_text(strip=True)
            # Check if it's a valid score (contains numbers)
            if re.match(r'\d+\s*-\s*\d+', score_text):
                score_match = re.match(r'(\d+)\s*-\s*(\d+)', score_text)
                if score_match:
                    match['home_score'] = int(score_match.group(1))
                    match['away_score'] = int(score_match.group(2))
                    
                    # Determine actual result
                    if match['home_score'] > match['away_score']:
                        match['actual_result'] = 'home'
                    elif match['home_score'] < match['away_score']:
                        match['actual_result'] = 'away'
                    else:
                        match['actual_result'] = 'draw'
                    
                    match['has_result'] = True
        
        # Extract predicted average score (avg_sc)
        avg_sc = container.find('div', class_='avg_sc')
        if avg_sc:
            try:
                match['predicted_avg_goals'] = float(avg_sc.get_text(strip=True))
            except ValueError:
                pass
        
        # Extract odds (lscrsp)
        odds_elem = container.find('span', class_='lscrsp')
        if odds_elem:
            try:
                match['odds'] = float(odds_elem.get_text(strip=True))
            except ValueError:
                pass
        
        return match if match.get('home_team') and match.get('away_team') else None
    
    def scrape_match_details(self, match_url: str) -> Optional[Dict]:
        """Scrape detailed information for a specific match."""
        full_url = f"{self.BASE_URL}{match_url}"
        print(f"Fetching match details: {full_url}")
        
        try:
            response = self.session.get(full_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching match: {e}")
            return None
        
        soup = BeautifulSoup(response.text, 'lxml')
        details = {}
        
        # Extract teams and league
        teams = soup.find_all('h1', class_='teams')
        if teams:
            details['teams_text'] = teams[0].get_text(strip=True)
        
        return details


def test_scraper():
    """Test the historical scraper with a specific date."""
    scraper = HistoricalForebetScraper()
    
    # Test with the provided URL (2026-02-01)
    date_str = "2026-02-01"
    
    print("=" * 60)
    print(f"Testing historical scraper for date: {date_str}")
    print("=" * 60)
    
    matches = scraper.scrape_historical_matches(date_str)
    
    print(f"\nExtracted {len(matches)} matches")
    print()
    
    # Show sample matches
    for i, match in enumerate(matches[:10]):
        print(f"Match {i+1}:")
        print(f"  Teams: {match.get('home_team', '?')} vs {match.get('away_team', '?')}")
        print(f"  Date: {match.get('date', '?')} {match.get('time', '?')}")
        print(f"  League: {match.get('league_code', '?')} - {match.get('league', '?')}")
        print(f"  Probabilities: {match.get('prob_home', '?')}% - {match.get('prob_draw', '?')}% - {match.get('prob_away', '?')}%")
        print(f"  Prediction: {match.get('prediction', '?')}")
        
        if match.get('has_result'):
            print(f"  RESULT: {match.get('home_score', '?')} - {match.get('away_score', '?')} ({match.get('actual_result', '?')})")
            # Check if prediction was correct
            if match.get('prediction') == match.get('actual_result'):
                print(f"  ✓ Prediction CORRECT")
            else:
                print(f"  ✗ Prediction INCORRECT")
        else:
            print(f"  Result: Not yet played")
        
        if match.get('predicted_avg_goals'):
            print(f"  Predicted Avg Goals: {match.get('predicted_avg_goals', '?')}")
        if match.get('odds'):
            print(f"  Odds: {match.get('odds', '?')}")
        
        print()
    
    # Summary statistics
    total_matches = len(matches)
    matches_with_results = sum(1 for m in matches if m.get('has_result'))
    correct_predictions = sum(1 for m in matches if m.get('has_result') and m.get('prediction') == m.get('actual_result'))
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total matches: {total_matches}")
    print(f"Matches with results: {matches_with_results}")
    print(f"Correct predictions: {correct_predictions}")
    if matches_with_results > 0:
        print(f"Accuracy: {correct_predictions / matches_with_results * 100:.1f}%")
    
    # Save to JSON for further analysis
    output_file = f"historical_matches_{date_str}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2, default=str)
    print(f"\nSaved data to {output_file}")


if __name__ == "__main__":
    test_scraper()
