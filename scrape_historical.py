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


def save_to_json(matches: List[Dict], date_str: str, base_url: str = "https://www.forebet.com"):
    """Save detailed match data to JSON.
    
    Matches with results are saved to historical_matches file.
    Matches without results have their full URLs appended to results.txt for later processing.
    """
    os.makedirs('data', exist_ok=True)
    
    # Separate matches with and without results
    matches_with_results = []
    matches_without_results = []
    
    for match in matches:
        if match.get('has_result'):
            matches_with_results.append(match)
        else:
            matches_without_results.append(match)
    
    # Save only matches with results to historical file
    filename = f"data/historical_matches_{date_str}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(matches_with_results, f, indent=2, default=str)
    print(f"  Saved {len(matches_with_results)} matches with results to {filename}")
    
    # Append matches without results to results.txt (with full URL)
    if matches_without_results:
        results_file = "results.txt"
        new_urls = []
        
        # Read existing URLs to avoid duplicates
        existing_urls = set()
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    existing_urls = set(line.strip() for line in f if line.strip())
            except:
                pass
        
        # Add new URLs (full URL with base)
        for match in matches_without_results:
            url = match.get('url', '')
            if url:
                # Construct full URL
                full_url = base_url + url if url.startswith('/') else url
                if full_url not in existing_urls:
                    new_urls.append(full_url)
                    existing_urls.add(full_url)
        
        # Append to results.txt
        if new_urls:
            with open(results_file, 'a') as f:
                for url in new_urls:
                    f.write(f"{url}\n")
            print(f"  Added {len(new_urls)} URLs without results to results.txt")


def combine_individual_files(delete_after=False):
    """Combine all individual date files into historical_matches_combined.json.
    
    This is the recommended approach for managing data:
    1. Individual date files accumulate over time (365+ files per year)
    2. Combined file is faster to load and process
    3. After combining, individual files can be deleted to save space
    
    Args:
        delete_after: If True, delete individual files after successful combine
    """
    import glob
    
    print("=" * 60)
    print("Combining individual date files into combined.json")
    print("=" * 60)
    
    combined_file = 'data/historical_matches_combined.json'
    
    # Find all individual date files
    pattern = 'data/historical_matches_????-??-??.json'
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("No individual date files found")
        return
    
    print(f"Found {len(files)} individual date files")
    
    # Load existing combined file if it exists
    all_matches = []
    if os.path.exists(combined_file):
        try:
            with open(combined_file, 'r') as f:
                all_matches = json.load(f)
            print(f"Loaded {len(all_matches)} existing matches from combined file")
        except:
            pass
    
    # Load matches from individual files
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                matches = json.load(f)
            all_matches.extend(matches)
            print(f"  {filepath}: {len(matches)} matches")
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
    
    print(f"\nTotal matches before dedup: {len(all_matches)}")
    
    # Deduplicate by URL, keeping version with results
    deduped = {}
    for match in all_matches:
        url = match.get('url', '')
        if not url:
            continue
        
        existing = deduped.get(url)
        if not existing:
            deduped[url] = match
        elif match.get('has_result') and not existing.get('has_result'):
            deduped[url] = match
        elif match.get('has_result') and existing.get('has_result'):
            # Both have results - keep one with more data
            if len(str(match)) > len(str(existing)):
                deduped[url] = match
    
    deduped_list = list(deduped.values())
    
    # Count stats
    with_results = sum(1 for m in deduped_list if m.get('has_result'))
    without_results = len(deduped_list) - with_results
    
    print(f"After dedup: {len(deduped_list)} unique matches")
    print(f"  With results: {with_results}")
    print(f"  Without results: {without_results}")
    
    # Save combined file
    with open(combined_file, 'w') as f:
        json.dump(deduped_list, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(deduped_list)} matches to {combined_file}")
    
    # Delete individual files if requested
    if delete_after:
        print(f"\nDeleting {len(files)} individual date files...")
        for filepath in files:
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"  Error deleting {filepath}: {e}")
        print("Done!")
    
    return len(deduped_list)


def incremental_update(scraper, days_back=1, delay=1.0):
    """Incrementally update combined file with recent matches.
    
    This is the recommended daily workflow:
    1. Scrape yesterday's matches
    2. Add to existing combined file (deduplicated)
    3. Matches without results:
       - If past date → skip (postponed/cancelled)
       - If upcoming → add to results.txt
    
    Args:
        scraper: HistoricalForebetScraper instance
        days_back: Number of days to look back (default: 1 = yesterday)
        delay: Delay between requests
    """
    from datetime import date
    
    print("=" * 60)
    print("INCREMENTAL UPDATE")
    print("=" * 60)
    
    combined_file = 'data/historical_matches_combined.json'
    results_file = 'results.txt'
    
    # Calculate date to scrape
    target_date = date.today() - timedelta(days=days_back)
    date_str = target_date.strftime('%Y-%m-%d')
    
    print(f"Scraping matches for: {date_str}")
    
    # Scrape the date
    matches = scraper.scrape_historical_matches(date_str)
    
    if not matches:
        print("No matches found")
        return
    
    with_results = sum(1 for m in matches if m.get('has_result'))
    without_results = len(matches) - with_results
    print(f"Found {len(matches)} matches ({with_results} with results, {without_results} without)")
    
    # Separate matches with and without results
    matches_with_results = [m for m in matches if m.get('has_result')]
    matches_without_results = [m for m in matches if not m.get('has_result')]
    
    # Handle matches without results
    now = datetime.now()
    upcoming_urls = []
    skipped_past = 0
    
    for match in matches_without_results:
        # Get match date if available
        match_date = match.get('match_date')
        if match_date:
            # Parse date if string
            if isinstance(match_date, str):
                try:
                    match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                except:
                    match_date = None
        
        # Check if match is in the past
        if match_date and isinstance(match_date, datetime):
            hours_since = (now - match_date).total_seconds() / 3600
            if hours_since > 24:
                # Past match with no result - skip (postponed/cancelled)
                skipped_past += 1
                continue
        
        # Upcoming or recent match - add to results.txt
        url = match.get('url', '')
        if url:
            full_url = scraper.BASE_URL + url if url.startswith('/') else url
            upcoming_urls.append(full_url)
    
    if skipped_past > 0:
        print(f"  Skipped {skipped_past} past matches with no result (postponed/cancelled)")
    
    if upcoming_urls:
        # Read existing URLs
        existing_urls = set()
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                existing_urls = set(line.strip() for line in f if line.strip())
        
        # Add new URLs
        new_count = 0
        for url in upcoming_urls:
            if url not in existing_urls:
                existing_urls.add(url)
                new_count += 1
        
        # Write back
        with open(results_file, 'w') as f:
            for url in existing_urls:
                f.write(f"{url}\n")
        
        print(f"  Added {new_count} upcoming matches to results.txt")
    
    # Load existing combined file
    all_matches = []
    if os.path.exists(combined_file):
        try:
            with open(combined_file, 'r') as f:
                all_matches = json.load(f)
            print(f"Loaded {len(all_matches)} existing matches from combined file")
        except:
            print("Could not load existing combined file, starting fresh")
    
    # Add only matches with results
    all_matches.extend(matches_with_results)
    
    # Deduplicate by URL
    deduped = {}
    for match in all_matches:
        url = match.get('url', '')
        if not url:
            continue
        
        existing = deduped.get(url)
        if not existing:
            deduped[url] = match
        elif match.get('has_result') and not existing.get('has_result'):
            deduped[url] = match
        elif match.get('has_result') and existing.get('has_result'):
            if len(str(match)) > len(str(existing)):
                deduped[url] = match
    
    deduped_list = list(deduped.values())
    
    # Stats
    with_results = sum(1 for m in deduped_list if m.get('has_result'))
    
    print(f"\nAfter merge: {len(deduped_list)} unique matches")
    print(f"  With results: {with_results}")
    
    # Save combined file
    with open(combined_file, 'w') as f:
        json.dump(deduped_list, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {combined_file}")
    
    # Also save individual date file for backup
    save_to_json(matches_with_results, date_str, scraper.BASE_URL)
    
    return len(deduped_list)


def main():
    parser = argparse.ArgumentParser(description='Scrape historical football data from Forebet')
    parser.add_argument('--start', default='2026-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2026-02-11', help='End date (YYYY-MM-DD)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (seconds)')
    parser.add_argument('--update-results', action='store_true', help='Update results for matches without them')
    parser.add_argument('--process-queue', action='store_true', help='Process URLs from results.txt')
    parser.add_argument('--max-urls', type=int, default=None, help='Max URLs to process from results.txt')
    parser.add_argument('--combine', action='store_true', help='Combine all individual date files into combined.json')
    parser.add_argument('--incremental', action='store_true', help='Incremental update: scrape yesterday and add to combined file')
    parser.add_argument('--days-back', type=int, default=1, help='Days back for incremental update (default: 1 = yesterday)')
    
    args = parser.parse_args()
    
    scraper = HistoricalForebetScraper()
    
    # Handle --combine mode (merge all individual files)
    if args.combine:
        combine_individual_files()
        return
    
    # Handle --incremental mode (daily update)
    if args.incremental:
        incremental_update(scraper, args.days_back, args.delay)
        return
    
    # Handle --process-queue mode (process results.txt)
    if args.process_queue:
        process_results_txt(scraper, args.delay, args.max_urls)
        return
    
    # Handle --update-results mode
    if args.update_results:
        update_missing_results(scraper, args.delay)
        return
    
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
        
        save_to_json(matches, date_str, scraper.BASE_URL)
        
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


def process_results_txt(scraper, delay=1.0, max_urls=None):
    """Process URLs from results.txt - fetch results and league info.
    
    This function:
    1. Reads URLs from results.txt
    2. Fetches each match page to get results and league info
    3. Saves matches with results to historical_matches_combined.json
    4. Updates league mapping database
    5. Removes processed URLs from results.txt
    """
    print("=" * 60)
    print("Processing URLs from results.txt")
    print("=" * 60)
    
    results_file = "results.txt"
    combined_file = 'data/historical_matches_combined.json'
    
    # Read URLs from results.txt
    if not os.path.exists(results_file):
        print("No results.txt file found")
        return
    
    with open(results_file, 'r') as f:
        all_queued_urls = set(line.strip() for line in f if line.strip())
    
    if not all_queued_urls:
        print("No URLs in results.txt")
        return
    
    urls = list(all_queued_urls)
    print(f"Found {len(urls)} URLs to process")
    
    if max_urls:
        urls = urls[:max_urls]
        print(f"Processing first {max_urls} URLs")
    
    # Load existing matches from combined file
    all_matches = []
    if os.path.exists(combined_file):
        try:
            with open(combined_file, 'r') as f:
                all_matches = json.load(f)
            print(f"Loaded {len(all_matches)} existing matches from combined file")
        except:
            pass
    
    # Create URL lookup for existing matches
    existing_by_url = {}
    for match in all_matches:
        url = match.get('url', '')
        if url:
            existing_by_url[url] = match
    
    # Process each URL
    updated = 0
    new_matches = 0
    leagues_added = 0
    
    for i, full_url in enumerate(urls):
        try:
            print(f"\n[{i+1}/{len(urls)}] Processing: {full_url}")
            
            response = scraper.session.get(full_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract match data from the page
            match_data = {'url': full_url.replace(scraper.BASE_URL, '')}
            
            # Get team names
            team_link = soup.find('a', class_='tnmscn')
            if team_link:
                home_team = team_link.find('span', class_='homeTeam')
                away_team = team_link.find('span', class_='awayTeam')
                
                if home_team:
                    home_name = home_team.find('span', itemprop='name')
                    raw_home = home_name.get_text(strip=True) if home_name else ''
                    match_data['home_team'] = scraper.normalize_team_name(raw_home)
                
                if away_team:
                    away_name = away_team.find('span', itemprop='name')
                    raw_away = away_name.get_text(strip=True) if away_name else ''
                    match_data['away_team'] = scraper.normalize_team_name(raw_away)
            
            # Get league info from breadcrumb or page
            breadcrumb = soup.find('div', class_='breadcrumb')
            if breadcrumb:
                links = breadcrumb.find_all('a')
                if len(links) >= 2:
                    match_data['country'] = links[-2].get_text(strip=True)
                    match_data['league'] = links[-1].get_text(strip=True)
            
            # Get short code from shortTag
            short_tag = soup.find('span', class_='shortTag')
            if short_tag:
                match_data['league_code'] = short_tag.get_text(strip=True)
            
            # Get match date/time from page
            date_elem = soup.find('span', class_='date_bah')
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                match_data['date_text'] = date_text
                # Try to parse the date
                try:
                    # Format: "13/02/2026 15:00" or "13/02 15:00" or "13 Feb 15:00"
                    # Try DD/MM/YYYY HH:MM format first
                    date_match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})', date_text)
                    if date_match:
                        day, month, year, hour, minute = date_match.groups()
                        match_data['match_date'] = datetime(int(year), int(month), int(day), int(hour), int(minute))
                    else:
                        # Try DD/MM HH:MM format
                        date_match = re.match(r'(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})', date_text)
                        if date_match:
                            day, month, hour, minute = date_match.groups()
                            current_year = datetime.now().year
                            match_data['match_date'] = datetime(current_year, int(month), int(day), int(hour), int(minute))
                        else:
                            # Try DD/MM/YYYY format (no time)
                            date_match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_text)
                            if date_match:
                                day, month, year = date_match.groups()
                                match_data['match_date'] = datetime(int(year), int(month), int(day))
                            else:
                                # Try DD MMM HH:MM format
                                date_match = re.match(r'(\d{1,2})\s+(\w{3})\s+(\d{1,2}):(\d{2})', date_text)
                                if date_match:
                                    day, month_str, hour, minute = date_match.groups()
                                    months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                                             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
                                    month = months.get(month_str.lower(), 1)
                                    current_year = datetime.now().year
                                    match_data['match_date'] = datetime(current_year, month, int(day), int(hour), int(minute))
                except Exception as e:
                    pass
            
            # Get match ID from URL
            url_match_id = re.search(r'-(\d{5,7})$', full_url)
            if url_match_id:
                match_data['match_id'] = url_match_id.group(1)
            
            # Extract 5-digit code for league mapping
            url_5digit_match = re.search(r'-(\d{5})\d{1,4}$', full_url)
            if url_5digit_match:
                match_data['url_5digit_code'] = url_5digit_match.group(1)
            
            # Get prediction probabilities
            fprc = soup.find('div', class_='fprc')
            if fprc:
                text = fprc.get_text(separator=' ').strip()
                numbers = re.findall(r'\d+(?:\.\d+)?', text)
                if len(numbers) >= 3:
                    match_data['prob_home'] = float(numbers[0])
                    match_data['prob_draw'] = float(numbers[1])
                    match_data['prob_away'] = float(numbers[2])
            
            # Get prediction
            forecell = soup.find('div', class_='predict_no') or soup.find('div', class_='predict_y')
            if forecell:
                forepr = forecell.find('span', class_='forepr')
                if forepr:
                    pred = forepr.get_text(strip=True)
                    rmap = {'1': 'home', 'X': 'draw', '2': 'away'}
                    match_data['prediction'] = rmap.get(pred.lower(), pred)
            
            # Get actual result (score)
            l_scr = soup.find('b', class_='l_scr')
            if l_scr:
                score_text = l_scr.get_text(strip=True)
                score_match = re.match(r'(\d+)\s*-\s*(\d+)', score_text)
                if score_match:
                    match_data['home_score'] = int(score_match.group(1))
                    match_data['away_score'] = int(score_match.group(2))
                    
                    if match_data['home_score'] > match_data['away_score']:
                        match_data['actual_result'] = 'home'
                    elif match_data['home_score'] < match_data['away_score']:
                        match_data['actual_result'] = 'away'
                    else:
                        match_data['actual_result'] = 'draw'
                    
                    match_data['has_result'] = True
                    print(f"  ✓ Result: {match_data['home_score']}-{match_data['away_score']}")
            
            # Save league mapping
            if match_data.get('url_5digit_code') and match_data.get('league'):
                scraper._save_league_mapping(
                    match_data['url_5digit_code'],
                    match_data.get('league_code', ''),
                    match_data.get('league', ''),
                    match_data.get('country', '')
                )
                leagues_added += 1
            
            # Add to matches list
            url_key = match_data['url']
            if url_key in existing_by_url:
                # Update existing match
                existing_by_url[url_key].update(match_data)
                if match_data.get('has_result'):
                    updated += 1
            else:
                # New match
                all_matches.append(match_data)
                existing_by_url[url_key] = match_data
                new_matches += 1
                if match_data.get('has_result'):
                    updated += 1
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save updated combined file
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")
    
    # Deduplicate matches by URL before saving
    deduped_matches = {}
    for match in all_matches:
        url_key = match.get('url', '')
        if not url_key:
            continue
        existing = deduped_matches.get(url_key)
        if not existing:
            deduped_matches[url_key] = match
        elif match.get('has_result') and not existing.get('has_result'):
            deduped_matches[url_key] = match
        elif match.get('has_result') and existing.get('has_result'):
            # Both have results - keep the one with more data
            if len(str(match)) > len(str(existing)):
                deduped_matches[url_key] = match
    
    all_matches = list(deduped_matches.values())
    
    # Convert datetime objects to strings for JSON serialization
    for match in all_matches:
        if 'match_date' in match and isinstance(match['match_date'], datetime):
            match['match_date'] = match['match_date'].isoformat()
    
    try:
        with open(combined_file, 'w') as f:
            json.dump(all_matches, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_matches)} matches to {combined_file}")
    except Exception as e:
        print(f"Error saving combined file: {e}")
    
    # Update results.txt - keep URLs that weren't processed AND URLs that still don't have results
    # Remove URLs that now have results OR are past matches with no results (postponed/cancelled)
    now = datetime.now()
    remaining_urls = []
    removed_past_no_result = 0
    
    for url in all_queued_urls:
        url_key = url.replace(scraper.BASE_URL, '')
        match = deduped_matches.get(url_key, {})
        
        if match.get('has_result'):
            # Has result - don't keep in queue
            continue
        
        # Check if match date has passed
        match_date = match.get('match_date')
        if match_date:
            # Parse date if it's a string (from JSON)
            if isinstance(match_date, str):
                try:
                    # Try ISO format first
                    match_date = datetime.fromisoformat(match_date.replace('Z', '+00:00'))
                except:
                    try:
                        # Try common formats
                        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d']:
                            try:
                                match_date = datetime.strptime(match_date, fmt)
                                break
                            except:
                                continue
                    except:
                        pass
            
            # If match was more than 24 hours ago and still no result, skip it (postponed/cancelled)
            if isinstance(match_date, datetime):
                hours_since_match = (now - match_date).total_seconds() / 3600
                if hours_since_match > 24:
                    removed_past_no_result += 1
                    print(f"  Removing past match with no result: {match.get('home_team', '?')} vs {match.get('away_team', '?')} ({match.get('date_text', '')})")
                    continue
        
        # Keep in queue - either upcoming or recent (within 24h)
        remaining_urls.append(url)
    
    with open(results_file, 'w') as f:
        for url in remaining_urls:
            f.write(f"{url}\n")
    
    print(f"\nSummary:")
    print(f"  URLs processed: {len(urls)}")
    print(f"  Results found: {updated}")
    print(f"  New matches added: {new_matches}")
    print(f"  League mappings added: {leagues_added}")
    print(f"  Past matches removed (no result): {removed_past_no_result}")
    print(f"  URLs remaining in results.txt: {len(remaining_urls)}")


def update_missing_results(scraper, delay=1.0):
    """Update results for matches that don't have them."""
    import glob
    
    print("=" * 60)
    print("Updating missing results from individual match pages")
    print("=" * 60)
    
    # Load from combined file first (it has the most complete data)
    combined_file = 'data/historical_matches_combined.json'
    all_matches = []
    
    try:
        with open(combined_file, 'r') as f:
            all_matches = json.load(f)
        print(f"Loaded {len(all_matches)} matches from combined file")
    except Exception as e:
        print(f"Error loading combined file: {e}")
        return
    
    # Find matches without results
    missing_results = [m for m in all_matches if not m.get('has_result')]
    print(f"\nFound {len(missing_results)} matches without results")
    
    updated = 0
    for i, match in enumerate(missing_results):
        url = match.get('url', '')
        if not url:
            continue
        
        # Full URL
        full_url = scraper.BASE_URL + url if url.startswith('/') else url
        
        try:
            print(f"[{i+1}/{len(missing_results)}] Checking {match.get('home_team', '?')} vs {match.get('away_team', '?')}...")
            
            response = scraper.session.get(full_url, timeout=30)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for score
            l_scr = soup.find('b', class_='l_scr')
            if l_scr:
                score_text = l_scr.get_text(strip=True)
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
                    updated += 1
                    print(f"  ✓ Result: {match['home_score']}-{match['away_score']}")
            
            time.sleep(delay)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nUpdated {updated} matches with results")
    
    # Update combined file
    print("\nUpdating combined file...")
    try:
        with open(combined_file, 'w') as f:
            json.dump(all_matches, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_matches)} matches to {combined_file}")
    except Exception as e:
        print(f"Error saving combined file: {e}")
    
    # Remove processed URLs from results.txt
    if updated > 0:
        results_file = "results.txt"
        try:
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    queued_urls = set(line.strip() for line in f if line.strip())
                
                # Remove URLs that now have results
                for match in all_matches:
                    if match.get('has_result') and match.get('url'):
                        url = match['url']
                        full_url = scraper.BASE_URL + url if url.startswith('/') else url
                        queued_urls.discard(full_url)
                
                # Write back remaining URLs
                with open(results_file, 'w') as f:
                    for url in queued_urls:
                        f.write(f"{url}\n")
                print(f"Updated results.txt: {len(queued_urls)} URLs remaining")
        except Exception as e:
            print(f"Error updating results.txt: {e}")


if __name__ == "__main__":
    main()
