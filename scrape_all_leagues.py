#!/usr/bin/env python3
"""
Create a comprehensive league database by scraping Forebet's league pages.
This extracts all leagues with their short codes from match pages.
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import re
from datetime import datetime
import time

def scrape_league_page(country_url, country_name):
    """Scrape a country's league page to get all leagues and their short codes."""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    leagues = []
    
    try:
        response = requests.get(country_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all match containers to extract league codes
        match_containers = soup.find_all('div', class_=lambda x: x and 'rcnt tr_' in x if x else False)
        
        for container in match_containers:
            # Get short code from shortTag
            short_tag = container.find('span', class_='shortTag')
            if short_tag:
                short_code = short_tag.get_text(strip=True)
            else:
                short_code = None
            
            # Get league name from onclick or URL
            img_with_onclick = container.find('img', onclick=True)
            league_name = None
            league_url_path = None
            
            if img_with_onclick:
                onclick = img_with_onclick.get('onclick', '')
                # Pattern: getstag(this,match_id,'Country','League','url_path','country_code')
                stag_match = re.search(r"getstag\(this,\d+,'([^']+)','([^']+)','([^']+)','([^']+)'", onclick)
                if stag_match:
                    league_name = stag_match.group(2)
                    league_url_path = stag_match.group(3)
            
            # Get match link
            team_link = container.find('a', class_='tnmscn')
            if team_link:
                href = team_link.get('href', '')
                # Extract league from URL if not found
                if not league_name and href:
                    # URL format: /en/football/matches/{team1}-{team2}-{id}
                    # Or: /en/football-predictions-for-{country}/{league}
                    pass
            
            if short_code or league_name:
                leagues.append({
                    'short_code': short_code,
                    'league': league_name,
                    'url_path': league_url_path,
                    'country': country_name
                })
        
        # Also look for league links on the page
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            
            # Pattern: /en/football-predictions-for-{country}/{league}
            if 'predictions-for' in href and country_name.lower().replace(' ', '-') in href.lower():
                parts = href.split('/')
                if len(parts) >= 2:
                    league_slug = parts[-1]
                    league_name = league_slug.replace('-', ' ').title()
                    
                    # Check if we already have this league
                    if not any(l.get('league') == league_name for l in leagues):
                        leagues.append({
                            'short_code': None,
                            'league': league_name,
                            'url_path': href,
                            'country': country_name
                        })
    
    except Exception as e:
        print(f"    Error scraping {country_url}: {e}")
    
    return leagues


def scrape_main_page_for_countries():
    """Scrape the main page to get all country links."""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    url = "https://www.forebet.com/en/football-predictions"
    print(f"Fetching {url}...")
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Error: {e}")
        return {}
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    countries = {}
    
    # Find all links with country patterns
    for link in soup.find_all('a', href=True):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        
        # Pattern: /en/football-predictions-for-{country}
        if 'predictions-for' in href:
            # Extract country from URL
            match = re.search(r'predictions-for-([^/]+)', href)
            if match:
                country_slug = match.group(1)
                country_name = country_slug.replace('-', ' ').title()
                
                # Skip if it's a league URL (has additional path)
                if '/' in href.split('predictions-for-')[-1]:
                    continue
                
                if country_name not in countries:
                    countries[country_name] = {
                        'slug': country_slug,
                        'url': f"https://www.forebet.com{href}" if href.startswith('/') else href
                    }
    
    return countries


def build_comprehensive_db():
    """Build a comprehensive league database."""
    
    print("=" * 60)
    print("BUILDING COMPREHENSIVE LEAGUE DATABASE")
    print("=" * 60)
    
    # Get all countries
    print("\n1. Fetching country list from Forebet...")
    countries = scrape_main_page_for_countries()
    print(f"   Found {len(countries)} countries")
    
    # Scrape each country for leagues
    print("\n2. Scraping leagues from each country...")
    all_leagues = {}
    short_code_lookup = {}
    
    for i, (country_name, country_info) in enumerate(countries.items()):
        print(f"   [{i+1}/{len(countries)}] {country_name}...", end=" ")
        
        leagues = scrape_league_page(country_info['url'], country_name)
        
        if leagues:
            all_leagues[country_name] = {
                'url': country_info['url'],
                'slug': country_info['slug'],
                'leagues': {}
            }
            
            for league in leagues:
                league_name = league.get('league') or league.get('url_path', '').split('/')[-1].replace('-', ' ').title()
                short_code = league.get('short_code')
                
                if league_name:
                    all_leagues[country_name]['leagues'][league_name] = {
                        'short_code': short_code,
                        'url_path': league.get('url_path', ''),
                        'match_count': 0
                    }
                    
                    # Add to short_code lookup
                    if short_code:
                        if short_code not in short_code_lookup:
                            short_code_lookup[short_code] = []
                        short_code_lookup[short_code].append({
                            'country': country_name,
                            'league': league_name
                        })
            
            print(f"({len(leagues)} leagues)")
        else:
            print("(no leagues found)")
        
        # Be nice to the server
        time.sleep(0.5)
    
    # Merge with existing historical data
    print("\n3. Merging with existing historical data...")
    data_dir = 'data'
    for filename in os.listdir(data_dir):
        if filename.startswith('historical_matches_') and filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    matches = json.load(f)
                
                for match in matches:
                    country = match.get('country', '')
                    league = match.get('league', '')
                    short_code = match.get('league_code', '')
                    
                    if country and league:
                        if country not in all_leagues:
                            all_leagues[country] = {
                                'url': '',
                                'slug': country.lower().replace(' ', '-'),
                                'leagues': {}
                            }
                        
                        if league not in all_leagues[country]['leagues']:
                            all_leagues[country]['leagues'][league] = {
                                'short_code': short_code,
                                'url_path': match.get('league_url_path', ''),
                                'match_count': 0
                            }
                        
                        all_leagues[country]['leagues'][league]['match_count'] += 1
                        
                        if short_code and short_code not in short_code_lookup:
                            short_code_lookup[short_code] = [{
                                'country': country,
                                'league': league
                            }]
            
            except Exception as e:
                pass
    
    # Save the database
    print("\n4. Saving database...")
    os.makedirs('data', exist_ok=True)
    
    output = {
        'countries': all_leagues,
        'short_code_lookup': short_code_lookup,
        'last_updated': datetime.now().isoformat(),
        'total_countries': len(all_leagues),
        'total_leagues': sum(len(c['leagues']) for c in all_leagues.values()),
        'total_short_codes': len(short_code_lookup)
    }
    
    with open('data/comprehensive_leagues_db.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved to data/comprehensive_leagues_db.json")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total countries: {output['total_countries']}")
    print(f"Total leagues: {output['total_leagues']}")
    print(f"Total short codes: {output['total_short_codes']}")
    
    # Show sample
    print("\nSample entries:")
    for i, (country, data) in enumerate(list(all_leagues.items())[:5]):
        print(f"\n{country}:")
        for league, info in list(data.get('leagues', {}).items())[:3]:
            sc = info.get('short_code', 'N/A')
            mc = info.get('match_count', 0)
            print(f"  - {league}: {sc} ({mc} matches)")
    
    return output


if __name__ == "__main__":
    build_comprehensive_db()
