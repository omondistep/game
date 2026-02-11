#!/usr/bin/env python3
"""
Update the league database by mapping 5-digit codes to short codes.
This allows future match URLs to automatically find their league names.
"""

import json
import re
import os


def extract_5digit_code(url: str) -> str:
    """Extract 5-digit code from match URL."""
    match = re.search(r'-(\d{5})\d{1,4}$', url)
    if match:
        return match.group(1)
    return None


def main():
    # Load existing leagues (short code -> name/country)
    leagues_db = {}
    if os.path.exists('data/leagues_db.json'):
        with open('data/leagues_db.json', 'r', encoding='utf-8') as f:
            leagues_db = json.load(f)
    
    # Load results.txt and build 5-digit code mapping
    five_digit_to_short = {}
    
    if os.path.exists('results.txt'):
        with open('results.txt', 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                url = parts[0]
                short_code = parts[3] if len(parts) > 3 else ''
                league_name = parts[4] if len(parts) > 4 else ''
                country = parts[5] if len(parts) > 5 else ''
                
                five_digit = extract_5digit_code(url)
                if five_digit and short_code:
                    five_digit_to_short[five_digit] = {
                        'short_code': short_code,
                        'league_name': league_name,
                        'country': country
                    }
    
    # Build complete mapping: 5-digit -> full info
    # Also check for divergence and use scraped data as precedence
    
    # Load existing 5-digit mappings from football_scraper.py
    existing_5digit_codes = {
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
    
    # Merge: scraped data takes precedence
    final_mapping = {}
    
    for code, info in five_digit_to_short.items():
        final_mapping[code] = {
            'short_code': info['short_code'],
            'league_name': info['league_name'],
            'country': info['country'],
            'source': 'scraped'
        }
        print(f"Scraped: {code} -> {info['short_code']} ({info['league_name']}, {info['country']})")
    
    for code, name in existing_5digit_codes.items():
        if code not in final_mapping:
            final_mapping[code] = {
                'short_code': '',
                'league_name': name,
                'country': '',
                'source': 'existing'
            }
            print(f"Existing: {code} -> {name}")
    
    # Save comprehensive mapping
    os.makedirs('data', exist_ok=True)
    
    with open('data/league_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(final_mapping, f, indent=2)
    
    print(f"\nTotal mappings: {len(final_mapping)}")
    print("Saved to data/league_mapping.json")
    
    # Also save short code -> full info mapping
    short_code_db = {}
    for code, info in final_mapping.items():
        short = info['short_code']
        if short and short not in short_code_db:
            short_code_db[short] = {
                'league_name': info['league_name'],
                'country': info['country'],
                'five_digit_code': code
            }
    
    with open('data/short_leagues_db.json', 'w', encoding='utf-8') as f:
        json.dump(short_code_db, f, indent=2)
    
    print(f"Short codes: {len(short_code_db)}")
    print("Saved to data/short_leagues_db.json")
    
    # Print sample of the new league mapping
    print("\n" + "=" * 60)
    print("Sample League Mapping:")
    print("=" * 60)
    for i, (code, info) in enumerate(list(final_mapping.items())[:10]):
        print(f"{code}: {info['short_code']} - {info['league_name']} ({info['country']})")


if __name__ == "__main__":
    main()
