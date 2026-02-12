#!/usr/bin/env python3
"""
Update the league database from historical matches.
This extracts league information from historical JSON files to build a league database.
"""

import json
import glob
import os
import re


def extract_match_info(url: str) -> tuple:
    """Extract match ID from URL and determine league code prefix.
    URL format: /en/football/matches/carabobo-portuguesa-2420044
    """
    parts = url.rstrip('/').split('-')
    if parts:
        match_id = parts[-1]
        # Extract prefix from match_id (first 3 digits)
        prefix = match_id[:3] if len(match_id) >= 3 else match_id
        return match_id, prefix
    return None, None


def main():
    # Load existing leagues database
    leagues_db = {}
    leagues_db_path = 'data/leagues_db.json'
    if os.path.exists(leagues_db_path):
        with open(leagues_db_path, 'r', encoding='utf-8') as f:
            leagues_db = json.load(f)
    
    print(f"Loaded {len(leagues_db)} existing leagues")
    
    # Find all historical matches JSON files
    json_files = []
    for pattern in ['historical_matches_*.json', 'data/historical_matches_*.json']:
        json_files.extend(glob.glob(pattern))
    
    # Also check data directory for dated files
    data_files = glob.glob('data/historical_matches_????-??-??.json')
    json_files.extend(data_files)
    json_files = list(set(json_files))  # Remove duplicates
    json_files.sort()
    
    print(f"Found {len(json_files)} historical match files")
    
    # Track teams per league
    league_teams = {}
    
    # Extract league info from all files
    match_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                matches = json.load(f)
            
            for match in matches:
                match_count += 1
                
                url = match.get('url', '')
                league_code = match.get('league_code', '')
                country = match.get('country', '')
                league = match.get('league', '')
                league_url_path = match.get('league_url_path', '')
                country_code = match.get('country_code', '')
                home_team = match.get('home_team', '')
                away_team = match.get('away_team', '')
                
                match_id, prefix = extract_match_info(url)
                
                if league_code and prefix:
                    # Create combined key
                    key = f"{league_code}_{prefix}"
                    
                    if key not in leagues_db:
                        leagues_db[key] = {
                            'league_code': league_code,
                            'match_id_prefix': prefix,
                            'country': country,
                            'league': league,
                            'league_url_path': league_url_path,
                            'country_code': country_code,
                            'match_count': 0,
                            'teams': {}
                        }
                    
                    # Update match count
                    leagues_db[key]['match_count'] = leagues_db[key].get('match_count', 0) + 1
                    
                    # Track teams
                    if 'teams' not in leagues_db[key]:
                        leagues_db[key]['teams'] = {}
                    
                    for team in [home_team, away_team]:
                        if team and team not in leagues_db[key]['teams']:
                            leagues_db[key]['teams'][team] = {'home_matches': 0, 'away_matches': 0}
        
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    # Save the updated leagues database
    os.makedirs('data', exist_ok=True)
    
    with open(leagues_db_path, 'w', encoding='utf-8') as f:
        json.dump(leagues_db, f, indent=2, ensure_ascii=False)
    
    print(f"\nTotal matches processed: {match_count}")
    print(f"Total unique leagues: {len(leagues_db)}")
    print(f"Saved to {leagues_db_path}")
    
    # Print sample of the league database
    print("\n" + "=" * 60)
    print("Sample League Database:")
    print("=" * 60)
    for i, (key, info) in enumerate(list(leagues_db.items())[:10]):
        print(f"{key}: {info['league']} ({info['country']})")
        print(f"   URL path: {info['league_url_path']}")
        print(f"   Matches: {info.get('match_count', 0)}")


def update_leagues_from_historical():
    """Update leagues database from historical matches files."""
    main()


if __name__ == "__main__":
    main()
