#!/usr/bin/env python3
"""
Build a comprehensive league database from historical match data.
This creates a complete mapping of all leagues with their details.
"""

import json
import os
from datetime import datetime
from collections import defaultdict

def build_comprehensive_league_db():
    """Build comprehensive league database from historical match data."""
    
    print("=" * 60)
    print("BUILDING COMPREHENSIVE LEAGUE DATABASE")
    print("=" * 60)
    
    # Data structures
    leagues_by_country = defaultdict(lambda: defaultdict(dict))
    short_code_lookup = {}
    league_url_lookup = {}
    match_id_prefix_lookup = {}
    
    # Read all historical match files
    data_dir = 'data'
    total_matches = 0
    
    for filename in os.listdir(data_dir):
        if filename.startswith('historical_matches_') and filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            print(f"Processing {filename}...")
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                
                for match in matches:
                    total_matches += 1
                    
                    # Extract all league-related fields
                    country = match.get('country', '').strip()
                    league = match.get('league', '').strip()
                    league_code = match.get('league_code', '').strip()
                    league_url_path = match.get('league_url_path', '').strip()
                    country_code = match.get('country_code', '').strip()
                    match_id = match.get('match_id', '').strip()
                    home_team = match.get('home_team', '').strip()
                    away_team = match.get('away_team', '').strip()
                    
                    if not country or not league:
                        continue
                    
                    # Build league entry
                    if league not in leagues_by_country[country]:
                        leagues_by_country[country][league] = {
                            'short_code': league_code,
                            'country_code': country_code,
                            'url_path': league_url_path,
                            'match_count': 0,
                            'teams': set(),
                            'match_id_prefixes': set()
                        }
                    
                    league_entry = leagues_by_country[country][league]
                    league_entry['match_count'] += 1
                    
                    # Update short_code if not set
                    if league_code and not league_entry.get('short_code'):
                        league_entry['short_code'] = league_code
                    
                    # Update country_code if not set
                    if country_code and not league_entry.get('country_code'):
                        league_entry['country_code'] = country_code
                    
                    # Update url_path if not set
                    if league_url_path and not league_entry.get('url_path'):
                        league_entry['url_path'] = league_url_path
                    
                    # Add teams
                    if home_team:
                        league_entry['teams'].add(home_team)
                    if away_team:
                        league_entry['teams'].add(away_team)
                    
                    # Add match_id prefix
                    if match_id and len(match_id) >= 3:
                        prefix = match_id[:3]
                        league_entry['match_id_prefixes'].add(prefix)
                        
                        # Build match_id_prefix lookup
                        if league_code:
                            prefix_key = f"{league_code}_{prefix}"
                            if prefix_key not in match_id_prefix_lookup:
                                match_id_prefix_lookup[prefix_key] = {
                                    'country': country,
                                    'league': league,
                                    'short_code': league_code,
                                    'country_code': country_code,
                                    'url_path': league_url_path
                                }
                    
                    # Build short_code lookup
                    if league_code:
                        if league_code not in short_code_lookup:
                            short_code_lookup[league_code] = {
                                'country': country,
                                'league': league,
                                'country_code': country_code,
                                'url_path': league_url_path,
                                'match_count': 0
                            }
                        short_code_lookup[league_code]['match_count'] += 1
                    
                    # Build url_path lookup
                    if league_url_path:
                        if league_url_path not in league_url_lookup:
                            league_url_lookup[league_url_path] = {
                                'country': country,
                                'league': league,
                                'short_code': league_code,
                                'country_code': country_code
                            }
            
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
    
    # Convert sets to lists for JSON serialization
    for country in leagues_by_country:
        for league in leagues_by_country[country]:
            entry = leagues_by_country[country][league]
            entry['teams'] = sorted(list(entry['teams']))
            entry['match_id_prefixes'] = sorted(list(entry['match_id_prefixes']))
    
    # Build the final database
    output = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_matches_processed': total_matches,
            'total_countries': len(leagues_by_country),
            'total_leagues': sum(len(leagues) for leagues in leagues_by_country.values()),
            'total_short_codes': len(short_code_lookup)
        },
        'countries': dict(leagues_by_country),
        'lookups': {
            'by_short_code': short_code_lookup,
            'by_url_path': league_url_lookup,
            'by_match_id_prefix': match_id_prefix_lookup
        }
    }
    
    # Save the database
    output_path = 'data/comprehensive_leagues_db.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total matches processed: {total_matches}")
    print(f"Total countries: {output['metadata']['total_countries']}")
    print(f"Total leagues: {output['metadata']['total_leagues']}")
    print(f"Total short codes: {output['metadata']['total_short_codes']}")
    
    # Show sample
    print("\nSample entries by country:")
    for i, (country, leagues) in enumerate(list(leagues_by_country.items())[:5]):
        print(f"\n{country}:")
        for league, info in list(leagues.items())[:3]:
            sc = info.get('short_code', 'N/A')
            mc = info.get('match_count', 0)
            teams_count = len(info.get('teams', []))
            prefixes = info.get('match_id_prefixes', [])
            print(f"  - {league}: {sc} ({mc} matches, {teams_count} teams, prefixes: {prefixes})")
    
    return output


if __name__ == "__main__":
    build_comprehensive_league_db()
