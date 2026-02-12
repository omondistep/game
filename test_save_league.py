#!/usr/bin/env python3
import json

def _save_league_to_db(league_code, country, league, league_url_path, country_code):
    """Save league info to the new leagues_db.json format."""
    if not league_code:
        return
    
    try:
        leagues_db = {}
        try:
            with open('data/leagues_db.json', 'r', encoding='utf-8') as f:
                leagues_db = json.load(f)
        except:
            pass
        
        # Use league_code as key
        league_key = league_code
        
        if league_key not in leagues_db:
            leagues_db[league_key] = {
                'league_code': league_code,
                'country': country,
                'league': league,
                'league_url_path': league_url_path,
                'country_code': country_code,
                'match_count': 0
            }
        
        with open('data/leagues_db.json', 'w', encoding='utf-8') as f:
            json.dump(leagues_db, f, indent=2, ensure_ascii=False)
        print(f"Saved league: {league_key} -> {league}")
    except Exception as e:
        print(f"Error saving league to DB: {e}")

# Test with the Baku league
_save_league_to_db("23824", "Azerbaijan", "First Division", "football-tips-and-predictions-for-azerbaijan/first-division", "az")
