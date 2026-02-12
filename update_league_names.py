#!/usr/bin/env python3
"""Update league names in leagues_db.json from URL paths."""

import json

def extract_league_name(league_url_path):
    """Extract league name from URL path."""
    if not league_url_path:
        return ""
    # Get the last part of the URL path
    parts = league_url_path.split('/')
    league_from_url = parts[-1] if parts else ""
    # Convert dash-separated to title case
    return league_from_url.replace('-', ' ').title()

# Load existing leagues
with open('data/leagues_db.json', 'r', encoding='utf-8') as f:
    leagues = json.load(f)

# Update league names
updated = 0
for key, info in leagues.items():
    if not info.get('league') and info.get('league_url_path'):
        league_name = extract_league_name(info['league_url_path'])
        if league_name:
            info['league'] = league_name
            updated += 1

# Save updated leagues
with open('data/leagues_db.json', 'w', encoding='utf-8') as f:
    json.dump(leagues, f, indent=2, ensure_ascii=False)

print(f"Updated {updated} league names")
