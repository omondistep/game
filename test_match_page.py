#!/usr/bin/env python3
"""
Test script to analyze Forebet match page structure
"""

import requests
from bs4 import BeautifulSoup
import json
import re

def test_forebet_page(url):
    """Fetch and analyze a Forebet match page."""
    print(f"Testing URL: {url}\n")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Page title
        title = soup.find('title')
        print(f"Title: {title.text if title else 'Not found'}\n")
        
        # League info
        print("=" * 60)
        print("LEAGUE INFO")
        print("=" * 60)
        
        # Try onclick
        img_with_onclick = soup.find('img', onclick=True)
        if img_with_onclick:
            onclick = img_with_onclick.get('onclick', '')
            print(f"onclick: {onclick}")
            stag_match = re.search(r"getstag\(this,\d+,'([^']+)','([^']+)','([^']+)','([^']+)'", onclick)
            if stag_match:
                print(f"Country: {stag_match.group(1)}")
                print(f"League: {stag_match.group(2)}")
                print(f"URL Path: {stag_match.group(3)}")
                print(f"Country Code: {stag_match.group(4)}")
        
        # Try shortTag
        short_tag = soup.find('span', class_='shortTag')
        if short_tag:
            print(f"shortTag: {short_tag.get_text(strip=True)}")
        
        # Try meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            print(f"Meta description: {meta_desc.get('content', '')[:200]}")
        
        # Team names
        print("\n" + "=" * 60)
        print("TEAM NAMES")
        print("=" * 60)
        
        # Try different selectors
        team_selectors = [
            ('span', {'class': 'tname'}),
            ('div', {'class': 'team-name'}),
            ('a', {'class': 'team-link'}),
            ('h2', {'class': 'team-name'}),
        ]
        
        for tag, attrs in team_selectors:
            elems = soup.find_all(tag, attrs)
            if len(elems) >= 2:
                print(f"{tag}@{attrs}: {[e.get_text(strip=True) for e in elems[:2]]}")
        
        # Try page title
        if title:
            title_text = title.get_text()
            vs_match = re.search(r'^(.+?)\s+vs\s+(.+?)\s+-', title_text)
            if vs_match:
                print(f"From title - Home: {vs_match.group(1)}, Away: {vs_match.group(2)}")
        
        # Script data
        print("\n" + "=" * 60)
        print("SCRIPT DATA")
        print("=" * 60)
        
        scripts = soup.find_all('script')
        for i, script in enumerate(scripts):
            script_text = script.get_text()
            if 'teamNames' in script_text or 'homeTeam' in script_text or '"teams"' in script_text:
                print(f"\nScript {i}:")
                # Show relevant parts
                lines = script_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if 'teamNames' in line or 'homeTeam' in line or 'awayTeam' in line or '"teams"' in line:
                        print(f"  {line[:200]}")
        
        # Form data
        print("\n" + "=" * 60)
        print("FORM DATA")
        print("=" * 60)
        
        for i, script in enumerate(scripts):
            script_text = script.get_text()
            # Look for form patterns
            patterns = [
                r'homeTeam.*?form.*?\[([^\]]+)\]',
                r'awayTeam.*?form.*?\[([^\]]+)\]',
                r'"formHome".*?"([^"]+)"',
                r'"formAway".*?"([^"]+)"',
            ]
            for pattern in patterns:
                match = re.search(pattern, script_text)
                if match:
                    print(f"Pattern '{pattern[:30]}...': {match.group(1)[:100]}")
        
        # Save HTML for debugging
        with open('debug_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("\n" + "=" * 60)
        print("Full HTML saved to 'debug_page.html'")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    url = "https://www.forebet.com/en/football/matches/hh-export-cd-walter-ferretti-2420379"
    if len(sys.argv) > 1:
        url = sys.argv[1]
    
    test_forebet_page(url)
