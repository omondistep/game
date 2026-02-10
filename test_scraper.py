#!/usr/bin/env python3
"""
Test script to explore Forebet page structure and identify scrapable data
"""

import requests
from bs4 import BeautifulSoup
import json

def test_forebet_scraping(url):
    """
    Test scraping a Forebet match page to identify available data
    """
    print(f"Testing URL: {url}\n")
    
    # Set headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        # Fetch the page
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.content)} bytes\n")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract page title
        title = soup.find('title')
        print(f"Page Title: {title.text if title else 'Not found'}\n")
        
        # Look for team names
        print("=" * 60)
        print("TEAM NAMES")
        print("=" * 60)
        team_elements = soup.find_all(['span', 'div', 'h1', 'h2'], class_=lambda x: x and ('team' in x.lower() or 'name' in x.lower()))
        for i, elem in enumerate(team_elements[:10]):
            print(f"{i+1}. Class: {elem.get('class')} | Text: {elem.text.strip()[:100]}")
        
        # Look for standings/table data
        print("\n" + "=" * 60)
        print("STANDINGS/TABLE DATA")
        print("=" * 60)
        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables")
        for i, table in enumerate(tables[:3]):
            print(f"\nTable {i+1}:")
            print(f"  Classes: {table.get('class')}")
            rows = table.find_all('tr')
            print(f"  Rows: {len(rows)}")
            if rows:
                print(f"  First row: {rows[0].text.strip()[:100]}")
        
        # Look for last matches/form
        print("\n" + "=" * 60)
        print("LAST MATCHES / FORM DATA")
        print("=" * 60)
        form_elements = soup.find_all(['div', 'span', 'ul'], class_=lambda x: x and ('form' in str(x).lower() or 'last' in str(x).lower() or 'recent' in str(x).lower()))
        for i, elem in enumerate(form_elements[:10]):
            print(f"{i+1}. Class: {elem.get('class')} | Text: {elem.text.strip()[:100]}")
        
        # Look for odds/predictions
        print("\n" + "=" * 60)
        print("ODDS/PREDICTIONS DATA")
        print("=" * 60)
        odds_elements = soup.find_all(['div', 'span', 'td'], class_=lambda x: x and ('odd' in str(x).lower() or 'pred' in str(x).lower() or 'prob' in str(x).lower()))
        for i, elem in enumerate(odds_elements[:15]):
            print(f"{i+1}. Class: {elem.get('class')} | Text: {elem.text.strip()[:100]}")
        
        # Look for home/away statistics
        print("\n" + "=" * 60)
        print("HOME/AWAY STATISTICS")
        print("=" * 60)
        stats_elements = soup.find_all(['div', 'span', 'td'], class_=lambda x: x and ('home' in str(x).lower() or 'away' in str(x).lower() or 'stat' in str(x).lower()))
        for i, elem in enumerate(stats_elements[:15]):
            print(f"{i+1}. Class: {elem.get('class')} | Text: {elem.text.strip()[:100]}")
        
        # Look for score predictions
        print("\n" + "=" * 60)
        print("SCORE PREDICTIONS")
        print("=" * 60)
        score_elements = soup.find_all(['div', 'span'], class_=lambda x: x and ('score' in str(x).lower() or 'result' in str(x).lower()))
        for i, elem in enumerate(score_elements[:10]):
            print(f"{i+1}. Class: {elem.get('class')} | Text: {elem.text.strip()[:100]}")
        
        # Look for over/under data
        print("\n" + "=" * 60)
        print("OVER/UNDER DATA")
        print("=" * 60)
        ou_elements = soup.find_all(['div', 'span', 'td'], class_=lambda x: x and ('over' in str(x).lower() or 'under' in str(x).lower() or 'goal' in str(x).lower()))
        for i, elem in enumerate(ou_elements[:10]):
            print(f"{i+1}. Class: {elem.get('class')} | Text: {elem.text.strip()[:100]}")
        
        # Save HTML for manual inspection
        with open('forebet_page.html', 'w', encoding='utf-8') as f:
            f.write(soup.prettify())
        print("\n" + "=" * 60)
        print("Full HTML saved to 'forebet_page.html' for manual inspection")
        print("=" * 60)
        
        # Look for JSON data in script tags
        print("\n" + "=" * 60)
        print("JAVASCRIPT DATA")
        print("=" * 60)
        scripts = soup.find_all('script')
        print(f"Found {len(scripts)} script tags")
        for i, script in enumerate(scripts):
            if script.string and len(script.string) > 100:
                content = script.string[:200]
                if 'var' in content or 'const' in content or 'let' in content or '{' in content:
                    print(f"\nScript {i+1} (first 200 chars):")
                    print(content)
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching page: {e}")
        return False
    except Exception as e:
        print(f"Error parsing page: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test URL
    test_url = "https://www.forebet.com/en/football/matches/juventus-lazio-2344437"
    
    print("FOREBET SCRAPER TEST")
    print("=" * 60)
    print("This script will test scraping the Forebet page to identify")
    print("what data is available for extraction.\n")
    
    success = test_forebet_scraping(test_url)
    
    if success:
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("Review the output above to identify scrapable data.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Test failed. Check the error messages above.")
        print("=" * 60)
