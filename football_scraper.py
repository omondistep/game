#!/usr/bin/env python3
"""
Football Match Data Scraper for Forebet
Extracts comprehensive match data including team stats, form, odds, standings,
head-to-head, home/away performance, shots, passes, trends, and more.
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import os
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta


class ForebetScraper:
    """Scraper for extracting football match data from Forebet"""

    def __init__(self):
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        }
        
        # Load leagues database
        self.leagues_db = {}
        self._load_leagues_db()
    
    def _load_leagues_db(self):
        """Load leagues from data/leagues_db.json and data/comprehensive_leagues_db.json."""
        # Load comprehensive database first (has more details)
        comprehensive_path = 'data/comprehensive_leagues_db.json'
        if os.path.exists(comprehensive_path):
            try:
                with open(comprehensive_path, 'r', encoding='utf-8') as f:
                    comp_db = json.load(f)
                
                # Load lookups
                lookups = comp_db.get('lookups', {})
                self.short_code_lookup = lookups.get('by_short_code', {})
                self.url_path_lookup = lookups.get('by_url_path', {})
                self.match_id_prefix_lookup = lookups.get('by_match_id_prefix', {})
                
                # Build leagues_db from match_id_prefix_lookup for backward compatibility
                self.leagues_db = self.match_id_prefix_lookup.copy()
                
                print(f"Loaded comprehensive database: {len(self.short_code_lookup)} short codes, {len(self.match_id_prefix_lookup)} prefixes")
            except Exception as e:
                print(f"Error loading comprehensive database: {e}")
                self.leagues_db = {}
                self.short_code_lookup = {}
                self.url_path_lookup = {}
                self.match_id_prefix_lookup = {}
        else:
            # Fallback to old leagues_db.json
            db_path = 'data/leagues_db.json'
            if os.path.exists(db_path):
                try:
                    with open(db_path, 'r', encoding='utf-8') as f:
                        self.leagues_db = json.load(f)
                    print(f"Loaded {len(self.leagues_db)} leagues from database")
                except Exception as e:
                    print(f"Error loading leagues_db: {e}")
                    self.leagues_db = {}
            
            self.short_code_lookup = {}
            self.url_path_lookup = {}
            self.match_id_prefix_lookup = {}
        
        # Also load league_mapping.json for 5-digit code lookups
        mapping_path = 'data/league_mapping.json'
        if os.path.exists(mapping_path):
            try:
                with open(mapping_path, 'r', encoding='utf-8') as f:
                    self.league_mapping = json.load(f)
            except Exception as e:
                print(f"Error loading league_mapping: {e}")
                self.league_mapping = {}
        else:
            self.league_mapping = {}
    
    def _extract_league_info_from_page(self, soup: BeautifulSoup, url: str = None) -> Dict:
        """Extract league info from page using the historical_matches approach.
        
        Returns dict with: league_code, country, league, league_url_path, country_code
        """
        info = {
            'league_code': None,
            'country': None,
            'league': None,
            'league_url_path': None,
            'country_code': None
        }
        
        # Country code to country name mapping (for numeric codes like '281' = Gibraltar)
        country_code_map = {
            '281': 'Gibraltar',
            # Add more as needed
        }
        
        # Extract match_id from URL
        match_id_match = re.search(r'-(\d{5,8})$', url) if url else None
        url_match_id = match_id_match.group(1) if match_id_match else None
        
        # Method 1: Find onclick handler that matches the URL match_id
        # This is the most reliable method
        if url_match_id:
            for img in soup.find_all('img', onclick=True):
                onclick = img.get('onclick', '')
                
                # Pattern: getstag(this,match_id,'Country','League','url_path','country_code')
                stag_match = re.search(r"getstag\(this,(\d+),'([^']*)','([^']*)','([^']*)','([^']*)'", onclick)
                if stag_match:
                    match_id = stag_match.group(1)
                    if match_id == url_match_id:
                        country = stag_match.group(2)
                        league = stag_match.group(3)
                        if country or league:  # At least one field populated
                            info['country'] = country if country else None
                            info['league'] = league if league else None
                            info['league_url_path'] = stag_match.group(4)
                            info['country_code'] = stag_match.group(5)
                            # Get shortTag from parent elements of this img (not global find)
                            parent = img.parent
                            for _ in range(5):  # Check up to 5 levels up
                                if parent is None:
                                    break
                                short_tag = parent.find('span', class_='shortTag')
                                if short_tag:
                                    info['league_code'] = short_tag.get_text(strip=True)
                                    break
                                parent = parent.parent
                            return info
                
                # Pattern: getstag(this,match_id,'','','','numeric_code')
                empty_match = re.search(r"getstag\(this,\s*(\d+),\s*'',\s*'',\s*'',\s*'(\d+)'\)", onclick)
                if empty_match:
                    match_id = empty_match.group(1)
                    if match_id == url_match_id:
                        numeric_code = empty_match.group(2)
                        if numeric_code in country_code_map:
                            info['country'] = country_code_map[numeric_code]
                            info['country_code'] = numeric_code
                            # Generate short_code from country name
                            country_prefix = info['country'][:2].capitalize()
                            info['league_code'] = f"{country_prefix}1"
                            return info
        
        # Method 2: Get shortTag (like "Az2", "Ve1", "Cl1")
        short_tag = soup.find('span', class_='shortTag')
        if short_tag:
            short_code = short_tag.get_text(strip=True)
            if short_code and len(short_code) >= 2:
                info['league_code'] = short_code
                
                # Try to look up in comprehensive database using short_code
                if hasattr(self, 'short_code_lookup') and self.short_code_lookup:
                    if short_code in self.short_code_lookup:
                        db_info = self.short_code_lookup[short_code]
                        info['country'] = db_info.get('country')
                        info['league'] = db_info.get('league')
                        info['league_url_path'] = db_info.get('league_url_path')
                        info['country_code'] = db_info.get('country_code')
                        return info  # Found in database, return early
        
        # Method 3: Check onclick handlers for any with populated country/league
        for img in soup.find_all('img', onclick=True):
            onclick = img.get('onclick', '')
            
            # Pattern: getstag(this,match_id,'Country','League','url_path','country_code')
            stag_match = re.search(r"getstag\(this,(\d+),'([^']*)','([^']*)','([^']*)','([^']*)'", onclick)
            if stag_match:
                country = stag_match.group(2)
                league = stag_match.group(3)
                if country and league:  # Both fields populated
                    info['country'] = country
                    info['league'] = league
                    info['league_url_path'] = stag_match.group(4)
                    info['country_code'] = stag_match.group(5)
                    # Generate league_code from country if not set
                    if not info['league_code']:
                        country_prefix = country[:2].capitalize()
                        info['league_code'] = f"{country_prefix}1"
                    return info  # Found reliable info, return early
        
        # Method 4: Check for numeric country codes (less reliable)
        for img in soup.find_all('img', onclick=True):
            onclick = img.get('onclick', '')
            
            # Pattern: getstag(this,match_id,'','','','numeric_code') - empty country/league
            empty_match = re.search(r"getstag\(this,\s*(\d+),\s*'',\s*'',\s*'',\s*'(\d+)'\)", onclick)
            if empty_match:
                numeric_code = empty_match.group(2)
                if numeric_code in country_code_map:
                    info['country'] = country_code_map[numeric_code]
                    info['country_code'] = numeric_code
                    # Generate short_code from country name (e.g., Gibraltar -> Gi1)
                    if info['country']:
                        country_prefix = info['country'][:2].capitalize()
                        info['league_code'] = f"{country_prefix}1"
                    return info  # Found info, return early
        
        # Method 3: Try to extract from breadcrumb or page title
        if not info['league'] or not info['country']:
            # Look for league info in breadcrumb links
            breadcrumb = soup.find('div', class_='breadcrumb') or soup.find('nav', class_='breadcrumb')
            if breadcrumb:
                links = breadcrumb.find_all('a')
                for link in links:
                    href = link.get('href', '')
                    text = link.get_text(strip=True)
                    # Pattern: /football-predictions-for-{country}/{league}
                    if 'predictions-for' in href:
                        parts = href.split('/')
                        if len(parts) >= 2:
                            if not info['country']:
                                # Extract country from URL
                                country_part = parts[-2] if len(parts) > 1 else ''
                                country_match = re.search(r'predictions-for-(.+)', country_part)
                                if country_match:
                                    info['country'] = country_match.group(1).replace('-', ' ').title()
                            if not info['league']:
                                info['league'] = parts[-1].replace('-', ' ').title()
        
        # Method 4: Try meta tags
        if not info['league']:
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                content = meta_desc.get('content')
                # Pattern: "...Netherlands Eerste Divisie..." or similar
                league_patterns = [
                    r'(\w+)\s+(Premier League|La Liga|Serie A|Bundesliga|Ligue \d|Eredivisie|Eerste Divisie|Championship|League \d|Primera Division|Liga)',
                    r'(Netherlands|England|Spain|Italy|Germany|France|Brazil|Argentina)\s+([\w\s]+?)(?:\s+predictions|\s+match|\.)',
                ]
                for pattern in league_patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        if not info['country']:
                            info['country'] = match.group(1)
                        if not info['league']:
                            info['league'] = match.group(2).strip()
                        break
        
        # Method 5: Look for league info in any script tags
        if not info['league']:
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'league' in script.string.lower():
                    # Try to extract league name from JSON-like content
                    league_match = re.search(r'"league"\s*:\s*"([^"]+)"', script.string)
                    if league_match:
                        info['league'] = league_match.group(1)
                        break
                    country_match = re.search(r'"country"\s*:\s*"([^"]+)"', script.string)
                    if country_match and not info['country']:
                        info['country'] = country_match.group(1)
        
        # Method 6: Extract league name from standings section or tabs
        if not info['league']:
            # Look for league name in standings header
            # Pattern: "4th place\nGibraltar National League\n2nd place"
            standings_div = soup.find('div', class_=re.compile(r'standings|table', re.I))
            if standings_div:
                # Get text and look for league name patterns
                text = standings_div.get_text(separator='\n', strip=True)
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                for i, line in enumerate(lines):
                    # Look for lines that look like league names (contain "League", "Division", "Liga", etc.)
                    if any(kw in line for kw in ['League', 'Division', 'Liga', 'Serie', 'Premier', 'Championship', 'Cup']):
                        # Skip if it's a team name with form (WWLLW pattern)
                        if not re.search(r'[WL]{3,}', line):
                            info['league'] = line
                            break
            
            # Also try tab links (e.g., "All National League Champions League...")
            if not info['league']:
                tabs = soup.find('div', class_=re.compile(r'tabs|filter', re.I))
                if tabs:
                    tab_links = tabs.find_all('a')
                    for link in tab_links:
                        text = link.get_text(strip=True)
                        # Skip "All" and short texts
                        if text and text != 'All' and len(text) > 3:
                            # Check if it looks like a league name
                            if any(kw in text for kw in ['League', 'Division', 'Liga', 'Serie', 'Premier', 'Championship']):
                                info['league'] = text
                                break
        
        # Method 7: Look for league name in page headings
        if not info['league']:
            # Look for h1, h2, h3 with league-like content
            for heading in soup.find_all(['h1', 'h2', 'h3']):
                text = heading.get_text(strip=True)
                if any(kw in text for kw in ['League', 'Division', 'Liga', 'Serie', 'Premier', 'Championship', 'Cup']):
                    # Skip if it contains team names or scores
                    if ' - ' not in text and ' vs ' not in text.lower():
                        info['league'] = text
                        break
        
        return info
    
    def _get_league_info(self, soup: BeautifulSoup, url: str) -> Dict:
        """Get complete league info from comprehensive leagues database.
        
        Uses multiple lookup methods in order of reliability:
        1. By short_code + url_path match (most reliable)
        2. By full match_id - for known matches
        3. By url_path only
        4. Use page_info if it has valid league name (from onclick handler)
        """
        # Extract match_id from URL
        match_id_match = re.search(r'-(\d{5,8})$', url) if url else None
        match_id = match_id_match.group(1) if match_id_match else None
        
        # Get page info for short_code - pass URL for matching
        page_info = self._extract_league_info_from_page(soup, url)
        league_code = page_info.get('league_code')
        page_url_path = page_info.get('league_url_path')
        
        if match_id:
            # Method 1: Try short_code lookup with url_path verification
            # This ensures we don't use wrong data when short_code is ambiguous
            if league_code and hasattr(self, 'short_code_lookup') and self.short_code_lookup:
                if league_code in self.short_code_lookup:
                    db_info = self.short_code_lookup[league_code]
                    db_url_path = db_info.get('url_path', db_info.get('league_url_path', ''))
                    # Only use db_info if url_path matches (same league)
                    if page_url_path and db_url_path and page_url_path == db_url_path:
                        return {
                            'match_id': match_id,
                            'league_code': league_code,
                            'country': db_info.get('country'),
                            'league': db_info.get('league'),
                            'league_url_path': db_url_path,
                            'country_code': db_info.get('country_code')
                        }
                    # url_path mismatch - short_code is ambiguous, use page_info instead
                    # Fall through to Method 4
            
            # Method 2: Try exact match_id lookup (for known matches)
            if match_id in self.leagues_db:
                db_info = self.leagues_db[match_id]
                return {
                    'match_id': match_id,
                    'league_code': db_info.get('league_code'),
                    'country': db_info.get('country'),
                    'league': db_info.get('league'),
                    'league_url_path': db_info.get('league_url_path'),
                    'country_code': db_info.get('country_code')
                }
            
            # Method 3: Try url_path lookup
            if page_url_path and hasattr(self, 'url_path_lookup') and self.url_path_lookup:
                if page_url_path in self.url_path_lookup:
                    db_info = self.url_path_lookup[page_url_path]
                    return {
                        'match_id': match_id,
                        'league_code': db_info.get('short_code') or db_info.get('league_code'),
                        'country': db_info.get('country'),
                        'league': db_info.get('league'),
                        'league_url_path': page_url_path,
                        'country_code': db_info.get('country_code')
                    }
            
            # Method 4: Use page_info if it has valid league name (from onclick handler)
            if page_info.get('league') and page_info.get('league_code'):
                # Save to database for future lookups
                self._save_new_league_to_db(league_code, page_info, match_id)
                return {
                    'match_id': match_id,
                    'league_code': league_code,
                    'country': page_info.get('country'),
                    'league': page_info.get('league'),
                    'league_url_path': page_info.get('league_url_path'),
                    'country_code': page_info.get('country_code')
                }
            
            # Not in database - save new league entry using short_code
            if league_code:
                self._save_new_league_to_db(league_code, page_info, match_id)
        
        # Fallback: return page info
        if match_id:
            page_info['match_id'] = match_id
        return page_info
    
    def _save_new_league_to_db(self, short_code: str, page_info: Dict, match_id: str):
        """Save a new league entry to the database using short_code as key."""
        if not short_code or not page_info:
            return
        
        try:
            # Create new entry
            new_entry = {
                'league_code': page_info.get('league_code', ''),
                'country': page_info.get('country', ''),
                'league': page_info.get('league', ''),
                'league_url_path': page_info.get('league_url_path', ''),
                'country_code': page_info.get('country_code', ''),
                'match_count': 1,
                'teams': {},
                'source': 'live_prediction'
            }
            
            # Add to short_code_lookup
            if hasattr(self, 'short_code_lookup'):
                self.short_code_lookup[short_code] = new_entry
            
            # Add to leagues_db for backward compatibility
            self.leagues_db[short_code] = new_entry
            
            # Save to file
            import os
            os.makedirs('data', exist_ok=True)
            with open('data/leagues_db.json', 'w', encoding='utf-8') as f:
                json.dump(self.leagues_db, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ Added new league to database: {short_code} -> {new_entry.get('country', '')} {new_entry.get('league', '')}")
        except Exception as e:
            print(f"  ⚠ Could not save new league: {e}")

    def _extract_league_code(self, url: str) -> Optional[str]:
        """Extract league code (first 5 digits) from URL.
        
        URL format: /matches/{team1}-{team2}-{league_code}{match_id}
        Example: /matches/cska-sofia-cska-1948-2423671 -> league_code = 24236
        """
        # Match 5-digit code before 1-3 digit match ID at end
        match = re.search(r'-(\d{5})\d{1,3}$', url)
        if match:
            return match.group(1)
        return None
    
    def _get_league_name(self, code: str, prompt_user: bool = False) -> Optional[str]:
        """Get league name from code, prompting user if unknown and prompt_user=True."""
        if not code:
            return None
        
        # Check known leagues
        if code in self.LEAGUE_CODES:
            return self.LEAGUE_CODES[code]
        
        # Check saved unknown leagues
        try:
            with open('data/unknown_leagues.json', 'r') as f:
                unknown = json.load(f)
            if code in unknown:
                value = unknown[code]
                # Handle both old format (string) and new format (dict with suggested_name)
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict):
                    return value.get('suggested_name') or value.get('name')
        except:
            pass
        
        # If prompt_user is True, ask user for league name
        if prompt_user:
            try:
                league = input(f"Enter league name for code {code}: ").strip()
                if league:
                    # Save to unknown_leagues.json for future use
                    try:
                        with open('data/unknown_leagues.json', 'r') as f:
                            unknown = json.load(f)
                    except:
                        unknown = {}
                    unknown[code] = league
                    with open('data/unknown_leagues.json', 'w') as f:
                        json.dump(unknown, f, indent=2)
                    return league
            except KeyboardInterrupt:
                return None
        
        # Return placeholder with code so training can proceed without user input
        return f"League {code}"

    def update_league_name(self, code: str, new_name: str) -> bool:
        """Update league name for a given code.
        
        Args:
            code: League code (5-digit code from URL)
            new_name: New league name to use
            
        Returns:
            True if update was successful, False otherwise
        """
        if not code or not new_name:
            return False
        
        # Update in unknown_leagues.json
        try:
            with open('data/unknown_leagues.json', 'r') as f:
                unknown = json.load(f)
        except:
            unknown = {}
        
        # Check if code exists in unknown_leagues or is a known league
        if code in self.LEAGUE_CODES or code in unknown:
            unknown[code] = new_name
            with open('data/unknown_leagues.json', 'w') as f:
                json.dump(unknown, f, indent=2)
            return True
        
        return False


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _extract_league_from_page(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract league name from match page HTML."""
        # First, check if we have a saved league name for this URL's code
        league_code = self._extract_league_code(url)
        if league_code:
            try:
                with open('data/unknown_leagues.json', 'r') as f:
                    unknown = json.load(f)
                if league_code in unknown:
                    value = unknown[league_code]
                    # Handle both old format (string) and new format (dict with suggested_name)
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, dict):
                        return value.get('suggested_name') or value.get('name')
            except:
                pass
        
        # Method 1: Try to find league info from onclick attributes (for major leagues)
        img_with_onclick = soup.find('img', onclick=True)
        if img_with_onclick:
            onclick = img_with_onclick.get('onclick', '')
            # Pattern: getstag(this,match_id,'Country','League',...)
            stag_match = re.search(r"getstag\(this,\d+,'([^']+)','([^']+)'", onclick)
            if stag_match:
                country = stag_match.group(1)
                league = stag_match.group(2)
                if league and len(league) > 2:
                    return league
        
        # Method 2: Try shortTag span (returns abbreviated codes like "RoC", "EPL")
        short_tag = soup.find('span', class_='shortTag')
        if short_tag:
            short_code = short_tag.get_text(strip=True)
            if short_code and len(short_code) >= 2:
                # Try to expand short code using known mappings
                expanded = self._expand_league_code(short_code, url)
                if expanded:
                    return expanded
                # If we can't expand, use the short code as fallback
                return short_code
        
        # Method 3: Extract from meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            content = meta_desc.get('content')
            # Pattern: "...this match of {Country} {League}..."
            match = re.search(r'this match of ([A-Za-z]+(?:\s+[A-Za-z]+)*)', content)
            if match:
                league_text = match.group(1).strip()
                # Filter out common non-league words
                words_to_skip = ['football', 'predictions', 'statistics', 'match']
                words = [w for w in league_text.split() if w.lower() not in words_to_skip]
                if words:
                    return ' '.join(words)
        
        # Method 4: Look for country in URL or page
        return None
    
    def _expand_league_code(self, short_code: str, url: str) -> Optional[str]:
        """Expand abbreviated league code to full name."""
        # Common short codes mapped to full names
        league_mappings = {
            'RoC': 'Romania Cupa',
            'RoL': 'Romania Liga 1',
            'RoL2': 'Romania Liga 2',
            'EPL': 'England Premier League',
            'ELC': 'England Championship',
            'EL1': 'England League One',
            'EL2': 'England League Two',
            'LL': 'Spain La Liga',
            'LL2': 'Spain Segunda Division',
            'SA': 'Italy Serie A',
            'SB': 'Italy Serie B',
            'BL1': 'Germany Bundesliga',
            'BL2': 'Germany Bundesliga 2',
            'FL1': 'France Ligue 1',
            'FL2': 'France Ligue 2',
            'NL': 'Netherlands Eredivisie',
            'PL': 'Portugal Primeira Liga',
            'SC': 'Scotland Premiership',
            'BEL': 'Belgium Pro League',
        }
        
        # Try exact match first
        if short_code in league_mappings:
            return league_mappings[short_code]
        
        # Try to find league info from URL
        url_lower = url.lower()
        
        # Common league names to check in URL
        for full_name in ['cupa', 'liga', 'premier', 'championship', 'bundesliga', 
                          'ligue', 'eredivisie', 'primeira', 'premiership']:
            if full_name in url_lower:
                if 'cupa' in url_lower:
                    return 'Romania Cupa'
                if 'liga' in url_lower:
                    # Try to determine which liga
                    if 'liga-1' in url_lower or 'liga-2' in url_lower:
                        continue  # Will be handled by code
                    if 'romania' in url_lower:
                        return 'Romania Liga'
                    return 'Liga'
                if 'premier' in url_lower:
                    if 'england' in url_lower or 'ukraine' in url_lower:
                        return 'Premier League'
                    return 'Premier League'
                if 'championship' in url_lower:
                    return 'Championship'
        
        return None
    
    def _auto_save_league(self, code: str, league_name: str):
        """Auto-save league to database."""
        if not code or not league_name:
            return
        
        # Check if already known
        if code in self.LEAGUE_CODES:
            return
        
        # Check unknown_leagues.json
        try:
            with open('data/unknown_leagues.json', 'r') as f:
                unknown = json.load(f)
            if code in unknown:
                return
        except:
            unknown = {}
        
        # Save new league
        unknown[code] = league_name
        with open('data/unknown_leagues.json', 'w') as f:
            json.dump(unknown, f, indent=2)
        print(f"  📚 Auto-saved league: {code} -> {league_name}")

    def scrape_match(self, url: str, prompt_user: bool = False) -> Optional[Dict]:
        """Scrape all available match data from a Forebet URL.
        
        Args:
            url: The Forebet match URL
            prompt_user: If True, prompt user for league name if unknown.
                         If False, use placeholder "League {code}".
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if match has been played and extract result
            result = self._extract_result_from_soup(soup)
            
            # Get league info using the historical approach (from page + database)
            league_info = self._get_league_info(soup, url)
            
            match_data = {
                'url': url,
                'timestamp': time.time(),
                'teams': self._extract_teams(soup),
                'match_info': league_info,
                'standings': self._extract_standings(soup),
                'league_table': self._extract_league_table(soup),
                'form': self._extract_form(soup),
                'last_6_matches': self._extract_last_6_matches(soup),
                'home_matches': self._extract_home_away_matches(soup, 'home'),
                'away_matches': self._extract_home_away_matches(soup, 'away'),
                'head_to_head': self._extract_head_to_head(soup),
                'odds': self._extract_odds(soup),
                'predictions': self._extract_predictions(soup),
                'goals_stats': self._extract_goals_stats(soup),
                'over_under_stats': self._extract_over_under_stats(soup),
                'both_to_score': self._extract_bts_stats(soup),
                'shots_stats': self._extract_shots_stats(soup),
                'passes_stats': self._extract_passes_stats(soup),
                'attacks_stats': self._extract_attacks_stats(soup),
                'discipline': self._extract_discipline(soup),
                'other_stats': self._extract_other_stats(soup),
                'trends': self._extract_trends(soup),
                'match_intro': self._extract_match_intro(soup),
                'injuries': self._extract_injuries(soup),
                'next_matches': self._extract_next_matches(soup),
                'js_detailed_stats': self._extract_javascript_data(soup),
                'actual_result': result,  # None if not played, result dict if played
            }
            return match_data
        except Exception as e:
            error_msg = f"Error scraping match: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            # Return error details instead of None
            return {'error': error_msg, 'url': url}
    
    def _extract_result_from_soup(self, soup: BeautifulSoup) -> Optional[Dict]:
        """Extract actual result from parsed HTML if match has been played."""
        result: Dict = {
            'home_score': None, 'away_score': None,
            'result': None, 'total_goals': None,
            'over_under_2_5': None,
        }
        
        page_text = soup.get_text()
        
        # Check if "FT" (Full Time) indicator is present
        ft_pattern = re.search(r'(\d+)\s*-\s*(\d+)\s*FT', page_text)
        if not ft_pattern:
            # Check for "Final" indicator
            ft_pattern = re.search(r'(\d+)\s*-\s*(\d+)\s*Final', page_text, re.IGNORECASE)
        
        if ft_pattern:
            result['home_score'] = int(ft_pattern.group(1))
            result['away_score'] = int(ft_pattern.group(2))
        else:
            # Fallback: look for score element
            score_elem = soup.find('span', class_='lscrsp')
            if score_elem:
                text = score_elem.get_text(strip=True)
                m = re.search(r'(\d+)\s*-\s*(\d+)', text)
                if m:
                    result['home_score'] = int(m.group(1))
                    result['away_score'] = int(m.group(2))
        
        if result['home_score'] is not None:
            result['total_goals'] = result['home_score'] + result['away_score']
            result['over_under_2_5'] = 'Over' if result['total_goals'] > 2.5 else 'Under'
            if result['home_score'] > result['away_score']:
                result['result'] = '1'
            elif result['home_score'] < result['away_score']:
                result['result'] = '2'
            else:
                result['result'] = 'X'
            return result
        
        return None  # Match not played yet

    def is_match_played(self, url: str) -> bool:
        """Check if a match has been played (has FT indicator and/or date in past)."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check 1: Look for "FT" (Full Time) indicator
            page_text = soup.get_text()
            if 'FT' in page_text:
                # Double check it's not in a future context
                ft_pattern = re.search(r'(\d+)\s*-\s*(\d+)\s*FT', page_text)
                if ft_pattern:
                    return True
            
            # Check 2: Look for "Final" indicator
            if 'Final' in page_text:
                final_pattern = re.search(r'(\d+)\s*-\s*(\d+)\s*Final', page_text, re.IGNORECASE)
                if final_pattern:
                    return True
            
            # Check 3: Look for date and check if it's in the past
            date_elem = soup.find('span', class_='date')
            if date_elem:
                text = date_elem.get_text(strip=True)
                d = re.search(r'(\d{2})/(\d{2})/(\d{4})', text)
                if d:
                    day, month, year = int(d.group(1)), int(d.group(2)), int(d.group(3))
                    match_date = datetime(year, month, day)
                    # If match date is today or in the past, it may have been played
                    # (we still need score to confirm)
                    if match_date <= datetime.now():
                        # Check if there's a score displayed
                        score_patterns = [
                            r'(\d+)\s*-\s*(\d+)',  # Basic score pattern
                            r'(\d+)\s*:\s*(\d+)',  # Alternative score pattern
                        ]
                        for pattern in score_patterns:
                            if re.search(pattern, page_text):
                                return True
            
            # Check 4: Look for score element (lscrsp class)
            score_elem = soup.find('span', class_='lscrsp')
            if score_elem:
                text = score_elem.get_text(strip=True)
                if re.search(r'(\d+)\s*-\s*(\d+)', text):
                    return True
            
            return False
        except Exception as e:
            print(f"Error checking if match is played: {e}")
            return False

    def extract_actual_result(self, url: str) -> Optional[Dict]:
        """Extract actual match result after the game is played."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            result: Dict = {
                'home_score': None, 'away_score': None,
                'result': None, 'total_goals': None,
                'over_under_2_5': None,
            }

            # Forebet shows the score in the prediction row after the match
            score_elem = soup.find('span', class_='lscrsp')
            if score_elem:
                text = score_elem.get_text(strip=True)
                m = re.search(r'(\d+)\s*-\s*(\d+)', text)
                if m:
                    result['home_score'] = int(m.group(1))
                    result['away_score'] = int(m.group(2))

            if result['home_score'] is None:
                # Fallback: look for any score-like element
                for elem in soup.find_all(['span', 'div'], class_=re.compile(r'score|result')):
                    text = elem.get_text(strip=True)
                    m = re.search(r'(\d+)\s*-\s*(\d+)', text)
                    if m:
                        result['home_score'] = int(m.group(1))
                        result['away_score'] = int(m.group(2))
                        break

            if result['home_score'] is not None:
                result['total_goals'] = result['home_score'] + result['away_score']
                result['over_under_2_5'] = 'Over' if result['total_goals'] > 2.5 else 'Under'
                if result['home_score'] > result['away_score']:
                    result['result'] = '1'
                elif result['home_score'] < result['away_score']:
                    result['result'] = '2'
                else:
                    result['result'] = 'X'

            return result
        except Exception as e:
            print(f"Error extracting result: {e}")
            return None

    # ------------------------------------------------------------------
    # Private extraction methods
    # ------------------------------------------------------------------

    def _extract_teams(self, soup: BeautifulSoup) -> Dict:
        """Extract team names."""
        teams = {'home': None, 'away': None}
        try:
            # Try to find team_name spans
            team_elems = soup.find_all('span', class_='team_name')
            if len(team_elems) >= 2:
                teams['home'] = team_elems[0].get_text(strip=True)
                teams['away'] = team_elems[1].get_text(strip=True)
                return teams
            
            # Try to find team_name class
            team_elems = soup.find_all(class_='team_name')
            if len(team_elems) >= 2:
                teams['home'] = team_elems[0].get_text(strip=True)
                teams['away'] = team_elems[1].get_text(strip=True)
                return teams
            
            # Fallback: extract from title or mainBox
            title = soup.title.get_text() if soup.title else ""
            if ' vs ' in title:
                parts = title.split(' vs ')
                teams['home'] = parts[0].strip()
                teams['away'] = parts[1].split(' Prediction')[0].strip() if len(parts) > 1 else None
                return teams
            
            main_box = soup.find('div', class_='mainBox')
            if main_box:
                text = main_box.get_text()
                parts = re.split(r'\s+v\s+|\s+vs\.?\s+', text)
                if len(parts) >= 2:
                    teams['home'] = parts[0].strip()
                    teams['away'] = parts[1].strip()
                    return teams
            
            # Last resort: look in page text for "Team1 vs Team2" pattern
            page_text = soup.get_text()
            m = re.search(r'([A-Za-z][A-Za-z\s&]+?)\s+(?:vs\.?|vs\s+)\s+([A-Za-z][A-Za-z\s&]+?)\s+(?:Prediction|Stats)', page_text)
            if m:
                teams['home'] = m.group(1).strip()
                teams['away'] = m.group(2).strip()
                return teams
                
        except Exception as e:
            print(f"Error extracting teams: {e}")
        return teams

    def _extract_match_info(self, soup: BeautifulSoup, league_code: str = None, league_name: str = None) -> Dict:
        """Extract match date, time, league, venue."""
        info = {'date': None, 'time': None, 'league': league_name, 'league_code': league_code, 'venue': None}
        
        # Try to get league from page if not from URL
        if not info['league']:
            league_elem = soup.find('span', class_='league_name')
            if league_elem:
                info['league'] = league_elem.get_text(strip=True)
        try:
            date_elem = soup.find('span', class_='date')
            if date_elem:
                text = date_elem.get_text(strip=True)
                d = re.search(r'(\d{2}/\d{2}/\d{4})', text)
                t = re.search(r'(\d{2}:\d{2})', text)
                if d: info['date'] = d.group(1)
                if t: info['time'] = t.group(1)
        except Exception as e:
            print(f"Error extracting match info: {e}")
        return info

    def _extract_standings(self, soup: BeautifulSoup) -> Dict:
        standings = {'home': None, 'away': None, 'home_points': None, 'away_points': None}
        try:
            tc = soup.find('div', class_='teamtablesp_container')
            if tc:
                positions = re.findall(r'(\d+)(?:st|nd|rd|th)\s+place', tc.get_text())
                if len(positions) >= 2:
                    standings['home'] = int(positions[0])
                    standings['away'] = int(positions[1])
        except Exception as e:
            print(f"Error extracting standings: {e}")
        return standings

    def _extract_league_table(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract the full league standings table."""
        table: List[Dict] = []
        try:
            # Find cells with class std_pos (position column)
            pos_cells = soup.find_all('td', class_='std_pos')
            for pc in pos_cells:
                row_cells = []
                sibling = pc
                while sibling:
                    if hasattr(sibling, 'get_text'):
                        row_cells.append(sibling.get_text(strip=True))
                    sibling = sibling.find_next_sibling('td')
                    if sibling and 'std_pos' in (sibling.get('class') or []):
                        break
                if len(row_cells) >= 9:
                    entry = {
                        'position': int(row_cells[0]),
                        'team': row_cells[1],
                        'points': int(row_cells[2]),
                        'played': int(row_cells[3]),
                        'won': int(row_cells[4]),
                        'drawn': int(row_cells[5]),
                        'lost': int(row_cells[6]),
                        'gf': int(row_cells[7]),
                        'ga': int(row_cells[8]),
                        'gd': int(row_cells[9]) if len(row_cells) > 9 else None,
                    }
                    table.append(entry)
        except Exception as e:
            print(f"Error extracting league table: {e}")
        return table

    def _extract_form(self, soup: BeautifulSoup) -> Dict:
        form: Dict = {'home': [], 'away': [], 'home_summary': {}, 'away_summary': {}}
        try:
            containers = soup.find_all('div', class_='prformcont')
            if len(containers) >= 2:
                form['home'] = list(containers[0].get_text(strip=True))
                form['away'] = list(containers[1].get_text(strip=True))

            # Extract W/D/L percentages from the page text
            # Pattern: "Win N\nXX%\nDraw N\nXX%\nLost N\nXX%"
            main_table = soup.find('table', class_='main')
            if main_table:
                full_text = main_table.get_text()
                # Find all Win/Draw/Lost blocks
                wdl_blocks = re.findall(
                    r'Win\s+(\d+)\s+(\d+)%\s*Draw\s+(\d+)\s+(\d+)%\s*Lost\s+(\d+)\s+(\d+)%',
                    full_text
                )
                if len(wdl_blocks) >= 1:
                    form['home_summary'] = {
                        'wins': int(wdl_blocks[0][0]),
                        'win_pct': int(wdl_blocks[0][1]),
                        'draws': int(wdl_blocks[0][2]),
                        'draw_pct': int(wdl_blocks[0][3]),
                        'losses': int(wdl_blocks[0][4]),
                        'loss_pct': int(wdl_blocks[0][5]),
                    }
                if len(wdl_blocks) >= 2:
                    form['away_summary'] = {
                        'wins': int(wdl_blocks[1][0]),
                        'win_pct': int(wdl_blocks[1][1]),
                        'draws': int(wdl_blocks[1][2]),
                        'draw_pct': int(wdl_blocks[1][3]),
                        'losses': int(wdl_blocks[1][4]),
                        'loss_pct': int(wdl_blocks[1][5]),
                    }
        except Exception as e:
            print(f"Error extracting form: {e}")
        return form

    def _parse_match_rows(self, text: str) -> List[Dict]:
        """Parse match rows from a text block containing date/score/team patterns."""
        matches = []
        # Pattern: DD/MM YYYY  Team1 N - N (N - N) Team2 Competition
        pattern = re.compile(
            r'(\d{2}/\d{2})\s*(\d{4})\s+'
            r'(.+?)\s*(\d+)\s*-\s*(\d+)\s*'
            r'\((\d+)\s*-\s*(\d+)\)\s*'
            r'(.+?)(?:\s+(It1|It[A-Z]|ItC|UCL|UEL|UECL|EPL|LaL|BuL|Li1|ErD|LiP|[A-Z][a-z]+\d?))?$',
            re.MULTILINE
        )
        for m in pattern.finditer(text):
            matches.append({
                'date': f"{m.group(1)}/{m.group(2)}",
                'home_team': m.group(3).strip(),
                'home_score': int(m.group(4)),
                'away_score': int(m.group(5)),
                'ht_home': int(m.group(6)),
                'ht_away': int(m.group(7)),
                'away_team': m.group(8).strip(),
                'competition': m.group(9) if m.group(9) else '',
            })
        return matches

    def _extract_last_6_matches(self, soup: BeautifulSoup) -> Dict:
        """Extract last 6 matches for both teams."""
        data: Dict = {'home': [], 'away': []}
        try:
            main_table = soup.find('table', class_='main')
            if not main_table:
                return data
            full_text = main_table.get_text('\n')

            # Split around "Last 6 matches" and "home matches"
            l6_match = re.search(r'Last\s+6\s+matches', full_text)
            hm_match = re.search(r'home\s+matches', full_text)

            if l6_match:
                start = l6_match.end()
                end = hm_match.start() if hm_match else start + 3000
                l6_text = full_text[start:end]

                # Split into home and away sections using "View all" as separator
                parts = re.split(r'View\s+all', l6_text)
                if len(parts) >= 1:
                    data['home'] = self._parse_match_rows(parts[0])
                if len(parts) >= 2:
                    data['away'] = self._parse_match_rows(parts[1])
        except Exception as e:
            print(f"Error extracting last 6 matches: {e}")
        return data

    def _extract_home_away_matches(self, soup: BeautifulSoup, which: str) -> List[Dict]:
        """Extract home matches (for home team) or away matches (for away team)."""
        matches: List[Dict] = []
        try:
            main_table = soup.find('table', class_='main')
            if not main_table:
                return matches
            full_text = main_table.get_text('\n')

            if which == 'home':
                marker = re.search(r'home\s+matches', full_text)
                end_marker = re.search(r'away\s+matches', full_text)
            else:
                marker = re.search(r'away\s+matches', full_text)
                end_marker = re.search(r'Overall|Home/Away|Full\s+time', full_text)

            if marker:
                start = marker.end()
                end = end_marker.start() if end_marker else start + 3000
                section = full_text[start:end]
                parts = re.split(r'View\s+all', section)
                if parts:
                    matches = self._parse_match_rows(parts[0])
        except Exception as e:
            print(f"Error extracting {which} matches: {e}")
        return matches

    def _extract_head_to_head(self, soup: BeautifulSoup) -> Dict:
        """Extract head-to-head record."""
        h2h: Dict = {'matches': [], 'summary': {}}
        try:
            # Find the H2H section
            main_table = soup.find('table', class_='main')
            if not main_table:
                return h2h
            full_text = main_table.get_text('\n')

            h2h_match = re.search(r'Head\s+to\s+head', full_text)
            l6_match = re.search(r'Last\s+6\s+matches', full_text)

            if h2h_match:
                start = h2h_match.end()
                end = l6_match.start() if l6_match else start + 3000
                h2h_text = full_text[start:end]
                h2h['matches'] = self._parse_match_rows(h2h_text)

                # Summary: "TeamA N\nXX%\nDraw N\nXX%\nTeamB N\nXX%"
                wdl = re.findall(
                    r'(\w[\w\s]*?)\s+(\d+)\s+(\d+)%\s*Draw\s+(\d+)\s+(\d+)%\s*(\w[\w\s]*?)\s+(\d+)\s+(\d+)%',
                    h2h_text
                )
                if wdl:
                    h2h['summary'] = {
                        'home_wins': int(wdl[0][1]),
                        'home_win_pct': int(wdl[0][2]),
                        'draws': int(wdl[0][3]),
                        'draw_pct': int(wdl[0][4]),
                        'away_wins': int(wdl[0][6]),
                        'away_win_pct': int(wdl[0][7]),
                    }
        except Exception as e:
            print(f"Error extracting H2H: {e}")
        return h2h

    def _extract_odds(self, soup: BeautifulSoup) -> Dict:
        odds: Dict = {'1': None, 'X': None, '2': None, 'over': None, 'under': None}
        try:
            elems = soup.find_all(class_='haodd')
            for elem in elems:
                # Get text from each span individually (haodd contains multiple spans)
                spans = elem.find_all('span')
                nums = []
                for span in spans:
                    txt = span.get_text(strip=True)
                    if txt.replace('.', '').isdigit():
                        nums.append(float(txt))
                if len(nums) >= 3:
                    odds['1'], odds['X'], odds['2'] = nums[0], nums[1], nums[2]
                elif len(nums) == 2:
                    odds['over'], odds['under'] = nums[0], nums[1]
                break

            # Fallback
            if odds['1'] is None:
                for elem in soup.find_all(class_='odd'):
                    txt = elem.get_text(strip=True)
                    m = re.search(r'([\d.]+)', txt)
                    if m and odds['1'] is None:
                        odds['1'] = float(m.group(1))
                    elif odds['X'] is None:
                        odds['X'] = float(m.group(1))
                    elif odds['2'] is None:
                        odds['2'] = float(m.group(1))
                        break
        except Exception as e:
            print(f"Error extracting odds: {e}")
        return odds

    def _extract_predictions(self, soup: BeautifulSoup) -> Dict:
        preds: Dict = {'result': None, 'probability': None, 'over_under': None}
        try:
            # Forebet's predicted outcome
            for elem in soup.find_all(['span', 'div'], class_=re.compile(r'pred|probability')):
                txt = elem.get_text(strip=True)
                if txt in ['1', 'X', '2']:
                    preds['result'] = txt
                prob = re.search(r'(\d+)%', txt)
                if prob:
                    preds['probability'] = int(prob.group(1))
        except Exception as e:
            print(f"Error extracting predictions: {e}")
        return preds

    def _extract_goals_stats(self, soup: BeautifulSoup) -> Dict:
        gs: Dict = {'home': {}, 'away': {}}
        try:
            # Use JavaScript data which has scr (scored), cnd (conceded), and pl (played)
            js = self._extract_javascript_data(soup)
            
            for side, key in [('home', 'home_stats'), ('away', 'away_stats')]:
                if key in js:
                    d = js[key]
                    scored = d.get('scr', [0])[0] if isinstance(d.get('scr'), list) else d.get('scr', 0)
                    conceded = d.get('cnd', [0])[0] if isinstance(d.get('cnd'), list) else d.get('cnd', 0)
                    games = d.get('pl', [28 if side == 'home' else 36, 0, 0])[0] if isinstance(d.get('pl'), list) else (28 if side == 'home' else 36)
                    
                    gs[side]['scored'] = scored
                    gs[side]['conceded'] = conceded
                    gs[side]['scored_avg'] = scored / games if games else None
                    gs[side]['conceded_avg'] = conceded / games if games else None
                    gs[side]['games'] = games
                    
        except Exception as e:
            print(f"Error extracting goals stats: {e}")
        return gs

    def _extract_over_under_stats(self, soup: BeautifulSoup) -> Dict:
        ou: Dict = {'home': {}, 'away': {}}
        try:
            txt = soup.get_text()
            # O/U patterns
            for side, pattern in [('home', r'Over\s*2\.5.*?(\d+)%\s*Under\s*2\.5.*?(\d+)%'),
                                  ('away', r'Under\s*2\.5.*?(\d+)%\s*Over\s*2\.5.*?(\d+)%')]:
                m = re.search(pattern, txt, re.IGNORECASE | re.DOTALL)
                if m:
                    ou[side]['over_pct'] = int(m.group(2)) if side == 'home' else int(m.group(2))
                    ou[side]['under_pct'] = int(m.group(1)) if side == 'home' else int(m.group(1))
        except Exception as e:
            print(f"Error extracting O/U stats: {e}")
        return ou

    def _extract_bts_stats(self, soup: BeautifulSoup) -> Dict:
        bts: Dict = {'home': {}, 'away': {}}
        try:
            txt = soup.get_text()
            # Both to score patterns
            for side, pattern in [('home', r'Both.*?score.*?Yes.*?(\d+)%.*?No.*?(\d+)%'),
                                  ('away', r'Both.*?score.*?No.*?(\d+)%.*?Yes.*?(\d+)%')]:
                m = re.search(pattern, txt, re.IGNORECASE | re.DOTALL)
                if m:
                    bts[side]['yes_pct'] = int(m.group(2)) if side == 'home' else int(m.group(2))
                    bts[side]['no_pct'] = int(m.group(1)) if side == 'home' else int(m.group(1))
        except Exception as e:
            print(f"Error extracting BTS stats: {e}")
        return bts

    def _extract_shots_stats(self, soup: BeautifulSoup) -> Dict:
        shots: Dict = {'home': {}, 'away': {}}
        try:
            js = self._extract_javascript_data(soup)
            for side, key in [('home', 'home_stats'), ('away', 'away_stats')]:
                if key in js:
                    d = js[key]
                    shots[side] = {
                        'total': d.get('shots_total'),
                        'on_target': d.get('shots_on_target'),
                        'off_target': d.get('shots_off_target'),
                        'inside_box': d.get('shots_insidebox'),
                        'outside_box': d.get('shots_outsidebox'),
                        'blocked': d.get('shots_blocked'),
                    }
        except Exception as e:
            print(f"Error extracting shots: {e}")
        return shots

    def _extract_passes_stats(self, soup: BeautifulSoup) -> Dict:
        passes: Dict = {'home': {}, 'away': {}}
        try:
            js = self._extract_javascript_data(soup)
            for side, key in [('home', 'home_stats'), ('away', 'away_stats')]:
                if key in js:
                    d = js[key]
                    passes[side] = {
                        'total': d.get('passes_total'),
                        'accurate': d.get('passes_accurate'),
                        'ball_poss': d.get('ball_poss'),
                    }
        except Exception as e:
            print(f"Error extracting passes: {e}")
        return passes

    def _extract_attacks_stats(self, soup: BeautifulSoup) -> Dict:
        attacks: Dict = {'home': {}, 'away': {}}
        try:
            js = self._extract_javascript_data(soup)
            for side, key in [('home', 'home_stats'), ('away', 'away_stats')]:
                if key in js:
                    d = js[key]
                    attacks[side] = {
                        'total': d.get('attacks'),
                        'dangerous': d.get('dan_attacks'),
                    }
        except Exception as e:
            print(f"Error extracting attacks: {e}")
        return attacks

    def _extract_discipline(self, soup: BeautifulSoup) -> Dict:
        disc: Dict = {'home': {}, 'away': {}}
        try:
            js = self._extract_javascript_data(soup)
            for side, key in [('home', 'home_stats'), ('away', 'away_stats')]:
                if key in js:
                    d = js[key]
                    disc[side] = {
                        'fouls': d.get('fouls'),
                        'yellow_cards': d.get('yellowcards'),
                        'red_cards': d.get('redcards'),
                    }
        except Exception as e:
            print(f"Error extracting discipline: {e}")
        return disc

    def _extract_other_stats(self, soup: BeautifulSoup) -> Dict:
        other: Dict = {'home': {}, 'away': {}}
        try:
            js = self._extract_javascript_data(soup)
            for side, key in [('home', 'home_stats'), ('away', 'away_stats')]:
                if key in js:
                    d = js[key]
                    other[side] = {
                        'corners': d.get('total_corners'),
                        'throw_ins': d.get('throw_in'),
                        'goal_kicks': d.get('goal_kick'),
                        'offsides': d.get('offsides'),
                        'saves': d.get('saves'),
                    }
        except Exception as e:
            print(f"Error extracting other stats: {e}")
        return other

    def _extract_trends(self, soup: BeautifulSoup) -> List[Dict]:
        trends: List[Dict] = []
        try:
            trend_elems = soup.find_all('div', class_=re.compile(r'trend|statbox'))
            for elem in trend_elems:
                text = elem.get_text(strip=True)
                m = re.search(r'([A-Za-z\s]+)\s*(\d+)/(\d+)\s*=\s*(\d+)%', text)
                if m:
                    trends.append({
                        'description': m.group(1).strip(),
                        'count': int(m.group(2)),
                        'total': int(m.group(3)),
                        'percentage': int(m.group(4)),
                    })
        except Exception as e:
            print(f"Error extracting trends: {e}")
        return trends

    def _extract_match_intro(self, soup: BeautifulSoup) -> str:
        intro = ''
        try:
            intro_elem = soup.find('p', class_='match_intro')
            if intro_elem:
                intro = intro_elem.get_text(strip=True)
            else:
                # Try any paragraph containing match preview
                for p in soup.find_all('p'):
                    txt = p.get_text(strip=True)
                    if len(txt) > 50 and any(kw in txt for kw in ['will face', 'host', 'upcoming', 'preview']):
                        intro = txt
                        break
        except Exception as e:
            print(f"Error extracting match intro: {e}")
        return intro

    def _extract_injuries(self, soup: BeautifulSoup) -> Dict:
        inj: Dict = {'home': [], 'away': []}
        try:
            # Look for injury/suspension indicators
            for elem in soup.find_all(['span', 'div'], class_=re.compile(r'injury|suspension|absent')):
                text = elem.get_text(strip=True)
                # Try to identify team
                if any(name in text for name in ['out', 'injured', 'suspended']):
                    inj['home'].append(text) if 'home' not in inj else inj['away'].append(text)
        except Exception as e:
            print(f"Error extracting injuries: {e}")
        return inj

    def scrape_team_injuries(self, team_name: str) -> List[Dict]:
        """Scrape injured players for a specific team from Forebet injured players page.
        
        The injuries page has one main table with team sections. Structure:
        - Team name header
        - Headers: Player, Games played, Injury, Status for a date
        - Player rows with injury data
        
        Args:
            team_name: Name of the team to search for
            
        Returns:
            List of dictionaries containing injured player information
        """
        injuries: List[Dict] = []
        try:
            url = "https://www.forebet.com/en/injured-players"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get all team names from h4 tags
            team_headers = soup.find_all('h4')
            team_names = [h4.get_text(strip=True) for h4 in team_headers]
            
            # Find our team in the list
            team_idx = None
            for i, name in enumerate(team_names):
                if team_name.lower() in name.lower():
                    team_idx = i
                    break
            
            if team_idx is None:
                return injuries
            
            # Find the table that contains team data
            # The table structure: team name in first th, then player rows
            main_table = soup.find('table', class_='main')
            
            if not main_table:
                # Try any table
                tables = soup.find_all('table')
                for table in tables:
                    if 'injury' in table.get_text().lower():
                        main_table = table
                        break
            
            if main_table:
                rows = main_table.find_all('tr')
                
                # Track which team we're currently in
                current_team = None
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if not cells:
                        continue
                    
                    first_cell = cells[0].get_text(strip=True)
                    
                    # Check if this row is a team header
                    is_team_header = False
                    for h4 in team_headers:
                        if h4.get_text(strip=True) == first_cell:
                            current_team = first_cell
                            is_team_header = True
                            break
                    
                    if is_team_header:
                        # Check if this is our team
                        if team_name.lower() not in first_cell.lower():
                            # We moved to another team, stop processing
                            if current_team and team_name.lower() in current_team.lower():
                                # Found our team earlier, now we left it
                                break
                        else:
                            # This is our team, reset and continue
                            current_team = first_cell
                        continue
                    
                    # If we're in our team section, extract player data
                    if current_team and team_name.lower() in current_team.lower():
                        # Check if this is a player row (has player name with position)
                        if len(cells) >= 2 and '(' in first_cell and ')' in first_cell:
                            games = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                            injury = cells[2].get_text(strip=True) if len(cells) > 2 else "Unknown"
                            status = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                            
                            injuries.append({
                                'name': first_cell,
                                'games_played': games,
                                'injury': injury,
                                'status': status,
                                'team': current_team
                            })
            
            # Remove duplicates
            seen = set()
            unique_injuries = []
            for inj in injuries:
                key = inj['name'].lower()
                if key not in seen:
                    seen.add(key)
                    unique_injuries.append(inj)
            injuries = unique_injuries
            
        except Exception as e:
            print(f"Error scraping injuries for {team_name}: {e}")
            import traceback
            traceback.print_exc()
        
        return injuries

    def _extract_next_matches(self, soup: BeautifulSoup) -> Dict:
        nxt: Dict = {'home': [], 'away': []}
        try:
            main_table = soup.find('table', class_='main')
            if not main_table:
                return nxt
            txt = main_table.get_text('\n')
            nm = re.search(r'next\s+matches', txt)
            if nm:
                section = txt[nm.end():]
                parts = re.split(r'View\s+all', section)
                for i, part in enumerate(parts[:2]):
                    side = 'home' if i == 0 else 'away'
                    lines = re.findall(r'(\d{2}/\d{2})\s+(\d{4})\s+\w+\s+(.+?)(?:\d+\s*$|\n)', part, re.MULTILINE)
                    for line in lines:
                        nxt[side].append({
                            'date': f"{line[0]}/{line[1]}",
                            'match': line[2].strip(),
                        })
        except Exception as e:
            print(f"Error extracting next matches: {e}")
        return nxt

    def _extract_javascript_data(self, soup: BeautifulSoup) -> Dict:
        """Extract detailed statistics from the get_ovd JavaScript function."""
        js_data: Dict = {}
        try:
            if hasattr(self, '_js_cache'):
                return self._js_cache

            # First, try to extract from JavaScript (this is the primary source)
            scripts = soup.find_all('script')
            for script in scripts:
                if script.string and 'get_ovd' in script.string:
                    # Extract all return statements with data objects
                    returns = re.findall(r'return\s+(\{.+?\})\s*;', script.string, re.DOTALL)
                    for i, ret in enumerate(returns):
                        try:
                            cleaned = ret.replace("'", '"').replace('\n', ' ')
                            data = json.loads(cleaned)
                            key = 'home_stats' if i == 0 else 'away_stats'
                            if 'all' in data and 'ft' in data['all']:
                                js_data[key] = data['all']['ft']
                            if 'all' in data and 'ht1' in data['all']:
                                js_data[key + '_ht'] = data['all']['ht1']
                        except (json.JSONDecodeError, KeyError):
                            pass
                    break  # Found the main get_ovd function, no need to check other scripts
            
            # If JS extraction failed or incomplete, use HTML extraction as fallback
            if not js_data or len(js_data) < 2:
                js_data = self._extract_stats_from_html(soup)
            
            self._js_cache = js_data
        except Exception as e:
            print(f"Error extracting JS data: {e}")
            # Fallback to HTML extraction
            js_data = self._extract_stats_from_html(soup)
        return js_data

    def _extract_stats_from_html(self, soup: BeautifulSoup) -> Dict:
        """Extract statistics directly from HTML when JavaScript parsing fails."""
        stats: Dict = {'home_stats': {}, 'away_stats': {}}
        try:
            # Find all stat containers
            stat_containers = soup.find_all(['div', 'td', 'span'], 
                class_=re.compile(r'stat|stats|value|data', re.I))
            
            # Extract shots data
            for container in stat_containers:
                text = container.get_text(' ', strip=True)
                # Look for numeric patterns with labels
                total_shots = re.search(r'Total\s+shots?\s+(\d+)', text)
                if total_shots:
                    if 'home_stats' not in stats or 'shots_total' not in stats['home_stats']:
                        stats['home_stats']['shots_total'] = int(total_shots.group(1))
                    else:
                        stats['away_stats']['shots_total'] = int(total_shots.group(1))
            
            # Extract from tables with class 'stat'
            stat_tables = soup.find_all('table', class_='stat')
            for table in stat_tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        home_val = cells[1].get_text(strip=True)
                        away_val = cells[2].get_text(strip=True) if len(cells) > 2 else None
                        
                        # Map common stat names
                        stat_map = {
                            'shots': 'shots_total',
                            'on target': 'shots_on_target',
                            'off target': 'shots_off_target',
                            'blocked': 'shots_blocked',
                            'inside box': 'shots_insidebox',
                            'outside box': 'shots_outsidebox',
                            'passes': 'passes_total',
                            'accurate passes': 'passes_accurate',
                            'possession': 'ball_poss',
                            'attacks': 'attacks',
                            'dangerous attacks': 'dan_attacks',
                            'fouls': 'fouls',
                            'yellow cards': 'yellowcards',
                            'red cards': 'redcards',
                        }
                        
                        for pattern, key in stat_map.items():
                            if pattern in label:
                                try:
                                    val = re.search(r'([\d.]+)', home_val)
                                    if val:
                                        stats['home_stats'][key] = float(val.group(1))
                                    if away_val:
                                        val = re.search(r'([\d.]+)', away_val)
                                        if val:
                                            stats['away_stats'][key] = float(val.group(1))
                                except ValueError:
                                    pass
        except Exception as e:
            print(f"Error extracting stats from HTML: {e}")
        
        return stats


if __name__ == "__main__":
    scraper = ForebetScraper()
    test_url = "https://www.forebet.com/en/football/matches/juventus-lazio-2344437"
    print("Testing Football Scraper...")
    print("=" * 60)
    data = scraper.scrape_match(test_url)
    if data:
        print("\nExtracted Data:")
        print(json.dumps(data, indent=2, default=str))
    else:
        print("Failed to scrape data")
