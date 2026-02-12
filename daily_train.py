#!/usr/bin/env python3
"""
Daily Training Script for Football Prediction Model
Automates daily model training by:
1. Scraping yesterday's match results from Forebet
2. Saving to JSON format
3. Retraining models using rebuild_data.py
4. Auto-discovers new leagues from JSON data

Usage:
    python daily_train.py              # Run full daily training
    python daily_train.py --scrape-only # Only scrape, don't train
    python daily_train.py --train-only # Only train, don't scrape
    python daily_train.py --status    # Show status without running

Cron setup (run at 8 AM daily):
    0 8 * * * cd /path/to/game && ./train.sh >> /var/log/football_training.log 2>&1
"""

import os
import sys
import json
import glob
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Constants
DATA_DIR = "data"
RESULTS_FILE = "results.txt"
LAST_TRAIN_FILE = "data/last_training.json"
LAST_SCRAPE_FILE = "data/last_scrape.json"
LEAGUES_DB_FILE = "data/leagues_db.json"


class DailyTrainer:
    """Handles daily model training with scraping and league auto-discovery."""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.leagues_db_file = LEAGUES_DB_FILE
        
    def get_yesterday_date(self) -> str:
        """Get yesterday's date in YYYY-MM-DD format."""
        yesterday = datetime.now() - timedelta(days=1)
        return yesterday.strftime('%Y-%m-%d')
    
    def get_today_date(self) -> str:
        """Get today's date in YYYY-MM-DD format."""
        return datetime.now().strftime('%Y-%m-%d')
    
    # ==================== SCRAPING ====================
    
    def get_last_scrape_date(self) -> Optional[str]:
        """Get the date of the last scrape."""
        if os.path.exists(LAST_SCRAPE_FILE):
            try:
                with open(LAST_SCRAPE_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('date')
            except Exception:
                return None
        return None
    
    def save_scrape_date(self, date: str):
        """Save the current scrape date."""
        with open(LAST_SCRAPE_FILE, 'w') as f:
            json.dump({'date': date, 'timestamp': datetime.now().isoformat()}, f)
    
    def scrape_yesterday_matches(self) -> Dict:
        """
        Scrape yesterday's matches from Forebet and save to JSON.
        Returns dict with scraped_count, new_leagues_found, errors.
        """
        from scrape_historical import HistoricalForebetScraper
        
        scraper = HistoricalForebetScraper()
        
        yesterday = self.get_yesterday_date()
        print(f"\n📅 Scraping matches for {yesterday}...")
        
        # Scrape historical matches for yesterday
        try:
            matches = scraper.scrape_historical_matches(yesterday)
            
            if not matches:
                print(f"   ⚠ No matches found for {yesterday}")
                return {'scraped_count': 0, 'new_leagues': 0, 'errors': []}
            
            print(f"   ✓ Found {len(matches)} matches")
            
            # Save to JSON file
            json_file = f"data/historical_matches_{yesterday}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(matches, f, indent=2, default=str)
            
            print(f"   ✓ Saved to {json_file}")
            
            # Count matches with results
            with_results = sum(1 for m in matches if m.get('has_result'))
            
            # Auto-discover leagues from JSON
            leagues_seen = set()
            for m in matches:
                league_code = m.get('league_code', '')
                match_id = m.get('match_id', '')
                league_key = f"{league_code}_{match_id[:3]}" if len(match_id) >= 3 else league_code
                leagues_seen.add(league_key)
            
            self.save_scrape_date(yesterday)
            
            return {
                'scraped_count': len(matches),
                'with_results': with_results,
                'new_leagues': len(leagues_seen),
                'json_file': json_file,
                'errors': []
            }
            
        except Exception as e:
            print(f"   ❌ Error scraping: {e}")
            return {'scraped_count': 0, 'with_results': 0, 'new_leagues': 0, 'errors': [str(e)]}
    
    # ==================== LEAGUE AUTO-DISCOVERY ====================
    
    def check_new_leagues(self) -> List[str]:
        """
        Auto-discover leagues from JSON files.
        Returns list of unique league keys found.
        """
        json_files = glob.glob(f"{self.data_dir}/historical_matches_*.json")
        
        if not json_files:
            return []
        
        leagues_db = set()
        if os.path.exists(self.leagues_db_file):
            try:
                with open(self.leagues_db_file, 'r', encoding='utf-8') as f:
                    leagues_data = json.load(f)
                    leagues_db = set(leagues_data.keys())
            except:
                pass
        
        new_leagues = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                    for match in matches:
                        league_code = match.get('league_code', '')
                        match_id = match.get('match_id', '')
                        league_key = f"{league_code}_{match_id[:3]}" if len(match_id) >= 3 else league_code
                        if league_key and league_key not in leagues_db:
                            new_leagues.append(league_key)
            except:
                pass
        
        return list(set(new_leagues))
    
    def update_league_database(self):
        """
        Update leagues database by auto-discovering from all JSON files.
        Uses comprehensive league info from historical_matches JSON.
        """
        leagues_db = {}
        
        if os.path.exists(self.leagues_db_file):
            try:
                with open(self.leagues_db_file, 'r', encoding='utf-8') as f:
                    leagues_db = json.load(f)
            except:
                pass
        
        # Scan all JSON files for league info
        json_files = glob.glob(f"{self.data_dir}/historical_matches_*.json")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                    for match in matches:
                        league_code = match.get('league_code', '')
                        match_id = match.get('match_id', '')
                        league_key = f"{league_code}_{match_id[:3]}" if len(match_id) >= 3 else league_code
                        
                        if league_key not in leagues_db:
                            leagues_db[league_key] = {
                                'league_code': league_code,
                                'country': match.get('country', 'Unknown'),
                                'league': match.get('league', 'Unknown'),
                                'league_url_path': match.get('league_url_path', ''),
                                'country_code': match.get('country_code', ''),
                                'added_date': self.get_today_date(),
                                'match_count': 0
                            }
                        
                        leagues_db[league_key]['match_count'] += 1
            except Exception as e:
                print(f"   ⚠ Error reading {json_file}: {e}")
        
        with open(self.leagues_db_file, 'w', encoding='utf-8') as f:
            json.dump(leagues_db, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ League database updated: {len(leagues_db)} leagues")
        return leagues_db
    
    # ==================== TRAINING ====================
    
    def get_last_training_time(self) -> Optional[datetime]:
        """Get the timestamp of the last training."""
        if os.path.exists(LAST_TRAIN_FILE):
            try:
                with open(LAST_TRAIN_FILE, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data.get('timestamp', ''))
            except Exception:
                return None
        return None
    
    def save_training_time(self):
        """Save the current timestamp as the last training time."""
        with open(LAST_TRAIN_FILE, 'w') as f:
            json.dump({'timestamp': datetime.now().isoformat()}, f)
    
    def should_train(self) -> bool:
        """Check if enough time has passed since last training."""
        last_train = self.get_last_training_time()
        if last_train is None:
            return True
        
        time_since = datetime.now() - last_train
        return time_since.total_seconds() > (24 * 3600)  # 24 hours
    
    def train_model(self) -> Dict:
        """
        Train the prediction model using rebuild_data.py.
        Reads from historical_matches JSON files.
        """
        import subprocess
        
        print("\n🔧 Training models using JSON data...")
        
        # First update league database
        print("\n📋 Auto-discovering leagues from JSON files...")
        self.update_league_database()
        
        try:
            # Run rebuild_data.py to train models
            result = subprocess.run(
                [sys.executable, 'rebuild_data.py'],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            if result.returncode != 0:
                print(f"   ❌ Training error: {result.stderr}")
                return {'error': result.stderr}
            
            # Parse output for stats
            output = result.stdout
            print(output)
            
            # Save training timestamp
            self.save_training_time()
            
            return {
                'success': True,
                'message': 'Training completed successfully'
            }
        except Exception as e:
            print(f"   ❌ Training error: {e}")
            return {'error': str(e)}
    
    # ==================== STATS ====================
    
    def show_stats(self):
        """Show current training statistics."""
        print("\n" + "=" * 60)
        print("📊 FOOTBALL PREDICTION MODEL - DAILY STATUS")
        print("=" * 60)
        
        # Count JSON files and matches
        json_files = glob.glob(f"{self.data_dir}/historical_matches_*.json")
        total_matches = 0
        for jf in json_files:
            try:
                with open(jf, 'r', encoding='utf-8') as f:
                    matches = json.load(f)
                    total_matches += len(matches)
            except:
                pass
        
        print(f"\n🗄️  Data Files:")
        print(f"   JSON files: {len(json_files)}")
        print(f"   Total matches: {total_matches}")
        
        # League stats
        if os.path.exists(self.leagues_db_file):
            try:
                with open(self.leagues_db_file, 'r', encoding='utf-8') as f:
                    leagues_db = json.load(f)
                    print(f"   Leagues: {len(leagues_db)}")
            except:
                pass
        
        print(f"\n📈 Latest Training:")
        last_train = self.get_last_training_time()
        if last_train:
            print(f"   Last training: {last_train.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"   No training recorded")
        
        print("\n" + "=" * 60)
    
    # ==================== MAIN ====================
    
    def run(self, scrape_only: bool = False, train_only: bool = False) -> Dict:
        """
        Run the daily training workflow.
        
        Args:
            scrape_only: Only scrape, don't train
            train_only: Only train, don't scrape
        """
        today = self.get_today_date()
        yesterday = self.get_yesterday_date()
        
        print("\n" + "=" * 60)
        print("⚽ FOOTBALL PREDICTION MODEL - DAILY TRAINING")
        print("=" * 60)
        print(f"\n📅 Date: {today}")
        print(f"   Yesterday: {yesterday}")
        
        results = {
            'date': today,
            'timestamp': datetime.now().isoformat(),
            'scrape': None,
            'train': None
        }
        
        # Step 1: Scrape yesterday's matches
        if not train_only:
            scrape_result = self.scrape_yesterday_matches()
            results['scrape'] = scrape_result
            print(f"\n📥 Scrape Results:")
            print(f"   Matches saved: {scrape_result.get('scraped_count', 0)}")
            print(f"   With results: {scrape_result.get('with_results', 0)}")
        
        # Step 2: Auto-discover leagues from JSON
        if not train_only:
            print(f"\n🔍 Auto-discovering leagues...")
            new_leagues = self.check_new_leagues()
            if new_leagues:
                print(f"   Found {len(new_leagues)} leagues")
            self.update_league_database()
        
        # Step 3: Train model
        if not scrape_only:
            train_result = self.train_model()
            results['train'] = train_result
            
            if 'error' not in train_result:
                print(f"\n✅ Training completed successfully")
        
        # Step 4: Show stats
        self.show_stats()
        
        return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Daily training for football prediction model'
    )
    parser.add_argument(
        '--scrape-only', '-s',
        action='store_true',
        help='Only scrape yesterday results, do not train'
    )
    parser.add_argument(
        '--train-only', '-t',
        action='store_true',
        help='Only train models, do not scrape'
    )
    parser.add_argument(
        '--status', '-v',
        action='store_true',
        help='Show current status without running'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force training even if recently trained'
    )
    
    args = parser.parse_args()
    
    trainer = DailyTrainer()
    
    if args.status:
        trainer.show_stats()
        return
    
    # Check if we should skip training
    if not args.force and not args.scrape_only and not trainer.should_train():
        last_train = trainer.get_last_training_time()
        if last_train:
            time_since = datetime.now() - last_train
            hours_left = 24 - (time_since.total_seconds() / 3600)
            print(f"\n⏰ Training recently completed.")
            print(f"   Next training in: {max(0, hours_left):.1f} hours")
            trainer.show_stats()
            return
    
    result = trainer.run(
        scrape_only=args.scrape_only,
        train_only=args.train_only
    )
    
    print(f"\n✅ Daily training completed at {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
