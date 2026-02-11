#!/usr/bin/env python3
"""
Daily Training Script for Football Prediction Model
Automates daily model training by:
1. Scraping yesterday's match results from Forebet
2. Checking for new leagues and updating database
3. Retraining the model with all available data
4. Showing training statistics

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
MATCHES_FILE = "data/matches.json"
LEAGUES_DB_FILE = "data/leagues_db.json"


class DailyTrainer:
    """Handles daily model training with scraping and league updates."""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.matches_file = MATCHES_FILE
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
        Scrape yesterday's matches from Forebet.
        Returns dict with scraped_count, new_leagues_found, errors.
        """
        from football_scraper import FootballScraper
        from data_storage import MatchDataStorage
        
        scraper = FootballScraper()
        storage = MatchDataStorage()
        
        yesterday = self.get_yesterday_date()
        print(f"\n📅 Scraping matches for {yesterday}...")
        
        # Scrape historical matches for yesterday
        try:
            historical_url = f"https://www.forebet.com/en/football-matches-predictions/{yesterday}/results"
            print(f"   URL: {historical_url}")
            
            matches = scraper.scrape_historical_matches(historical_url)
            
            if not matches:
                print(f"   ⚠ No matches found for {yesterday}")
                return {'scraped_count': 0, 'new_leagues': 0, 'errors': []}
            
            print(f"   ✓ Found {len(matches)} matches")
            
            # Save matches and extract results
            saved_count = 0
            new_leagues = set()
            
            for match in matches:
                league = match.get('league', 'Unknown')
                url = match.get('url', '')
                home_score = match.get('home_score')
                away_score = match.get('away_score')
                
                # Check if this is a new league
                if league not in storage.get_all_leagues():
                    new_leagues.add(league)
                
                # Save match with result
                if home_score is not None and away_score is not None:
                    ok = storage.save_match_with_result(match)
                    if ok:
                        saved_count += 1
            
            # Update leagues database if new leagues found
            if new_leagues:
                print(f"   🆕 New leagues found: {len(new_leagues)}")
                for league in new_leagues:
                    print(f"      - {league}")
                self.update_league_database(list(new_leagues))
            
            self.save_scrape_date(yesterday)
            
            return {
                'scraped_count': saved_count,
                'total_found': len(matches),
                'new_leagues': len(new_leagues),
                'errors': []
            }
            
        except Exception as e:
            print(f"   ❌ Error scraping: {e}")
            return {'scraped_count': 0, 'total_found': 0, 'new_leagues': 0, 'errors': [str(e)]}
    
    def update_league_database(self, new_leagues: List[str]):
        """Update leagues database with new leagues."""
        leagues_db = {}
        
        if os.path.exists(self.leagues_db_file):
            try:
                with open(self.leagues_db_file, 'r') as f:
                    leagues_db = json.load(f)
            except Exception:
                leagues_db = {}
        
        for league in new_leagues:
            leagues_db[league] = {
                'name': league,
                'added_date': self.get_today_date(),
                'country': 'Unknown',
                'tier': 3,
                'active': True
            }
        
        with open(self.leagues_db_file, 'w') as f:
            json.dump(leagues_db, f, indent=2)
        
        print(f"   ✅ Updated league database with {len(new_leagues)} new leagues")
    
    def check_new_leagues(self) -> List[str]:
        """Check if there are new leagues in matches that need database updates."""
        if not os.path.exists(self.matches_file):
            return []
        
        with open(self.matches_file, 'r') as f:
            matches = json.load(f)
        
        leagues_db = {}
        if os.path.exists(self.leagues_db_file):
            with open(self.leagues_db_file, 'r') as f:
                leagues_db = json.load(f)
        
        new_leagues = []
        for match in matches:
            league = match.get('info', {}).get('league')
            if league and league not in leagues_db:
                new_leagues.append(league)
        
        return list(set(new_leagues))
    
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
    
    def get_training_data_stats(self) -> Dict:
        """Get statistics about the training data."""
        if not os.path.exists(MATCHES_FILE):
            return {'total_matches': 0, 'leagues': 0}
        
        with open(MATCHES_FILE, 'r') as f:
            matches = json.load(f)
        
        leagues = set()
        for match in matches:
            league = match.get('info', {}).get('league')
            if league:
                leagues.add(league)
        
        return {
            'total_matches': len(matches),
            'leagues': len(leagues)
        }
    
    def train_model(self) -> Dict:
        """
        Train the prediction model with all available data.
        """
        from prediction_model import FootballPredictor
        import pickle
        
        print("\n🔧 Training models...")
        
        predictor = FootballPredictor()
        
        # Load training data
        training_file = "data/training_data.pkl"
        if not os.path.exists(training_file):
            print("   ❌ No training data found!")
            return {'error': 'No training data'}
        
        with open(training_file, 'rb') as f:
            training_data = pickle.load(f)
        
        # Combine all examples for global model
        all_examples = []
        for entry in training_data:
            examples = entry.get('examples', [])
            all_examples.extend(examples)
        
        print(f"   Training on {len(all_examples)} examples...")
        
        # Train global model
        global_result = predictor.train(all_examples)
        
        if 'error' in global_result:
            print(f"   ⚠ Global model training: {global_result['error']}")
        else:
            print(f"   ✅ Global model trained ({global_result.get('training_examples', 0)} examples)")
        
        # Train league-specific models
        league_results = {}
        leagues_trained = 0
        
        for entry in training_data:
            league = entry.get('league')
            examples = entry.get('examples', [])
            
            if len(examples) >= 10:  # Minimum for league-specific model
                result = predictor.train(training_data, league=league)
                if 'error' not in result:
                    leagues_trained += 1
                    league_results[league] = {
                        'examples': len(examples),
                        'accuracy': result.get('result_accuracy', 0)
                    }
        
        print(f"   ✅ Trained {leagues_trained} league-specific models")
        
        # Save training timestamp
        self.save_training_time()
        
        return {
            'global_model': global_result,
            'league_models': league_results,
            'total_examples': len(all_examples),
            'leagues_trained': leagues_trained
        }
    
    # ==================== STATS ====================
    
    def show_stats(self):
        """Show current training statistics."""
        from model_stats import get_training_stats
        
        stats = get_training_stats()
        
        print("\n" + "=" * 60)
        print("📊 FOOTBALL PREDICTION MODEL - DAILY STATUS")
        print("=" * 60)
        
        print(f"\n🗄️  Training Data:")
        print(f"   Total examples: {stats.get('training_count', 0)}")
        
        print(f"\n📈 Latest Training:")
        if stats.get('last_training'):
            last = datetime.fromisoformat(stats['last_training'])
            print(f"   Last training: {last.strftime('%Y-%m-%d %H:%M')}")
        
        result_acc = stats.get('result_accuracy', 0)
        ou_acc = stats.get('ou_accuracy', 0)
        
        print(f"\n   Match Result Accuracy: {result_acc:.1%}")
        print(f"   Over/Under Accuracy:   {ou_acc:.1%}")
        
        print(f"\n📜 Training History: {stats.get('total_trainings', 0)} trainings")
        
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
            print(f"   New leagues: {scrape_result.get('new_leagues', 0)}")
        
        # Step 2: Check for new leagues
        new_leagues = self.check_new_leagues()
        if new_leagues and not train_only:
            print(f"\n🆕 New leagues in matches: {len(new_leagues)}")
            self.update_league_database(new_leagues)
        
        # Step 3: Train model
        if not scrape_only:
            train_result = self.train_model()
            results['train'] = train_result
            
            if 'error' not in train_result.get('global_model', {}):
                print(f"\n✅ Training completed:")
                print(f"   Examples: {train_result.get('total_examples', 0)}")
                print(f"   League models: {train_result.get('leagues_trained', 0)}")
        
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
