#!/usr/bin/env python3
"""
Auto-Training Script for Football Prediction Model
Automatically trains the model using links from results.txt if more than 20 hours
have passed since the last training. This ensures the model learns from scraped
match data before making predictions.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import argparse

# Constants
RESULTS_FILE = "results.txt"
LAST_TRAIN_FILE = "data/last_training.json"
TRAINING_THRESHOLD_HOURS = 20
MATCH_DATA_DIR = "match_data"


class AutoTrainer:
    """Handles automatic model training based on time and data availability."""
    
    def __init__(self, results_file: str = RESULTS_FILE, 
                 last_train_file: str = LAST_TRAIN_FILE):
        self.results_file = results_file
        self.last_train_file = last_train_file
        self.matches_file = os.path.join(MATCH_DATA_DIR, "matches.json")
        
    def get_last_training_time(self) -> Optional[datetime]:
        """Get the timestamp of the last training."""
        if os.path.exists(self.last_train_file):
            try:
                with open(self.last_train_file, 'r') as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data.get('timestamp', ''))
            except Exception:
                return None
        return None
    
    def save_training_time(self):
        """Save the current timestamp as the last training time."""
        with open(self.last_train_file, 'w') as f:
            json.dump({'timestamp': datetime.now().isoformat()}, f)
    
    def should_train(self) -> bool:
        """Check if enough time has passed since last training."""
        last_train = self.get_last_training_time()
        if last_train is None:
            return True
        
        time_since = datetime.now() - last_train
        return time_since.total_seconds() > (TRAINING_THRESHOLD_HOURS * 3600)
    
    def get_pending_urls(self) -> List[str]:
        """Get URLs from results.txt that need to be processed."""
        urls = []
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        urls.append(line)
        return urls
    
    def get_saved_match_count(self) -> int:
        """Get the count of saved matches."""
        if os.path.exists(self.matches_file):
            with open(self.matches_file, 'r') as f:
                matches = json.load(f)
                return len(matches)
        return 0
    
    def train_model(self, urls: List[str] = None) -> Dict:
        """Train the model using the prediction system."""
        from football_prediction_system import FootballPredictionSystem
        
        system = FootballPredictionSystem()
        
        # Get URLs to process
        if urls is None:
            urls = self.get_pending_urls()
        
        # Scrape matches and extract results in a single request
        print(f"Scraping {len(urls)} matches (will extract results if available)...")
        scraped_count = 0
        results_extracted = 0
        errors = []
        processed_urls = []  # Track URLs that have been processed
        
        for i, url in enumerate(urls, 1):
            try:
                # prompt_user=False for batch training - use placeholders
                match_data = system.scraper.scrape_match(url, prompt_user=False)
                if match_data:
                    # Save match data (also updates training data if result is available)
                    ok = system.storage.save_match_with_result(match_data)
                    if ok:
                        scraped_count += 1
                        if match_data.get('actual_result'):
                            results_extracted += 1
                            processed_urls.append(url)  # Mark as processed
                else:
                    errors.append({'url': url, 'error': 'Failed to scrape'})
            except Exception as e:
                errors.append({'url': url, 'error': str(e)})
            
            # Progress indicator every 50 matches
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(urls)} matches processed...")
        
        # Remove processed URLs from results.txt
        if processed_urls:
            remaining_urls = [u for u in urls if u not in processed_urls]
            with open(self.results_file, 'w') as f:
                for url in remaining_urls:
                    f.write(url + '\n')
            print(f"Removed {len(processed_urls)} processed URLs from {self.results_file}")
        
        print(f"Matches scraped: {scraped_count}")
        print(f"Results extracted: {results_extracted}")
        
        # Train the model
        from prediction_model import FootballPredictor
        predictor = FootballPredictor()
        
        # Train global model
        global_result = predictor.train()
        
        # Train league-specific models
        league_results = {}
        leagues = set()
        if os.path.exists(self.matches_file):
            with open(self.matches_file, 'r') as f:
                matches = json.load(f)
                for match in matches:
                    league = match.get('info', {}).get('league')
                    if league:
                        leagues.add(league)
        
        for league in leagues:
            try:
                result = predictor.train(league=league)
                if 'error' not in result:
                    league_results[league] = result
            except Exception as e:
                league_results[league] = {'error': str(e)}
        
        # Save training timestamp
        self.save_training_time()
        
        return {
            'scraped_count': scraped_count,
            'results_extracted': results_extracted,
            'total_matches': self.get_saved_match_count(),
            'global_model': global_result,
            'league_models': league_results,
            'errors': errors[:5]  # Limit errors to first 5
        }
    
    def run(self, force: bool = False) -> Dict:
        """
        Main entry point for auto-training.
        
        Args:
            force: If True, train regardless of time elapsed
            
        Returns:
            Training result dictionary
        """
        # Check if we should train
        if not force and not self.should_train():
            last_train = self.get_last_training_time()
            remaining = timedelta(hours=TRAINING_THRESHOLD_HOURS) - (datetime.now() - last_train)
            hours_left = remaining.total_seconds() / 3600
            return {
                'skipped': True,
                'message': f'Training skipped. {hours_left:.1f} hours until next training.',
                'last_training': last_train.isoformat() if last_train else None
            }
        
        # Get pending URLs
        urls = self.get_pending_urls()
        
        if not urls and self.get_saved_match_count() == 0:
            return {
                'skipped': True,
                'message': 'No URLs in results.txt and no saved matches to train on.'
            }
        
        print(f"Starting auto-training at {datetime.now().isoformat()}")
        print(f"Pending URLs: {len(urls)}")
        print(f"Saved matches: {self.get_saved_match_count()}")
        
        result = self.train_model(urls)
        result['auto_train'] = True
        result['timestamp'] = datetime.now().isoformat()
        
        print(f"Training completed.")
        print(f"  Results extracted: {result.get('results_extracted', 0)}")
        print(f"  Matches scraped: {result['scraped_count']}")
        
        return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Auto-train the football prediction model'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force training regardless of time elapsed'
    )
    parser.add_argument(
        '--status', '-s',
        action='store_true',
        help='Check training status without training'
    )
    
    args = parser.parse_args()
    
    trainer = AutoTrainer()
    
    if args.status:
        # Just check status
        last_train = trainer.get_last_training_time()
        should = trainer.should_train()
        saved = trainer.get_saved_match_count()
        pending = len(trainer.get_pending_urls())
        
        if last_train:
            time_since = datetime.now() - last_train
            hours_elapsed = time_since.total_seconds() / 3600
            hours_left = max(0, TRAINING_THRESHOLD_HOURS - hours_elapsed)
            print(f"Last training: {last_train.isoformat()}")
            print(f"Time elapsed: {hours_elapsed:.1f} hours")
            print(f"Next training in: {hours_left:.1f} hours")
        else:
            print("No previous training found.")
        
        print(f"Saved matches: {saved}")
        print(f"Pending URLs: {pending}")
        print(f"Should train now: {should}")
        
    elif args.force:
        result = trainer.run(force=True)
        print(json.dumps(result, indent=2))
    else:
        result = trainer.run()
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
