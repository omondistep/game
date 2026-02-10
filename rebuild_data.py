#!/usr/bin/env python3
"""
Rebuild Data Script
Re-processes all URLs in results.txt to rebuild match data and results.
Useful when data files are corrupted.
"""

import json
import os
from datetime import datetime
from football_prediction_system import FootballPredictionSystem


def main():
    """Rebuild all data from results.txt."""
    results_file = "results.txt"
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return
    
    # Read URLs
    with open(results_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        print("No URLs found in results.txt")
        return
    
    print(f"Processing {len(urls)} URLs to rebuild data...")
    print()
    
    system = FootballPredictionSystem()
    saved_count = 0
    result_count = 0
    
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url[:80]}...")
        try:
            # Scrape and save match data
            match_data = system.scraper.scrape_match(url)
            if match_data:
                system.storage.save_match_data(match_data)
                saved_count += 1
                print(f"  ✓ Match data saved")
            else:
                print(f"  ✗ Failed to scrape match data")
            
            # Try to extract result
            ok = system.add_match_result(url)
            if ok:
                result_count += 1
                print(f"  ✓ Result extracted")
            else:
                print(f"  - Result not available yet")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print()
    print("=" * 50)
    print(f"Rebuild Complete:")
    print(f"  Match data saved: {saved_count}")
    print(f"  Results extracted: {result_count}")
    print("=" * 50)


if __name__ == '__main__':
    main()
