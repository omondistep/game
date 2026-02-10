#!/usr/bin/env python3
"""
Extract Results Script
Reads URLs from results.txt and extracts actual match results
for all played matches. This builds the training dataset.
"""

import json
import os
from datetime import datetime
from football_prediction_system import FootballPredictionSystem


def main():
    """Extract results for all URLs in results.txt."""
    results_file = "results.txt"
    
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        return
    
    # Read URLs from results.txt
    with open(results_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        print("No URLs found in results.txt")
        return
    
    print(f"Processing {len(urls)} URLs from {results_file}")
    print()
    
    system = FootballPredictionSystem()
    success_count = 0
    error_count = 0
    processed_urls = []  # Track URLs with results extracted
    
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Processing: {url}")
        try:
            ok = system.add_match_result(url)
            if ok:
                print(f"  ✓ Result extracted successfully")
                success_count += 1
                processed_urls.append(url)  # Mark as processed
            else:
                print(f"  ✗ Failed or match not played yet")
                error_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_count += 1
    
    # Remove processed URLs from results.txt
    if processed_urls:
        remaining_urls = [u for u in urls if u not in processed_urls]
        with open(results_file, 'w') as f:
            for url in remaining_urls:
                f.write(url + '\n')
        print(f"\nRemoved {len(processed_urls)} processed URLs from {results_file}")
    
    print()
    print("=" * 50)
    print(f"Summary:")
    print(f"  Total URLs: {len(urls)}")
    print(f"  Success: {success_count}")
    print(f"  Failed/Not Played: {error_count}")
    print("=" * 50)


if __name__ == '__main__':
    main()
