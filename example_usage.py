#!/usr/bin/env python3
"""
Example Usage of Football Prediction System
Demonstrates how to use the system programmatically
"""

from football_prediction_system import FootballPredictionSystem

def main():
    # Initialize the system
    print("Initializing Football Prediction System...")
    system = FootballPredictionSystem()
    
    # Example match URL
    match_url = "https://www.forebet.com/en/football/matches/juventus-lazio-2344437"
    
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Predict a Single Match")
    print("=" * 60)
    
    # Predict the match
    result = system.predict_match(match_url, save_data=True)
    
    # Display the prediction
    if 'error' not in result:
        system.display_prediction(result)
    else:
        print(f"Error: {result['error']}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Check System Statistics")
    print("=" * 60)
    
    # Get statistics
    stats = system.get_statistics()
    print(f"\nMatches scraped: {stats['storage']['total_matches']}")
    print(f"Results added: {stats['storage']['total_results']}")
    print(f"Training examples: {stats['storage']['training_examples']}")
    print(f"Model trained: {stats['model_trained']}")
    print(f"Ready for training: {stats['ready_for_training']}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Batch Processing Multiple Matches")
    print("=" * 60)
    
    # Example: Process multiple matches
    match_urls = [
        "https://www.forebet.com/en/football/matches/juventus-lazio-2344437",
        # Add more URLs here when you have them
    ]
    
    print(f"\nProcessing {len(match_urls)} match(es)...")
    
    for i, url in enumerate(match_urls, 1):
        print(f"\n[{i}/{len(match_urls)}] Processing: {url}")
        result = system.predict_match(url, save_data=True)
        
        if 'error' not in result:
            print(f"✓ Prediction: {result['our_prediction']['result']['prediction']}")
            print(f"  Confidence: {result['our_prediction']['result']['confidence']:.1%}")
        else:
            print(f"✗ Error: {result['error']}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Training Workflow")
    print("=" * 60)
    
    print("\nTo train the model:")
    print("1. Predict 10+ matches (done above)")
    print("2. Wait for matches to be played")
    print("3. Add results using:")
    print("   system.add_match_result(url)")
    print("4. Train the model using:")
    print("   system.train_model()")
    
    # Check if we can train
    if stats['ready_for_training']:
        print("\n✓ System is ready for training!")
        print("Run: system.train_model()")
    else:
        needed = 10 - stats['storage']['training_examples']
        print(f"\n⚠ Need {needed} more match results before training")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Accessing Prediction Data")
    print("=" * 60)
    
    if 'error' not in result:
        print("\nYou can access prediction data programmatically:")
        print(f"Home team: {result['match_info']['home_team']}")
        print(f"Away team: {result['match_info']['away_team']}")
        print(f"Predicted result: {result['our_prediction']['result']['prediction']}")
        print(f"Home win probability: {result['our_prediction']['result']['probabilities']['1']:.1%}")
        print(f"Draw probability: {result['our_prediction']['result']['probabilities']['X']:.1%}")
        print(f"Away win probability: {result['our_prediction']['result']['probabilities']['2']:.1%}")
        print(f"Over/Under prediction: {result['our_prediction']['over_under']['prediction']}")
        
        # Access team statistics
        print(f"\nHome team form: {result['team_stats']['home']['form']}")
        print(f"Away team form: {result['team_stats']['away']['form']}")
        
        # Access analysis
        if result['analysis']['key_factors']:
            print(f"\nKey factors:")
            for factor in result['analysis']['key_factors']:
                print(f"  - {factor}")
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nFor more information:")
    print("- Read QUICKSTART.md for basic usage")
    print("- Read README.md for detailed documentation")
    print("- Check the source code for advanced customization")


if __name__ == "__main__":
    main()
