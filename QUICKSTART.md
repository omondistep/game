# Quick Start Guide

## Get Started in 3 Steps

### Step 1: Predict Your First Match

```bash
./football_env/bin/python football_prediction_system.py predict --url "https://www.forebet.com/en/football/matches/juventus-lazio-2344437"
```

This will:
- Scrape match data from Forebet
- Analyze team statistics and form
- Make predictions for 1X2 and Over/Under markets
- Save the data for future training

### Step 2: Add Match Results (After Games Are Played)

```bash
./football_env/bin/python football_prediction_system.py result --url "https://www.forebet.com/en/football/matches/juventus-lazio-2344437"
```

This adds the actual result to your training dataset.

### Step 3: Train the Model (After 10+ Results)

```bash
./football_env/bin/python football_prediction_system.py train
```

This trains the machine learning models on your collected data.

## Example Workflow

### Week 1-2: Collect Data
```bash
# Predict upcoming matches
./football_env/bin/python football_prediction_system.py predict --url "URL1"
./football_env/bin/python football_prediction_system.py predict --url "URL2"
./football_env/bin/python football_prediction_system.py predict --url "URL3"
# ... predict 10-20 matches
```

### Week 3: Add Results
```bash
# After matches are played, add results
./football_env/bin/python football_prediction_system.py result --url "URL1"
./football_env/bin/python football_prediction_system.py result --url "URL2"
./football_env/bin/python football_prediction_system.py result --url "URL3"
# ... add all results
```

### Week 3: Train Model
```bash
# Train the model with collected data
./football_env/bin/python football_prediction_system.py train
```

### Week 4+: Use Trained Model
```bash
# Now predictions will use your trained model!
./football_env/bin/python football_prediction_system.py predict --url "NEW_URL"
```

## Check Your Progress

```bash
./football_env/bin/python football_prediction_system.py stats
```

This shows:
- Number of matches scraped
- Number of results added
- Training examples available
- Whether model is trained

## Finding Match URLs

1. Go to https://www.forebet.com
2. Navigate to your league (e.g., Premier League, La Liga, Serie A)
3. Click on an upcoming match
4. Copy the URL from your browser
5. Use it with the predict command

Example URLs:
- `https://www.forebet.com/en/football/matches/manchester-united-liverpool-1234567`
- `https://www.forebet.com/en/football/matches/barcelona-real-madrid-2345678`
- `https://www.forebet.com/en/football/matches/bayern-munich-dortmund-3456789`

## Understanding the Output

### Match Result Prediction
- **1**: Home team wins
- **X**: Draw
- **2**: Away team wins

### Confidence Levels
- **High**: >60% confidence in prediction
- **Medium**: 45-60% confidence
- **Low**: <45% confidence

### Value Bets
Shows when market odds are higher than our computed fair odds, indicating potential value.

## Tips for Best Results

1. **Collect diverse data**: Include different leagues, teams, and match types
2. **Add results promptly**: Update results soon after matches finish
3. **Retrain regularly**: Retrain every 20-30 new results
4. **Focus on one league**: Better to have deep data on one league than shallow data on many
5. **Track your accuracy**: Keep notes on predictions vs actual results

## Troubleshooting

### "Insufficient training data"
- You need at least 10 matches with results
- Keep using the system in odds-based mode until you have enough data

### Scraping fails
- Check your internet connection
- Verify the URL is correct
- Try a different match URL

### Low accuracy
- Add more training data (50+ matches recommended)
- Ensure you're adding results correctly
- Retrain the model

## Next Steps

Once you're comfortable with the basics:
- Read the full [README.md](README.md) for advanced features
- Explore the Python API for batch processing
- Customize the feature extraction in [`data_storage.py`](data_storage.py)
- Experiment with different models in [`prediction_model.py`](prediction_model.py)

## Support

For issues or questions:
1. Check the [README.md](README.md) for detailed documentation
2. Review the code comments in each module
3. Test with the provided example URL first

Happy predicting! 🎯⚽
