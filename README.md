# Football Match Outcome Prediction System

A comprehensive machine learning system that scrapes football match data from Forebet and predicts match outcomes including 1X2 results and Over/Under markets.

## Features

- **Web Scraping**: Extracts detailed match data from Forebet including:
  - Team names and standings
  - Last 6 matches form (W/D/L)
  - Goals scored/conceded statistics
  - Home/Away performance
  - Market odds
  - Match information (date, time, venue, league)

- **Data Storage**: Persistent storage system that:
  - Saves all scraped match data
  - Stores actual match results
  - Builds training dataset automatically
  - Tracks historical data for learning

- **Machine Learning Prediction**: Advanced prediction models that:
  - Predict match outcomes (1/X/2)
  - Predict Over/Under 2.5 goals
  - Compute fair odds from probabilities
  - Identify value betting opportunities
  - Improve accuracy with more training data

- **Comprehensive Analysis**: Provides:
  - Team statistics comparison
  - Form analysis
  - Key factors affecting the match
  - Confidence levels
  - Value bet identification

## Installation

1. **Create and activate virtual environment** (if not already done):
```bash
python -m venv football_env
source football_env/bin/activate  # On Linux/Mac
# or
football_env\Scripts\activate  # On Windows
```

2. **Install required packages**:
```bash
pip install requests beautifulsoup4 lxml scikit-learn numpy streamlit
```

3. **Run the Streamlit app**:
```bash
streamlit run streamlit_app.py
```

## Usage

### Streamlit Web Interface (Recommended)

Launch the modern web interface:
```bash
streamlit run streamlit_app.py
```

Features:
- **🔮 Predict Tab**: Enter a match URL and get instant predictions
- **📊 Statistics Tab**: View model performance and league statistics
- **🧠 Train Model Tab**: Train or retrain the ML models
- **✅ Add Result Tab**: Record match results for training
- **🏆 Leagues Tab**: Manage league mappings and mappings database
- **📅 Historical Data Tab**: Import historical match data from Forebet date pages
- **⚙️ Settings Tab**: View configuration and deployment options

### 1. Predict a Match

To predict the outcome of an upcoming match:

```bash
python football_prediction_system.py predict --url "https://www.forebet.com/en/football/matches/juventus-lazio-2344437"
```

This will:
- Scrape match data from the URL
- Extract relevant features
- Make predictions using the model (or odds-based if not trained)
- Display comprehensive analysis
- Save the prediction to a JSON file
- Store data for future training

### 2. Add Match Result

After a match is played, add the actual result for training:

```bash
python football_prediction_system.py result --url "https://www.forebet.com/en/football/matches/juventus-lazio-2344437"
```

This will:
- Extract the actual match result
- Save it to the results database
- Add it to the training dataset

### 3. Train the Model

Once you have at least 10 matches with results:

```bash
python football_prediction_system.py train
```

This will:
- Load all training data
- Train Random Forest and Gradient Boosting models
- Display training accuracy
- Save the trained models
- Show feature importance

### 4. View Statistics

Check system statistics:

```bash
python football_prediction_system.py stats
```

## System Architecture

### Components

1. **football_scraper.py**: Web scraping module
   - `ForebetScraper` class for extracting match data
   - Handles team info, form, odds, predictions, statistics

2. **data_storage.py**: Data persistence module
   - `MatchDataStorage` class for saving/loading data
   - Manages matches, results, and training datasets
   - Automatic feature extraction

3. **prediction_model.py**: Machine learning module
   - `FootballPredictor` class with ML models
   - Random Forest for match results
   - Gradient Boosting for Over/Under
   - Feature scaling and preprocessing

4. **football_prediction_system.py**: Main system
   - Integrates all components
   - Command-line interface
   - Prediction display and analysis

### Data Flow

```
Forebet URL → Scraper → Match Data → Storage → Features → Model → Predictions
                                         ↓
                                    Results → Training Data → Model Training
```

## Features Extracted

The system extracts and uses the following features for prediction:

- **Team Position**: League standings position
- **Form Points**: Points from last 6 matches (W=3, D=1, L=0)
- **Recent Form**: Points from last 3 matches
- **Goals Statistics**: Average goals scored/conceded
- **Market Odds**: Bookmaker odds for 1/X/2 and Over/Under
- **Forebet Prediction**: Forebet's own prediction and probability

## Prediction Output

The system provides:

### Match Result Prediction
- Predicted outcome (1/X/2)
- Probability for each outcome
- Confidence level
- Computed fair odds

### Over/Under Prediction
- Predicted Over/Under 2.5 goals
- Probabilities for each
- Confidence level
- Computed fair odds

### Analysis
- Key factors affecting the match
- Value betting opportunities
- Odds comparison (market vs computed)
- Overall confidence assessment

## Example Output

```
==============================================================
MATCH PREDICTION
==============================================================

Juventus vs Lazio
Date: 08 Feb 2026 20:45
Venue: Allianz Stadium
League: Serie A

--------------------------------------------------------------
TEAM STATISTICS
--------------------------------------------------------------

Juventus (Home):
  Position: #4
  Form: LWDWWL
  Goals: 1.67 scored, 0.94 conceded per game

Lazio (Away):
  Position: #8
  Form: WDLWDL
  Goals: 1.04 scored, 1.20 conceded per game

--------------------------------------------------------------
OUR PREDICTIONS
--------------------------------------------------------------

Match Result: 1
  Confidence: 45.2%
  Probabilities:
    Home Win (1): 45.2%
    Draw (X):     32.1%
    Away Win (2): 22.7%

Over/Under 2.5: Under
  Confidence: 55.3%
  Probabilities:
    Over:  44.7%
    Under: 55.3%

--------------------------------------------------------------
KEY FACTORS
--------------------------------------------------------------
  • Home team in excellent form (4 wins in last 6)
  • Home team significantly higher in table (#4 vs #8)
  • Low-scoring match expected (avg 2.3 goals)

Overall Confidence: MEDIUM
```

## Training the Model

The system uses a **continuous learning approach**:

1. **Initial Phase**: Uses odds-based predictions (no training needed)
2. **Learning Phase**: As you add match results, the training dataset grows
3. **Trained Phase**: Once 10+ results are available, train the ML models
4. **Improvement**: Model accuracy improves with more training data

### Recommended Workflow

1. Predict 20-30 upcoming matches
2. Wait for matches to be played
3. Add results for all matches
4. Train the model
5. Continue predicting and adding results
6. Retrain periodically (every 20-30 new results)

## Auto-Training Script

The system includes an **auto-training script** (`auto_train.py`) that automatically trains the model using URLs from `results.txt` when enough time has passed since the last training.

### Quick Start

```bash
# Training commands
./fbtrain              # Auto-train if 20+ hours passed
./fbtrain status       # Check training status
./fbtrain force        # Force training now

# Extract results (after matches are played)
./fbextract            # Extract results from results.txt

# Model statistics
./fbstats              # Show accuracy and improvement trend

# Alternative shell scripts
./train.sh              # Same as ./fbtrain
./extract_results.py     # Same as ./fbextract
./model_stats.py         # Same as ./fbstats
```

### results.txt File

- **Location**: `results.txt` is located in the game folder root (`/home/stom/game/results.txt`)
- **Purpose**: Stores URLs of predicted matches for future learning
- **Persistence**: The file persists even after clearing - it stays in the folder but becomes empty
- **Auto-population**: When you predict matches without training, URLs are automatically added to this queue

### Automatic Training

The auto-training script runs in the background and:
1. Checks if 20+ hours have passed since the last training
2. Reads pending URLs from `results.txt`
3. Scrapes match data for each URL
4. Trains the global model
5. Trains league-specific models for each league found
6. Updates the last training timestamp
7. Clears the `results.txt` queue

### Usage

#### Check Training Status

```bash
python auto_train.py --status
```

This shows:
- Last training timestamp
- Time elapsed since last training
- Hours until next auto-training
- Number of saved matches
- Number of pending URLs in queue

#### Force Training

```bash
python auto_train.py --force
```

Forces training regardless of time elapsed.

#### Auto-Train (Default)

```bash
python auto_train.py
```

Runs training only if 20+ hours have passed since last training.

### Setting Up Auto-Training (Cron/Linux)

To run auto-training automatically every day:

```bash
# Open crontab
crontab -e

# Add this line to run at 2 AM daily
0 2 * * * cd /home/stom/game && source football_env/bin/activate && python auto_train.py >> /home/stom/game/auto_train.log 2>&1
```

### Setting Up Auto-Training (Windows Task Scheduler)

1. Create a batch file `run_auto_train.bat`:
```batch
@echo off
cd /home/stom/game
call football_env\Scripts\activate.bat
python auto_train.py
```

2. Schedule the batch file to run daily via Task Scheduler

## Historical Data Scraping

The system can scrape historical match data from Forebet's date pages to build a comprehensive training dataset.

### Scrape Historical Matches

```bash
# Scrape matches for a specific date range
python scrape_historical.py --start 2026-01-25 --end 2026-02-11

# Scrape a single date
python scrape_historical.py --date 2026-01-25

# Scrape last N days
python scrape_historical.py --days 7
```

This will:
1. Fetch match listings from Forebet's date pages
2. Scrape detailed match data for each match
3. Extract actual results for completed matches
4. Save data to `data/historical_matches_YYYY-MM-DD.json`
5. Update the training dataset automatically

### Rebuild Training Database

To rebuild the entire training database from historical data:

```bash
# Rebuild from all historical data files
python rebuild_data.py

# Rebuild with specific date range
python rebuild_data.py --start 2026-01-25 --end 2026-02-11
```

This will:
1. Load all historical match data files
2. Extract features and results
3. Build a comprehensive training dataset
4. Train models for each league with sufficient data
5. Save training statistics

### Build League Database

The system maintains a comprehensive league database for accurate league identification:

```bash
# Build/update league database from historical data
python build_league_db.py

# This creates data/leagues_db.json with:
# - short_codes: League short codes (e.g., "E0" for Premier League)
# - url_paths: League URL paths for lookup
# - country/league mappings
```

### Calculate Data-Driven Weights

The prediction model uses data-driven weights based on model accuracy:

```bash
# Calculate weights from model performance
python calculate_weights.py

# This creates data/factor_weights.json with:
# - Model accuracy-based weights
# - League-specific performance metrics
```

## Daily Training Script

The `daily_train.py` script provides automated daily training with historical data updates:

```bash
# Run daily training
python daily_train.py

# This will:
# 1. Scrape yesterday's matches
# 2. Update historical data
# 3. Rebuild training database
# 4. Retrain all models
# 5. Log training results
```

### Systemd Service (Linux)

The system includes systemd service files for automated daily training:

```bash
# Install the service
sudo cp football_train.service /etc/systemd/system/
sudo cp football_train.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable football_train.timer
sudo systemctl start football_train.timer

# Check status
systemctl status football_train.timer
```

The timer runs daily at 2 AM to:
1. Scrape the previous day's matches
2. Update results for completed matches
3. Retrain models with new data

## League-Specific Models

The system supports **league-specific models** that capture nuances of different leagues:

- Models are stored in `models/leagues/{league_name}/`
- When predicting, the system uses the league-specific model if available
- Falls back to global model if league-specific model doesn't exist
- Automatically trains league-specific models when enough data is available

## Training Commands

### Train Global Model

```bash
python football_prediction_system.py train
```

### Train Specific League

```bash
python football_prediction_system.py train --league "Serie A"
```

### View Statistics

```bash
python football_prediction_system.py stats
```

## File Structure

```
game/
├── streamlit_app.py              # Modern Streamlit web interface
├── football_prediction_system.py  # Main system with CLI
├── football_scraper.py            # Web scraping module
├── data_storage.py                # Data persistence module
├── prediction_model.py            # ML models (Random Forest, Gradient Boosting)
├── auto_train.py                  # Auto-training script (20-hour threshold)
├── scrape_historical.py          # Historical match data scraper
├── train.sh                      # Quick training script
├── fbtrain                       # Training command (executable)
├── fbextract                     # Extract results command (executable)
├── fbstats                       # Model statistics command (executable)
├── extract_results.py            # Extract results script
├── model_stats.py                # Statistics script
├── test_scraper.py               # Testing script
├── README.md                     # This file
├── results.txt                   # URL queue (persists, cleared after training)
├── training_history.json         # Training history for tracking improvement
├── last_training.json            # Last training timestamp
├── data/                         # Data directory
│   ├── matches.json             # Scraped match data
│   ├── results.json             # Actual results
│   ├── training_data.pkl        # Training dataset
│   ├── league_mapping.json      # League code to name mappings
│   ├── leagues_db.json          # League database
│   └── historical_matches_*.json # Historical match data by date
├── models/                      # Models directory
│   ├── result_model.pkl         # Trained global result model
│   ├── ou_model.pkl             # Trained global O/U model
│   ├── scaler.pkl               # Feature scaler
│   └── leagues/                 # League-specific models
│       └── {league_name}/
│           ├── result_model.pkl  # League result model
│           ├── ou_model.pkl     # League O/U model
│           └── scaler.pkl      # League scaler
├── api/                          # FastAPI web API
│   └── main.py                   # API endpoints
├── old_game/                     # Legacy code
└── prediction_*.json             # Saved predictions
```

## Advanced Usage

### Python API

You can also use the system programmatically:

```python
from football_prediction_system import FootballPredictionSystem

# Initialize system
system = FootballPredictionSystem()

# Predict a match
url = "https://www.forebet.com/en/football/matches/team1-team2-123456"
result = system.predict_match(url)

# Display prediction
system.display_prediction(result)

# Add result after match is played
system.add_match_result(url)

# Train model
metrics = system.train_model()
print(f"Model accuracy: {metrics['result_accuracy']:.2%}")
```

### Batch Processing

Process multiple matches:

```python
urls = [
    "https://www.forebet.com/en/football/matches/match1",
    "https://www.forebet.com/en/football/matches/match2",
    "https://www.forebet.com/en/football/matches/match3"
]

for url in urls:
    result = system.predict_match(url)
    system.display_prediction(result)
```

## Limitations

- Requires active internet connection for scraping
- Forebet page structure changes may break scraper
- Initial predictions use odds only (before model training)
- Minimum 10 matches with results needed for training
- Accuracy depends on quality and quantity of training data

## Future Enhancements

- [ ] Add more leagues and competitions
- [ ] Include head-to-head statistics
- [ ] Add player injury/suspension data
- [ ] Implement ensemble models
- [ ] Add betting strategy recommendations
- [ ] Create web interface
- [ ] Add real-time odds monitoring
- [ ] Include weather data
- [ ] Add team news sentiment analysis

## Troubleshooting

### "Insufficient training data" error
- You need at least 10 matches with results to train the model
- Use the system in odds-based mode until you have enough data

### Scraping fails
- Check internet connection
- Verify the URL is correct and accessible
- Forebet may have changed their page structure

### Low prediction accuracy
- Add more training data (50+ matches recommended)
- Ensure results are being added correctly
- Retrain the model with new data

## License

This project is for educational purposes. Please respect Forebet's terms of service when scraping their website.

## Contributing

Contributions are welcome! Areas for improvement:
- Better feature engineering
- Additional data sources
- Improved scraping robustness
- More sophisticated models
- Better visualization

## Disclaimer

This system is for educational and research purposes only. Sports betting involves risk. Always gamble responsibly and within your means. Past performance does not guarantee future results.
