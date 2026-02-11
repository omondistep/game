# Football Prediction System - Deployment Guide

## Local Usage

### Terminal (CLI)
```bash
# Install dependencies
pip install -r requirements.txt

# Run predictions from terminal
python football_prediction_system.py --url "https://www.forebet.com/en/football/matches/..."
```

### Streamlit Web UI
```bash
# Install streamlit
pip install streamlit

# Run the web interface
streamlit run streamlit_app.py
```

## Cloud Deployment

### Streamlit Community Cloud (Recommended - Free)
1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set the main file path: `streamlit_app.py`
5. Deploy!

### HuggingFace Spaces (Free + GPU support)
1. Create a new Space on https://huggingface.co/spaces
2. Choose Streamlit as the SDK
3. Push your code
4. The app will automatically deploy

## Features
- ⚽ Predict match outcomes (Home/Draw/Away)
- 📊 Over/Under 2.5 goals prediction
- 🎯 Weighted predictions based on form
- 📈 Historical data analysis

## Dependencies
See `requirements.txt` for full list of Python dependencies.
