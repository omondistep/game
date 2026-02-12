#!/usr/bin/env python3
"""
Football Prediction System - Streamlit App
Deploy on: Streamlit Community Cloud or HuggingFace Spaces

Features:
- Match Predictions with ML & Statistical Analysis
- System Statistics & Training Data
- Model Training Interface
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os
import glob

# Import prediction system
from football_prediction_system import FootballPredictionSystem
from update_league_db import update_leagues_from_historical

# Initialize system
system = FootballPredictionSystem()

# ════════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="⚽ Football AI Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    [data-testid="stSidebar"] {
        background: rgba(30, 41, 59, 0.95);
    }
    .card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 1rem;
    }
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    }
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 8px;
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚽ Football AI Predictor")
    st.markdown("---")
    
    # System status
    st.markdown("### 📊 System Status")
    
    # Check models
    model_count = len(system.predictor.league_models) if hasattr(system.predictor, 'league_models') else 0
    st.metric("Models Loaded", model_count)
    
    # Check leagues
    leagues_db_path = 'data/leagues_db.json'
    if os.path.exists(leagues_db_path):
        with open(leagues_db_path, 'r') as f:
            leagues_db = json.load(f)
        st.metric("Leagues in DB", len(leagues_db))
    
    # Check historical data
    hist_files = glob.glob('historical_matches_*.json') + glob.glob('data/historical_matches_*.json')
    st.metric("Historical Files", len(hist_files))
    
    st.markdown("---")
    st.markdown("### 📁 Quick Links")
    st.markdown("- [GitHub Repository](https://github.com/omondistep/game)")
    st.markdown("- [Forebet](https://www.forebet.com)")

# ════════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ════════════════════════════════════════════════════════════════════════════════

tab_predict, tab_stats, tab_train, tab_leagues, tab_settings = st.tabs([
    "🔮 Predict",
    "📊 Statistics",
    "🧠 Train Model",
    "🏆 Leagues",
    "⚙️ Settings"
])

# ════════════════════════════════════════════════════════════════════════════════
# PREDICT TAB
# ════════════════════════════════════════════════════════════════════════════════

with tab_predict:
    st.markdown("### 🔮 Match Prediction")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        match_url = st.text_input(
            "Forebet Match URL",
            placeholder="https://www.forebet.com/en/football/matches/...",
            label_visibility="collapsed"
        )
    with col2:
        predict_btn = st.button("🔮 Predict", use_container_width=True)
    
    if predict_btn and match_url:
        with st.spinner("🤖 AI is analyzing match data..."):
            result = system.predict_match(match_url)
            
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                # Display prediction
                system.display_prediction(result)
    
    elif predict_btn and not match_url:
        st.warning("Please enter a match URL")

# ════════════════════════════════════════════════════════════════════════════════
# STATISTICS TAB
# ════════════════════════════════════════════════════════════════════════════════

with tab_stats:
    st.markdown("### 📊 System Statistics")
    
    # Model stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", model_count)
    
    with col2:
        if os.path.exists('data/leagues_db.json'):
            st.metric("Leagues", len(leagues_db))
    
    with col3:
        total_matches = 0
        for info in leagues_db.values():
            total_matches += info.get('match_count', 0)
        st.metric("Total Matches", total_matches)
    
    with col4:
        total_teams = 0
        for info in leagues_db.values():
            total_teams += len(info.get('teams', {}))
        st.metric("Total Teams", total_teams)
    
    st.markdown("---")
    
    # Training history
    if os.path.exists('data/last_training.json'):
        with open('data/last_training.json', 'r') as f:
            last_training = json.load(f)
        st.markdown(f"**Last Training:** {last_training.get('timestamp', 'N/A')}")
        st.markdown(f"**Examples Used:** {last_training.get('example_count', 0)}")

# ════════════════════════════════════════════════════════════════════════════════
# TRAIN MODEL TAB
# ════════════════════════════════════════════════════════════════════════════════

with tab_train:
    st.markdown("### 🧠 Train Model")
    
    st.info("Training a new model will use all available historical data.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        league_filter = st.selectbox(
            "League Filter (leave empty for all)",
            options=[""] + list(leagues_db.keys()),
            format_func=lambda x: "All Leagues" if x == "" else x
        )
    
    with col2:
        st.markdown("### ")
        train_btn = st.button("🧠 Train New Model", use_container_width=True)
    
    if train_btn:
        with st.spinner("Training model..."):
            # Extract league from filter if selected
            league = None if league_filter == "" else league_filter
            
            # Get training data
            training_data = system.storage.get_league_training_data(league)
            
            if not training_data:
                st.error("No training data available!")
            else:
                examples = training_data[0].get('examples', []) if training_data else []
                if len(examples) < 5:
                    st.error(f"Not enough examples! Found {len(examples)}, need at least 5.")
                else:
                    result = system.train_model(training_data, league=league)
                    
                    if 'error' in result:
                        st.error(f"Training failed: {result['error']}")
                    else:
                        st.success(f"Model trained successfully!")
                        st.markdown(f"- Examples used: {len(examples)}")
                        st.markdown(f"- League: {league if league else 'All'}")

# ════════════════════════════════════════════════════════════════════════════════
# LEAGUES TAB
# ════════════════════════════════════════════════════════════════════════════════

with tab_leagues:
    st.markdown("### 🏆 League Database")
    
    # Update leagues button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Update leagues from historical data files")
    with col2:
        update_btn = st.button("📥 Update Leagues", use_container_width=True)
    
    if update_btn:
        with st.spinner("Updating leagues..."):
            update_leagues_from_historical()
            st.success("Leagues updated successfully!")
            st.rerun()
    
    st.markdown("---")
    
    # Display leagues
    if leagues_db:
        # Convert to DataFrame for display
        league_data = []
        for key, info in leagues_db.items():
            league_data.append({
                'Code': key,
                'League': info.get('league', 'Unknown'),
                'Country': info.get('country', 'Unknown'),
                'Matches': info.get('match_count', 0),
                'Teams': len(info.get('teams', {}))
            })
        
        df = pd.DataFrame(league_data)
        df = df.sort_values('Matches', ascending=False)
        
        # Pagination
        page_size = 20
        total_pages = (len(df) + page_size - 1) // page_size
        
        col1, col2 = st.columns([1, 3])
        with col1:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        st.dataframe(
            df.iloc[start_idx:end_idx],
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown(f"Showing {start_idx + 1}-{min(end_idx, len(df))} of {len(df)} leagues")
    else:
        st.info("No leagues found. Update from historical data first.")

# ════════════════════════════════════════════════════════════════════════════════
# SETTINGS TAB
# ════════════════════════════════════════════════════════════════════════════════

with tab_settings:
    st.markdown("### ⚙️ Settings")
    
    # Data paths
    st.markdown("#### 📁 Data Files")
    
    data_files = [
        ("leagues_db.json", "League database"),
        ("last_training.json", "Last training info"),
        ("historical_matches_*.json", "Historical match data"),
    ]
    
    for pattern, desc in data_files:
        files = glob.glob(f'data/{pattern}') + glob.glob(pattern)
        st.markdown(f"**{desc}** ({len(files)} files)")
    
    st.markdown("---")
    
    # Model directory
    st.markdown("#### 🤖 Models")
    model_dirs = glob.glob('models/*/')
    st.markdown(f"**Country Models:** {len(model_dirs)}")
    
    # Clear cache
    st.markdown("#### 🗑️ Cache")
    if st.button("Clear All Caches"):
        st.cache_resource.clear()
        st.rerun()
