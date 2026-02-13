#!/usr/bin/env python3
"""
Football Prediction System - Streamlit Web App
A modern web interface for the football prediction system
Separate from CLI version - all features available through web UI
"""

import streamlit as st
import sys
import os
import json
from datetime import datetime, timedelta
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import prediction system components
from football_prediction_system import FootballPredictionSystem
from football_scraper import ForebetScraper
from data_storage import MatchDataStorage
from prediction_model import FootballPredictor

# Page config
st.set_page_config(
    page_title="Football Prediction System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .prediction-box {
        background: rgba(0,212,255,0.1);
        border-left: 4px solid #00d4ff;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .factor-positive { color: #22c55e; }
    .factor-negative { color: #ef4444; }
    .factor-neutral { color: #f59e0b; }
    .confidence-high { color: #22c55e; font-weight: bold; }
    .confidence-medium { color: #f59e0b; font-weight: bold; }
    .confidence-low { color: #ef4444; font-weight: bold; }
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background: rgba(255,255,255,0.05);
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_system():
    """Initialize prediction system (cached)."""
    return FootballPredictionSystem()


@st.cache_resource
def get_scraper():
    """Initialize scraper (cached)."""
    return ForebetScraper()


@st.cache_resource
def get_storage():
    """Initialize storage (cached)."""
    return MatchDataStorage()


@st.cache_resource
def get_predictor():
    """Initialize predictor (cached)."""
    return FootballPredictor()


def display_prediction(result):
    """Display prediction result in a formatted way."""
    if 'error' in result:
        st.error(f"❌ {result['error']}")
        return
    
    pred = result.get('prediction', {})
    analysis = result.get('analysis', {})
    match_data = result.get('match_data', {})
    teams = match_data.get('teams', {})
    home = teams.get('home', 'Home')
    away = teams.get('away', 'Away')
    
    # Match header
    st.markdown(f"""
    <div class="prediction-box">
        <h2>⚽ {home} vs {away}</h2>
        <p style="color: #888;">League: {match_data.get('match_info', {}).get('league', 'Unknown')} | Country: {match_data.get('match_info', {}).get('country', '')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Match Result")
        result_pred = pred.get('result', {})
        prediction = result_pred.get('prediction', '?')
        confidence = result_pred.get('confidence', 0) * 100
        
        # Map prediction to team name
        pred_label = {'1': f'{home} Win', 'X': 'Draw', '2': f'{away} Win'}.get(prediction, prediction)
        
        # Confidence color
        conf_class = 'confidence-high' if confidence >= 60 else ('confidence-medium' if confidence >= 40 else 'confidence-low')
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h3>{pred_label}</h3>
            <p class="{conf_class}">{confidence:.0f}% confidence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilities
        probs = result_pred.get('probabilities', {})
        st.write("**Probabilities:**")
        
        # Create probability bars
        prob_data = {
            f'{home} Win': probs.get('1', 0),
            'Draw': probs.get('X', 0),
            f'{away} Win': probs.get('2', 0)
        }
        for label, prob in prob_data.items():
            st.write(f"{label}: {prob*100:.1f}%")
            st.progress(prob)
    
    with col2:
        st.subheader("⚽ Over/Under 2.5")
        ou_pred = pred.get('over_under', {})
        ou_prediction = ou_pred.get('prediction', '?')
        ou_confidence = ou_pred.get('confidence', 0) * 100
        
        ou_conf_class = 'confidence-high' if ou_confidence >= 60 else ('confidence-medium' if ou_confidence >= 40 else 'confidence-low')
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px;">
            <h3>{ou_prediction} 2.5</h3>
            <p class="{ou_conf_class}">{ou_confidence:.0f}% confidence</p>
        </div>
        """, unsafe_allow_html=True)
        
        ou_probs = ou_pred.get('probabilities', {})
        st.write("**Probabilities:**")
        st.write(f"Over 2.5: {ou_probs.get('Over', 0)*100:.1f}%")
        st.progress(ou_probs.get('Over', 0))
        st.write(f"Under 2.5: {ou_probs.get('Under', 0)*100:.1f}%")
        st.progress(ou_probs.get('Under', 0))
    
    # Key Factors
    st.markdown("---")
    st.subheader("🔑 Key Factors")
    
    key_factors = analysis.get('key_factors', [])
    if key_factors:
        cols = st.columns(2)
        for i, factor in enumerate(key_factors[:6]):
            with cols[i % 2]:
                if '🔥' in factor or '✅' in factor:
                    st.success(factor)
                elif '⚠️' in factor:
                    st.warning(factor)
                else:
                    st.info(factor)
    else:
        st.info("No key factors identified")
    
    # Value Bets
    value_bets = analysis.get('value_bets', [])
    if value_bets:
        st.markdown("---")
        st.subheader("💰 Value Bets")
        for vb in value_bets:
            outcome_label = {'1': f'{home} Win', 'X': 'Draw', '2': f'{away} Win'}.get(vb['outcome'], vb['outcome'])
            st.success(f"✅ {outcome_label}: Market {vb['market']:.2f} vs Fair {vb['fair']:.2f} (+{vb['value_pct']:.1f}% value)")
    
    # Confidence Level
    st.markdown("---")
    conf = analysis.get('confidence', 'MEDIUM')
    conf_color = {'HIGH': 'green', 'MEDIUM': 'orange', 'LOW': 'red'}.get(conf, 'gray')
    st.markdown(f"**Overall Confidence:** :{conf_color}[{conf}]")
    
    # Model Agreement
    convergence = result.get('convergence', {})
    if convergence:
        st.write("**Model Agreement:**")
        col1, col2 = st.columns(2)
        with col1:
            if convergence.get('result_match'):
                st.success(f"✓ Both models predict: {convergence.get('ml_result')}")
            else:
                st.warning(f"⚠ ML: {convergence.get('ml_result')}, Weighted: {convergence.get('weighted_result')}")
        with col2:
            if convergence.get('ou_match'):
                st.success(f"✓ Both models predict: {convergence.get('ml_ou')} O/U")
            else:
                st.warning(f"⚠ ML: {convergence.get('ml_ou')}, Weighted: {convergence.get('weighted_ou')}")


def display_match_data(match_data):
    """Display detailed match data."""
    teams = match_data.get('teams', {})
    home = teams.get('home', 'Home')
    away = teams.get('away', 'Away')
    
    # Standings
    standings = match_data.get('standings', {})
    if standings:
        st.subheader("📊 Standings")
        col1, col2 = st.columns(2)
        
        home_stand = standings.get('home', {})
        away_stand = standings.get('away', {})
        
        with col1:
            st.markdown(f"**{home}**")
            st.write(f"Position: #{home_stand.get('position', 'N/A')}")
            st.write(f"Points: {home_stand.get('points', 'N/A')}")
            st.write(f"Form: {home_stand.get('form', 'N/A')}")
        
        with col2:
            st.markdown(f"**{away}**")
            st.write(f"Position: #{away_stand.get('position', 'N/A')}")
            st.write(f"Points: {away_stand.get('points', 'N/A')}")
            st.write(f"Form: {away_stand.get('form', 'N/A')}")
    
    # Form
    form = match_data.get('form', {})
    if form:
        st.subheader("📋 Recent Form")
        col1, col2 = st.columns(2)
        
        home_form = form.get('home', [])
        away_form = form.get('away', [])
        
        with col1:
            st.markdown(f"**{home}** (Last 6)")
            form_str = " ".join([
                "🟢" if r == 'W' else "🟡" if r == 'D' else "🔴"
                for r in home_form
            ])
            st.write(form_str if form_str else "No data")
        
        with col2:
            st.markdown(f"**{away}** (Last 6)")
            form_str = " ".join([
                "🟢" if r == 'W' else "🟡" if r == 'D' else "🔴"
                for r in away_form
            ])
            st.write(form_str if form_str else "No data")
    
    # H2H
    h2h = match_data.get('head_to_head', {})
    if h2h and h2h.get('matches'):
        st.subheader("🤝 Head to Head")
        summary = h2h.get('summary', {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{home} Wins", f"{summary.get('home_win_pct', 0)}%")
        with col2:
            st.metric("Draws", f"{summary.get('draw_pct', 0)}%")
        with col3:
            st.metric(f"{away} Wins", f"{summary.get('away_win_pct', 0)}%")
        
        with st.expander("View H2H Matches"):
            for match in h2h.get('matches', [])[:5]:
                st.write(f"{match.get('date', '')}: {match.get('home', '')} {match.get('score', '')} {match.get('away', '')}")


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚽ Football Prediction System</h1>
        <p>ML-based match predictions with weighted factor analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    system = get_system()
    storage = get_storage()
    predictor = get_predictor()
    
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Quick stats in sidebar
    st.sidebar.markdown("### Quick Stats")
    try:
        match_count = storage.get_saved_match_count()
        st.sidebar.metric("Saved Matches", match_count)
    except:
        st.sidebar.metric("Saved Matches", "N/A")
    
    try:
        training_data = storage.get_training_data()
        st.sidebar.metric("Training Examples", len(training_data))
    except:
        st.sidebar.metric("Training Examples", "N/A")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔮 Predict", 
        "📝 Add Result", 
        "📊 Statistics", 
        "🧠 Train",
        "📅 Historical",
        "⚙️ Settings"
    ])
    
    # Tab 1: Predict
    with tab1:
        st.header("Match Prediction")
        
        url = st.text_input(
            "Forebet Match URL",
            placeholder="https://www.forebet.com/en/football/matches/...",
            help="Paste a Forebet match URL to get predictions"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            save_data = st.checkbox("Save data for training", value=True)
        with col2:
            show_details = st.checkbox("Show detailed match data", value=False)
        
        if st.button("🔮 Predict", type="primary", use_container_width=True):
            if not url:
                st.error("Please enter a URL")
            else:
                progress_bar = st.progress(0, text="Initializing...")
                
                try:
                    progress_bar.progress(20, text="Scraping match data...")
                    result = system.predict_match(url, save_data=save_data)
                    
                    progress_bar.progress(80, text="Processing prediction...")
                    
                    if 'error' not in result:
                        progress_bar.progress(100, text="Complete!")
                        display_prediction(result)
                        
                        if show_details:
                            st.markdown("---")
                            st.subheader("📋 Detailed Match Data")
                            display_match_data(result.get('match_data', {}))
                    else:
                        progress_bar.empty()
                        st.error(f"❌ {result['error']}")
                        
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"Error: {str(e)}")
    
    # Tab 2: Add Result
    with tab2:
        st.header("Add Match Result")
        st.info("Record the actual result of a match to improve model training.")
        
        result_url = st.text_input(
            "Forebet Match URL",
            placeholder="https://www.forebet.com/en/football/matches/...",
            key="result_url"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            home_score = st.number_input("Home Score", min_value=0, value=0)
        with col2:
            away_score = st.number_input("Away Score", min_value=0, value=0)
        with col3:
            st.write("")  # Spacer
            st.write("")  # Spacer
        
        if st.button("📝 Save Result", type="primary", use_container_width=True):
            if not result_url:
                st.error("Please enter a URL")
            else:
                with st.spinner("Recording result..."):
                    try:
                        success = system.record_result(result_url, home_score, away_score)
                        if success:
                            st.success("✅ Result recorded successfully! The model will use this for future training.")
                        else:
                            st.error("Failed to record result. Check if the URL is valid.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Tab 3: Statistics
    with tab3:
        st.header("Model Statistics")
        
        if st.button("📊 Load Statistics", type="primary"):
            with st.spinner("Loading statistics..."):
                try:
                    stats = system.get_model_stats()
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Matches", stats.get('total_matches', 0))
                    with col2:
                        st.metric("Leagues", stats.get('leagues_count', 0))
                    with col3:
                        st.metric("Training Examples", stats.get('training_examples', 0))
                    with col4:
                        st.metric("Predictions Made", stats.get('predictions_count', 0))
                    
                    # Model accuracy
                    st.subheader("Model Accuracy")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        result_acc = stats.get('result_accuracy', 0) * 100
                        st.metric("Result Accuracy", f"{result_acc:.1f}%")
                    
                    with col2:
                        ou_acc = stats.get('ou_accuracy', 0) * 100
                        st.metric("O/U Accuracy", f"{ou_acc:.1f}%")
                    
                    # Last training
                    if stats.get('last_training'):
                        st.info(f"Last trained: {stats['last_training']}")
                    
                    # Raw stats
                    with st.expander("View Raw Statistics"):
                        st.json(stats)
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # League models
        st.subheader("League Models")
        if st.button("Load League Models"):
            try:
                models_dir = os.path.join(PROJECT_ROOT, 'models')
                if os.path.exists(models_dir):
                    leagues = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
                    
                    if leagues:
                        st.write(f"Found {len(leagues)} trained league models:")
                        
                        # Display in grid
                        cols = st.columns(4)
                        for i, league in enumerate(sorted(leagues)):
                            with cols[i % 4]:
                                st.markdown(f"📁 {league}")
                    else:
                        st.info("No league models found. Train the model first.")
                else:
                    st.info("Models directory not found.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Tab 4: Train
    with tab4:
        st.header("Train Models")
        st.info("Train the ML models with collected data. This may take several minutes depending on the amount of data.")
        
        col1, col2 = st.columns(2)
        with col1:
            force_train = st.checkbox("Force training (ignore time threshold)", value=False)
        with col2:
            train_leagues = st.checkbox("Train league-specific models", value=True)
        
        if st.button("🧠 Train Models", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Starting training...")
            
            try:
                progress_bar.progress(10, text="Loading training data...")
                
                progress_bar.progress(30, text="Training global model...")
                result = system.train_models(force=force_train)
                
                progress_bar.progress(70, text="Training league models...")
                
                progress_bar.progress(100, text="Complete!")
                
                st.success("✅ Training complete!")
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Result Accuracy", f"{result.get('result_accuracy', 0)*100:.1f}%")
                with col2:
                    st.metric("O/U Accuracy", f"{result.get('ou_accuracy', 0)*100:.1f}%")
                
                with st.expander("View Training Details"):
                    st.json(result)
                    
            except Exception as e:
                progress_bar.empty()
                st.error(f"Error: {str(e)}")
        
        # Calculate weights
        st.markdown("---")
        st.subheader("Calculate Optimal Weights")
        st.info("Calculate data-driven weights for prediction factors based on model performance.")
        
        if st.button("⚖️ Calculate Weights", type="secondary"):
            with st.spinner("Calculating weights..."):
                try:
                    from calculate_weights import main as calc_weights
                    # Run weight calculation
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, os.path.join(PROJECT_ROOT, 'calculate_weights.py')],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Weights calculated successfully!")
                        st.code(result.stdout)
                    else:
                        st.error(f"Error: {result.stderr}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Tab 5: Historical Data
    with tab5:
        st.header("Historical Data Scraping")
        st.info("Scrape historical match data from Forebet to build training dataset.")
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
        
        # Options
        incremental = st.checkbox("Incremental update (yesterday only)", value=True)
        
        if st.button("📅 Scrape Historical Data", type="primary"):
            if incremental:
                with st.spinner("Running incremental update..."):
                    try:
                        import subprocess
                        result = subprocess.run(
                            [sys.executable, os.path.join(PROJECT_ROOT, 'scrape_historical.py'), '--incremental'],
                            capture_output=True,
                            text=True,
                            cwd=PROJECT_ROOT
                        )
                        
                        if result.returncode == 0:
                            st.success("✅ Incremental update complete!")
                            st.code(result.stdout)
                        else:
                            st.error(f"Error: {result.stderr}")
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                if start_date > end_date:
                    st.error("Start date must be before end date")
                else:
                    with st.spinner(f"Scraping {start_date} to {end_date}..."):
                        try:
                            import subprocess
                            result = subprocess.run(
                                [
                                    sys.executable, 
                                    os.path.join(PROJECT_ROOT, 'scrape_historical.py'),
                                    '--start', str(start_date),
                                    '--end', str(end_date)
                                ],
                                capture_output=True,
                                text=True,
                                cwd=PROJECT_ROOT
                            )
                            
                            if result.returncode == 0:
                                st.success("✅ Historical scraping complete!")
                                st.code(result.stdout)
                            else:
                                st.error(f"Error: {result.stderr}")
                                
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        # Process queue
        st.markdown("---")
        st.subheader("Process Results Queue")
        st.info("Check URLs in results.txt for match results and update training data.")
        
        if st.button("📋 Process Queue", type="secondary"):
            with st.spinner("Processing queue..."):
                try:
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, os.path.join(PROJECT_ROOT, 'scrape_historical.py'), '--process-queue'],
                        capture_output=True,
                        text=True,
                        cwd=PROJECT_ROOT
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Queue processed!")
                        st.code(result.stdout)
                    else:
                        st.error(f"Error: {result.stderr}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Rebuild database
        st.markdown("---")
        st.subheader("Rebuild Training Database")
        st.info("Rebuild the training database from historical match data.")
        
        if st.button("🔧 Rebuild Database", type="secondary"):
            with st.spinner("Rebuilding database..."):
                try:
                    import subprocess
                    result = subprocess.run(
                        [sys.executable, os.path.join(PROJECT_ROOT, 'rebuild_data.py')],
                        capture_output=True,
                        text=True,
                        cwd=PROJECT_ROOT
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Database rebuilt!")
                        st.code(result.stdout)
                    else:
                        st.error(f"Error: {result.stderr}")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Tab 6: Settings
    with tab6:
        st.header("Settings & Configuration")
        
        # System info
        st.subheader("System Information")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Project Root:**")
            st.code(PROJECT_ROOT)
        with col2:
            st.write("**Python Version:**")
            st.code(f"{sys.version}")
        
        # Data directories
        st.subheader("Data Directories")
        
        data_dir = os.path.join(PROJECT_ROOT, 'data')
        models_dir = os.path.join(PROJECT_ROOT, 'models')
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                st.metric("Data Files", len(files))
            else:
                st.metric("Data Files", 0)
        
        with col2:
            if os.path.exists(models_dir):
                models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
                st.metric("Trained Models", len(models))
            else:
                st.metric("Trained Models", 0)
        
        with col3:
            results_file = os.path.join(PROJECT_ROOT, 'results.txt')
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    urls = [l.strip() for l in f if l.strip()]
                st.metric("Pending URLs", len(urls))
            else:
                st.metric("Pending URLs", 0)
        
        # View configuration
        st.subheader("Configuration Files")
        
        config_files = [
            ('Factor Weights', os.path.join(data_dir, 'factor_weights.json')),
            ('League Database', os.path.join(data_dir, 'leagues_db.json')),
            ('Comprehensive Leagues', os.path.join(data_dir, 'comprehensive_leagues_db.json')),
        ]
        
        for name, path in config_files:
            if os.path.exists(path):
                with st.expander(f"📄 {name}"):
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        st.json(data)
                    except Exception as e:
                        st.error(f"Error loading: {str(e)}")
        
        # Help
        st.markdown("---")
        st.subheader("Help & Documentation")
        
        st.markdown("""
        **Quick Start Guide:**
        
        1. **Predict**: Enter a Forebet URL to get match predictions
        2. **Add Result**: Record actual match results for training
        3. **Statistics**: View model performance metrics
        4. **Train**: Train or retrain the ML models
        5. **Historical**: Scrape historical data for training
        6. **Settings**: View system configuration
        
        **Tips:**
        - Save predictions to build your training dataset
        - Add results after matches are played
        - Train models when you have 50+ results
        - Use incremental updates for daily maintenance
        """)


if __name__ == "__main__":
    main()
