#!/usr/bin/env python3
"""
Football Prediction System - Modern Streamlit App
Deploy on: Streamlit Community Cloud or HuggingFace Spaces

Features:
- Match Predictions with ML & Statistical Analysis
- System Statistics & Training Data
- Model Training Interface
- Match Results Entry
- Historical Data Import
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import re
import os
import glob

# Import prediction system
from football_prediction_system import FootballPredictionSystem
from scrape_historical import HistoricalForebetScraper

# Initialize system
system = FootballPredictionSystem()

# ════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="⚽ Football AI Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '### ⚽ Football AI Predictor\nML-powered football match predictions using historical data and form analysis.',
    }
)

# Custom CSS for modern dark theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --secondary: #10b981;
        --accent: #f59e0b;
        --danger: #ef4444;
        --bg-dark: #0f172a;
        --bg-card: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }
    
    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(30, 41, 59, 0.95);
        border-right: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Cards */
    .card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.2);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        transform: scale(1.02);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        color: #f8fafc;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.5);
        padding: 8px;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 12px 24px;
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    
    /* Dividers */
    hr {
        border-color: rgba(99, 102, 241, 0.2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        color: #f8fafc;
    }
    
    /* Tables */
    [data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
    }
 </style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════

def render_card(title, icon="📊", **kwargs):
    """Render a styled card with title and metrics."""
    st.markdown(f"""
    <div class="card">
        <h3 style="margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;">
            <span>{icon}</span> {title}
        </h3>
    </div>
    """, unsafe_allow_html=True)

def render_prediction_badge(prediction, confidence, color="green"):
    """Render a prediction badge."""
    color_map = {
        "green": ("🟢", "High Confidence"),
        "yellow": ("🟡", "Medium Confidence"),
        "red": ("🔴", "Low Confidence"),
    }
    icon, label = color_map.get(color, ("⚪", "Unknown"))
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem; background: rgba(99, 102, 241, 0.1); border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.3);">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 1.5rem; font-weight: 700;">{prediction}</div>
        <div style="color: #94a3b8; font-size: 0.875rem;">{label} • {confidence:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

def render_progress_bar(label, value, max_value=100, color="#6366f1"):
    """Render a styled progress bar."""
    pct = value / max_value if max_value else 0
    st.markdown(f"""
    <div style="margin: 0.5rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="color: #94a3b8;">{label}</span>
            <span style="font-weight: 600;">{value:.1f}%</span>
        </div>
        <div style="background: rgba(99, 102, 241, 0.2); border-radius: 8px; height: 8px; overflow: hidden;">
            <div style="background: {color}; width: {pct * 100}%; height: 100%; border-radius: 8px; transition: width 0.5s;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_injury_info(home_team, away_team, injuries_data):
    """Display injury information for both teams."""
    if not injuries_data:
        return
    
    st.markdown("### 🏥 Injury Report")
    
    for team, players in injuries_data.items():
        if not players:
            continue
        
        # Check if this is home or away team
        team_lower = team.lower()
        is_home = any(part in team_lower for part in home_team.lower().split())
        is_away = any(part in team_lower for part in away_team.lower().split())
        
        if not is_home and not is_away:
            continue
        
        team_label = "🏠 " + home_team if is_home else "✈️ " + away_team
        
        # Count key players out
        key_out = [p for p in players if 'will not play' in p.get('status', '').lower()]
        
        if players:
            with st.expander(f"{team_label} - {len(players)} injured player(s)", expanded=bool(key_out)):
                for player in players:
                    status = player.get('status', '')
                    status_emoji = "🔴" if 'will not play' in status.lower() else ("🟡" if 'doubt' in status.lower() else "🟢")
                    injury = player.get('injury', '')
                    games = player.get('games_played', '')
                    
                    st.markdown(f"""
                    **{status_emoji} {player.get('name', 'Unknown')}**
                    - Injury: {injury if injury else 'Not specified'}
                    - Games Played: {games if games else 'N/A'}
                    - Status: {status}
                    """)
    
    # Generate narrative summary
    injury_narrative = generate_injury_narrative(home_team, away_team, injuries_data)
    if injury_narrative:
        st.markdown(f"""
        <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #6366f1; margin: 1rem 0;">
            <h4 style="margin: 0 0 0.5rem 0; color: #6366f1;">📋 Injury Impact Analysis</h4>
            <p style="margin: 0; color: #94a3b8;">{injury_narrative}</p>
        </div>
        """, unsafe_allow_html=True)


def generate_injury_narrative(home_team, away_team, injuries_data):
    """Generate a narrative summary about injury impact."""
    narratives = []
    
    for team, players in injuries_data.items():
        if not players:
            continue
        
        team_lower = team.lower()
        is_home = any(part in team_lower for part in home_team.lower().split())
        is_away = any(part in team_lower for part in away_team.lower().split())
        
        if not is_home and not is_away:
            continue
        
        team_label = home_team if is_home else away_team
        
        key_out = [p for p in players if 'will not play' in p.get('status', '').lower()]
        
        if key_out:
            player_names = [p.get('name', 'Unknown') for p in key_out[:3]]
            if len(player_names) == 1:
                narratives.append(f"{team_label} will be without key player {player_names[0]}")
            elif len(player_names) == 2:
                narratives.append(f"{team_label} will be without {player_names[0]} and {player_names[1]}")
            else:
                narratives.append(f"{team_label} will be without {player_names[0]}, {player_names[1]} and {len(key_out)-2} more")
        elif players:
            injured_count = len(players)
            narratives.append(f"{team_label} has {injured_count} injured player(s) but all are potentially available")
    
    return " ".join(narratives) if narratives else ""

# ════════════════════════════════════════════════════════════════════════
# MAIN APP
# ════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; margin: 0; background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            ⚽ Football AI Predictor
        </h1>
        <p style="color: #94a3b8; font-size: 1.25rem; margin-top: 0.5rem;">
            ML-powered predictions using historical data & form analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabbed interface
    tab_predict, tab_stats, tab_train, tab_results, tab_leagues, tab_historical, tab_settings = st.tabs([
        "🔮 Predict", 
        "📊 Statistics", 
        "🧠 Train Model", 
        "✅ Add Result", 
        "🏆 Leagues",
        "📅 Historical Data",
        "⚙️ Settings"
    ])
    
    # ════════════════════════════════════════════════════════════════════════
    # PREDICT TAB
    # ════════════════════════════════════════════════════════════════════════
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
            predict_btn = st.button("🔮 Predict", width='stretch')
        
        if predict_btn and match_url:
            with st.spinner("🤖 AI is analyzing match data..."):
                result = system.predict_match(match_url)
                
                if 'error' in result:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    display_prediction_result(result)
    
    # ════════════════════════════════════════════════════════════════════════
    # STATS TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_stats:
        display_stats()
    
    # ════════════════════════════════════════════════════════════════════════
    # TRAIN TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_train:
        display_train()
    
    # ════════════════════════════════════════════════════════════════════════
    # RESULTS TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_results:
        display_results()
    
    # ════════════════════════════════════════════════════════════════════════
    # LEAGUES TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_leagues:
        display_leagues_tab()
    
    # ════════════════════════════════════════════════════════════════════════
    # HISTORICAL DATA TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_historical:
        display_historical_data()
    
    # ════════════════════════════════════════════════════════════════════════
    # SETTINGS TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_settings:
        display_settings()

# ════════════════════════════════════════════════════════════════════════════════
# PREDICTION DISPLAY
# ════════════════════════════════════════════════════════════════════════════════

def display_prediction_result(result):
    """Display complete prediction results in a modern format."""
    md = result.get('match_data', {})
    pred = result.get('prediction', {})
    feats = result.get('features', {})
    analysis = result.get('analysis', {})
    conv = result.get('convergence', {})
    
    teams = md.get('teams', {})
    # Note: scraper stores info as 'match_info'
    info = md.get('match_info', md.get('info', {}))
    home = teams.get('home', 'Home')
    away = teams.get('away', 'Away')
    
    # Match header card
    st.markdown(f"""
    <div class="card" style="text-align: center; background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));">
        <div style="font-size: 1.25rem; color: #94a3b8;">vs</div>
        <h2 style="font-size: 2.5rem; margin: 0.5rem 0;">{home} ⚔️ {away}</h2>
        <p style="color: #94a3b8;">
            📅 {info.get('date', 'N/A')} • 🏆 {info.get('league', 'Unknown League')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # FINAL RECOMMENDATION (MOST PROMINENT)
    # ============================================
    rp = pred.get('result', {})
    op = pred.get('over_under', {})
    
    rmap = {'1': f'{home} Win', 'X': 'Draw', '2': f'{away} Win'}
    
    # Build recommendations list and sort by confidence
    recommendations = [
        {'type': 'Match Result', 'label': rmap.get(rp.get('prediction', '?'), '?'), 'confidence': rp.get('confidence', 0) * 100, 'prediction': rp.get('prediction', ''), 'ou_type': ''},
        {'type': 'Over/Under 2.5', 'label': f"{op.get('prediction', 'Over')} 2.5 Goals", 'confidence': op.get('confidence', 0) * 100, 'prediction': op.get('prediction', ''), 'ou_type': 'over_under'}
    ]
    
    # Sort by confidence (highest first)
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Top recommendation
    top_rec = recommendations[0]
    top_conf = top_rec['confidence']
    
    # Confidence level
    if top_conf >= 60:
        conf_level = "🟢 HIGH"
    elif top_conf >= 40:
        conf_level = "🟡 MEDIUM"
    else:
        conf_level = "🔴 LOW"
    
    st.markdown(f"""
    <div class="card" style="text-align: center; background: linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(5, 150, 105, 0.3)); border: 2px solid #10b981;">
        <h3 style="margin: 0; color: #94a3b8; font-size: 1rem;">🎯 FINAL RECOMMENDATION</h3>
        <h1 style="margin: 0.5rem 0; font-size: 2.5rem; color: #10b981;">{top_rec['label']}</h1>
        <p style="margin: 0; color: #94a3b8;">Confidence: {conf_level} ({top_conf:.0f}%)</p>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; color: #94a3b8;">{top_rec['type']} • #{recommendations.index(top_rec) + 1} Choice</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show all recommendations in order
    st.markdown("### 📊 All Recommendations")
    rec_cols = st.columns(len(recommendations))
    for i, rec in enumerate(recommendations):
        with rec_cols[i]:
            emoji = "🎯" if i == 0 else "📈"
            st.markdown(f"""
            <div class="card" style="text-align: center; padding: 1rem;">
                <p style="margin: 0; font-size: 0.9rem; color: #94a3b8;">{emoji} {rec['type']}</p>
                <p style="margin: 0.5rem 0; font-size: 1.2rem; font-weight: bold;">{rec['label']}</p>
                <p style="margin: 0; color: #10b981;">{rec['confidence']:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main predictions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3 style="text-align: center; margin-bottom: 1rem;">🏆 Match Result</h3>
        </div>
        """, unsafe_allow_html=True)
        
        rmap = {'1': f'{home} Win', 'X': 'Draw', '2': f'{away} Win'}
        pred_result = rmap.get(rp.get('prediction', '?'), '?')
        
        conf = rp.get('confidence', 0) * 100
        conf_color = "green" if conf >= 55 else ("yellow" if conf >= 40 else "red")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            render_prediction_badge(f"{home}\nWin", min(conf, 100), "green" if rp.get('prediction') == '1' else "gray")
        with col_b:
            render_prediction_badge("Draw", rp.get('probabilities', {}).get('X', 0) * 100, "yellow")
        with col_c:
            render_prediction_badge(f"{away}\nWin", rp.get('probabilities', {}).get('2', 0) * 100, "red" if rp.get('prediction') == '2' else "gray")
        
        st.markdown(f"**Recommended:** {pred_result} • Confidence: {conf:.0f}%")
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3 style="text-align: center; margin-bottom: 1rem;">📈 Over/Under 2.5</h3>
        </div>
        """, unsafe_allow_html=True)
        
        ou_pred = op.get('prediction', 'Over')
        ou_conf = op.get('confidence', 0) * 100
        
        col_o, col_u = st.columns(2)
        with col_o:
            render_prediction_badge("Over 2.5", min(ou_conf, 100), "green" if ou_pred == 'Over' else "gray")
        with col_u:
            render_prediction_badge("Under 2.5", 100 - ou_conf, "red" if ou_pred == 'Under' else "gray")
        
        st.markdown(f"**Recommended:** {ou_pred} 2.5 • Confidence: {ou_conf:.0f}%")
    
    # Injuries
    injuries_data = result.get('injuries', {})
    if injuries_data:
        display_injury_info(home, away, injuries_data)
    
    # Analysis
    st.markdown("---")
    st.markdown("### 📊 Detailed Analysis")
    
    key_factors = analysis.get('key_factors', [])
    if key_factors:
        st.markdown("#### 🔑 Key Factors")
        for factor in key_factors[:5]:
            st.markdown(f"* {factor}")
    
    # Convergence
    if conv:
        st.markdown("### 🔗 Convergence Analysis")
        col_conv1, col_conv2 = st.columns(2)
        
        with col_conv1:
            st.markdown("#### Match Result")
            ml_pred = conv.get('ml_result', '?')
            w_pred = conv.get('weighted_result', '?')
            agreement = conv.get('result_agreement', False)
            convergence = conv.get('result_convergence', 0)
            
            st.markdown(f"* ML Prediction: {ml_pred}")
            st.markdown(f"* Weighted Prediction: {w_pred}")
            st.markdown(f"* Agreement: {'✅ AGREE' if agreement else '⚠️ DISAGREE'}")
            st.markdown(f"* Convergence: {convergence:.0f}%")
        
        with col_conv2:
            st.markdown("#### Over/Under")
            ml_ou = conv.get('ml_ou', '?')
            w_ou = conv.get('weighted_ou', '?')
            ou_agreement = conv.get('ou_agreement', False)
            ou_convergence = conv.get('ou_convergence', 0)
            
            st.markdown(f"* ML Prediction: {ml_ou}")
            st.markdown(f"* Weighted Prediction: {w_ou}")
            st.markdown(f"* Agreement: {'✅ AGREE' if ou_agreement else '⚠️ DISAGREE'}")
            st.markdown(f"* Convergence: {ou_convergence:.0f}%")
    
    # Features
    if feats:
        st.markdown("### 📈 Feature Values")
        feat_cols = st.columns(4)
        feature_list = list(feats.items())[:8]
        for i, (name, value) in enumerate(feature_list):
            with feat_cols[i % 4]:
                st.metric(name.replace('_', ' ').title(), f"{value:.3f}" if isinstance(value, float) else value)


# ════════════════════════════════════════════════════════════════════════════════
# STATS TAB
# ════════════════════════════════════════════════════════════════════════════════

def display_stats():
    """Display system statistics."""
    try:
        # Get training stats
        from model_stats import get_training_stats
        training_stats = get_training_stats()
        
        # Get prediction stats
        stats = system.get_statistics()
        
        # Training metrics
        train_examples = training_stats.get('training_count', 0)
        train_acc = training_stats.get('train_result_accuracy', 0) * 100
        test_acc = training_stats.get('test_result_accuracy')
        test_acc_display = f"{test_acc:.1f}%" if test_acc is not None else "N/A"
        
        st.markdown("""
        <div class="card" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));">
            <h2 style="text-align: center; margin-bottom: 1rem;">📊 Model Performance</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <h1 style="font-size: 2.5rem; margin: 0;">📚</h1>
                <p style="color: #94a3b8; margin: 0;">Training Examples</p>
                <h2 style="margin: 0.5rem 0 0 0;">{train_examples}</h2>
            </div>
            """.format(train_examples), unsafe_allow_html=True)
        
        with col2:
            acc_color = "#10b981" if (test_acc or train_acc) >= 50 else "#f59e0b" if (test_acc or train_acc) >= 40 else "#ef4444"
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <h1 style="font-size: 2.5rem; margin: 0;">🎯</h1>
                <p style="color: #94a3b8; margin: 0;">Test Accuracy</p>
                <h2 style="margin: 0.5rem 0 0 0; color: {acc_color};">{test_acc_display}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <h1 style="font-size: 2.5rem; margin: 0;">📈</h1>
                <p style="color: #94a3b8; margin: 0;">Train Accuracy</p>
                <h2 style="margin: 0.5rem 0 0 0;">{train_acc:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            total_preds = stats.get('total_predictions', 0)
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <h1 style="font-size: 2.5rem; margin: 0;">🔮</h1>
                <p style="color: #94a3b8; margin: 0;">Total Predictions</p>
                <h2 style="margin: 0.5rem 0 0 0;">{total_preds}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Show accuracy interpretation
        if test_acc is not None:
            if test_acc >= 50:
                st.success(f"✅ Model accuracy ({test_acc:.1f}%) is good! Better than random guessing.")
            elif test_acc >= 40:
                st.warning(f"⚠️ Model accuracy ({test_acc:.1f}%) is moderate. More data may help.")
            else:
                st.error(f"❌ Model accuracy ({test_acc:.1f}%) needs improvement.")
        
        st.markdown("### 📊 Prediction Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            accuracy = stats.get('accuracy', 0) * 100
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <h1 style="font-size: 3rem; margin: 0;">🎯</h1>
                <p style="color: #94a3b8; margin: 0;">Live Accuracy</p>
                <h2 style="margin: 0.5rem 0 0 0;">{accuracy:.1f}%</h2>
            </div>
            """.format(accuracy), unsafe_allow_html=True)
        
        with col2:
            correct = stats.get('correct_predictions', 0)
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <h1 style="font-size: 3rem; margin: 0;">✅</h1>
                <p style="color: #94a3b8; margin: 0;">Correct Predictions</p>
                <h2 style="margin: 0.5rem 0 0 0;">{correct}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total = stats.get('total_predictions', 0)
            st.markdown(f"""
            <div class="card" style="text-align: center;">
                <h1 style="font-size: 3rem; margin: 0;">📈</h1>
                <p style="color: #94a3b8; margin: 0;">Total Predictions</p>
                <h2 style="margin: 0.5rem 0 0 0;">{total}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🏆 League Statistics")
        
        league_stats = stats.get('league_stats', {})
        if league_stats:
            data = []
            for league, s in league_stats.items():
                data.append({
                    'League': league,
                    'Matches': s.get('count', 0),
                    'Accuracy': f"{s.get('accuracy', 0) * 100:.1f}%"
                })
            
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, width='stretch')
        
        if 'recent_predictions' in stats:
            st.markdown("### 🕐 Recent Predictions")
            recent = stats['recent_predictions'][-10:]
            for p in recent:
                emoji = "✅" if p.get('correct') else "❌" if p.get('correct') is False else "⏳"
                st.markdown(f"{emoji} {p.get('home', '?')} vs {p.get('away', '?')}: {p.get('prediction', '?')} ({p.get('confidence', 0) * 100:.0f}%)")
    
    except Exception as e:
        st.error(f"Error loading stats: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# TRAIN TAB
# ════════════════════════════════════════════════════════════════════════════════

def display_train():
    """Display model training interface."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🧠 Train Models")
        
        st.markdown("""
        <div class="card">
            <p style="color: #94a3b8;">Training the models will use all available historical data with known results to improve prediction accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Train Model Now", width='stretch'):
            with st.spinner("Training in progress..."):
                result = system.train_model()
                
                if result.get('success'):
                    st.success(f"✅ Models trained successfully!")
                    st.info(f"📊 Train Accuracy: {result.get('train_accuracy', 0):.1f}%")
                    st.info(f"📈 Test Accuracy: {result.get('test_accuracy', 0):.1f}% (on {result.get('test_examples', 0)} examples)")
                elif 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    st.warning("⚠️ No new data to train on (or training was skipped")
    
    with col2:
        st.markdown("### 📊 Training Info")
        
        # Get training statistics from storage
        stats = system.get_statistics()
        total_examples = stats.get('training_examples', 0)
        total_matches = stats.get('total_matches', 0)
        total_results = stats.get('total_results', 0)
        
        st.markdown(f"* **Total Matches:** {total_matches}")
        st.markdown(f"* **Matches with Results:** {total_results}")
        st.markdown(f"* **Training Examples:** {total_examples}")
        
        # Show top 5 leagues by example count
        training_data = system.storage.get_training_data()
        if training_data:
            # Get top 5 leagues by example count
            league_counts = [(e.get('league', 'Unknown'), len(e.get('examples', []))) for e in training_data]
            league_counts.sort(key=lambda x: x[1], reverse=True)
            top_leagues = [(league, count) for league, count in league_counts if count > 0][:5]
            
            if top_leagues:
                st.markdown("### 📈 Top 5 Leagues by Training Examples")
                for league, count in top_leagues:
                    st.markdown(f"* **{league}**: {count} examples")
            else:
                st.info("No training data available yet.")


# ════════════════════════════════════════════════════════════════════════════════
# RESULTS TAB
# ════════════════════════════════════════════════════════════════════════════════

def display_results():
    """Display match results interface."""
    st.markdown("### ✅ Add Match Result")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        result_url = st.text_input(
            "Match URL",
            placeholder="https://www.forebet.com/en/football/matches/..."
        )
    
    with col2:
        add_btn = st.button("➕ Add Result", width='stretch')
    
    if add_btn and result_url:
        result = system.scraper.extract_actual_result(result_url)
        
        if result and result.get('home_score') is not None:
            st.success(f"✅ Result: {result['home_score']} - {result['away_score']}")
            
            col_h, col_a = st.columns(2)
            with col_h:
                home_score = st.number_input("Home Score", value=result['home_score'], min_value=0)
            with col_a:
                away_score = st.number_input("Away Score", value=result['away_score'], min_value=0)
            
            if st.button("💾 Save Result"):
                success = system.record_result(result_url, home_score, away_score)
                
                if success:
                    st.success("✅ Result saved successfully!")
                else:
                    st.error("❌ Failed to save result")
        else:
            st.error("❌ Could not extract result - match may not have been played yet.")
    
    st.markdown("---")
    
    # Show recent results
    try:
        with open('data/results.json', 'r') as f:
            results = json.load(f)
        
        st.markdown("### 📊 Recent Results")
        
        if results:
            recent = list(results.items())[-10:][::-1]
            data = []
            for url, r in recent:
                teams = r.get('teams', {})
                data.append({
                    'Home': teams.get('home', '?'),
                    'Away': teams.get('away', '?'),
                    'Score': f"{r.get('home_score', '?')}-{r.get('away_score', '?')}",
                    'Date': r.get('date', '')[:10] if r.get('date') else ''
                })
            
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, width='stretch')
    except Exception as e:
        st.info("No results available yet.")


# ════════════════════════════════════════════════════════════════════════════════
# LEAGUES TAB
# ════════════════════════════════════════════════════════════════════════════════

def load_league_mapping():
    """Load league code to name mapping from file."""
    try:
        with open('data/league_mapping.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}


def save_league(code: str, name: str) -> bool:
    """Save a league code to the mapping."""
    mapping = load_league_mapping()
    
    if code in mapping:
        return False
    
    mapping[code] = name
    
    try:
        with open('data/league_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        return True
    except:
        return False


def upload_leagues_json(json_data: dict) -> int:
    """Upload leagues from JSON data."""
    mapping = load_league_mapping()
    count = 0
    
    for code, name in json_data.items():
        if code not in mapping:
            mapping[code] = name
            count += 1
    
    if count > 0:
        with open('data/league_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    return count


def display_leagues_tab():
    """Display league management interface."""
    st.markdown("### 🏆 League Management")
    
    st.markdown("""
    <div class="card">
        <p style="color: #94a3b8;">
            Manage league code mappings. When you encounter an unknown league code during prediction,
            enter it here with the full league name for future reference.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show current leagues
    st.markdown("#### 📋 Current Leagues")
    
    mapping = load_league_mapping()
    st.markdown(f"**Total leagues mapped:** {len(mapping)}")
    
    if mapping:
        # Convert to DataFrame for display
        data = []
        for code, info in mapping.items():
            if isinstance(info, dict):
                name = info.get('league_name', info.get('name', str(info)))
                country = info.get('country', '')
            else:
                name = str(info)
                country = ''
            
            # Try to extract country from name if not present
            if not country and name:
                # Common country prefixes in league names
                country_prefixes = ['England', 'Spain', 'Italy', 'Germany', 'France', 'Netherlands', 
                                   'Portugal', 'Turkey', 'Greece', 'Russia', 'Scotland', 'Austria',
                                   'Switzerland', 'Belgium', 'Poland', 'Ukraine', 'Czech', 'Hungary',
                                   'Romania', 'Denmark', 'Sweden', 'Norway', 'Finland', 'Ireland',
                                   'Brazil', 'Argentina', 'USA', 'Mexico', 'Japan', 'South Korea',
                                   'China', 'Australia', 'Saudi', 'UAE', 'Qatar', 'Kenya', 'Uganda',
                                           'Tanzania', 'South Africa', 'Egypt', 'Morocco', 'Tunisia']
                for prefix in country_prefixes:
                    if name.lower().startswith(prefix.lower()):
                        country = prefix
                        break
            
            data.append({
                'Code': code,
                'Name': name,
                'Country': country if country else 'Unknown'
            })
        
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, width='stretch')
    
    st.markdown("---")
    
    # Add single league
    st.markdown("#### ➕ Add Single League")
    
    col1, col2 = st.columns(2)
    with col1:
        code = st.text_input("League Code (5 digits)", placeholder="24236")
    with col2:
        name = st.text_input("League Name", placeholder="Bahrain Premier League")
    
    if st.button("💾 Save League"):
        if code and name:
            if save_league(code.strip(), name.strip()):
                st.success(f"✅ Saved: {name}")
                st.rerun()
            else:
                st.warning(f"⚠️ League {code} already exists")
        else:
            st.error("Please enter both code and name")
    
    st.markdown("---")
    
    # Upload multiple leagues
    st.markdown("#### 📤 Upload Multiple Leagues")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Option 1: Upload JSON file**")
        league_file = st.file_uploader("Choose a JSON file", type=['json'], key="league_json")
        
        if league_file is not None:
            try:
                json_data = json.load(league_file)
                st.write(f"Found {len(json_data)} entries")
                
                if st.button("📤 Upload All"):
                    count = upload_leagues_json(json_data)
                    if count > 0:
                        st.success(f"✅ Uploaded {count} leagues!")
                        st.rerun()
                    else:
                        st.warning("No new leagues to add")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col2:
        st.markdown("**Option 2: Paste JSON**")
        json_text = st.text_area("Paste JSON here", placeholder='{"24236": "Bahrain Premier League", "24206": "Kenya Premier League"}', height=100)
        
        if json_text:
            try:
                json_data = json.loads(json_text)
                st.write(f"Found {len(json_data)} entries")
                
                if st.button("📤 Upload Pasted"):
                    count = upload_leagues_json(json_data)
                    if count > 0:
                        st.success(f"✅ Uploaded {count} leagues!")
                        st.rerun()
                    else:
                        st.warning("No new leagues to add")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    
    st.markdown("---")
    
    # Download current leagues
    st.markdown("#### 📥 Download League List")
    
    if st.button("📥 Download Database"):
        mapping = load_league_mapping()
        st.download_button(
            label="💾 Download league_mapping.json",
            data=json.dumps(mapping, indent=2, ensure_ascii=False),
            file_name="league_mapping.json",
            mime="application/json"
        )


# ════════════════════════════════════════════════════════════════════════════════
# HISTORICAL DATA TAB
# ════════════════════════════════════════════════════════════════════════════════

def display_historical_data():
    """Display interface for adding historical match data."""
    st.markdown("### 📅 Add Historical Match Data")
    
    st.markdown("""
    <div class="card">
        <p style="color: #94a3b8;">
            Add historical match data by pasting a Forebet predictions page URL for a specific date.
            This will extract all match URLs, save league information, and organize data by date.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        date_url = st.text_input(
            "Forebet Date URL",
            placeholder="https://www.forebet.com/en/football-predictions/predictions-1x2/2026-02-19",
            help="URL format: https://www.forebet.com/en/football-predictions/predictions-1x2/YYYY-MM-DD"
        )
    
    with col2:
        extract_btn = st.button("📅 Extract Matches", width='stretch')
    
    if extract_btn and date_url:
        # Extract date from URL using simple pattern
        date_match = re.search(r'/(\d{4}-\d{2}-\d{2})/?$', date_url)
        if not date_match:
            st.error("Invalid URL format. Use: https://www.forebet.com/en/football-predictions/predictions-1x2/YYYY-MM-DD")
            return
        
        date_str = date_match.group(1)
        
        with st.spinner(f"Extracting matches for {date_str}..."):
            try:
                # Initialize scraper
                scraper = HistoricalForebetScraper()
                
                # Scrape matches
                matches = scraper.scrape_historical_matches(date_str)
                
                if not matches:
                    st.warning("No matches found for this date.")
                    return
                
                # Save to JSON file with date
                os.makedirs('data', exist_ok=True)
                filename = f"data/historical_matches_{date_str}.json"
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(matches, f, indent=2, default=str)
                
                # Display results
                st.success(f"Extracted {len(matches)} matches for {date_str}")
                
                # Show summary
                with_results = sum(1 for m in matches if m.get('has_result'))
                st.info(f"Matches with results: {with_results}")
                
                # Show match list
                st.markdown("#### Extracted Matches")
                
                # Create a DataFrame for display
                match_data = []
                for m in matches:
                    if m.get('home_team') and m.get('away_team'):
                        row = {
                            'Home': m.get('home_team', ''),
                            'Away': m.get('away_team', ''),
                            'League': m.get('short_code', m.get('league_name', '')),
                            'Result': f"{m.get('home_score', '?')}-{m.get('away_score', '?')}" if m.get('has_result') else 'Pending',
                            'URL': m.get('url', '')
                        }
                        match_data.append(row)
                
                if match_data:
                    df = pd.DataFrame(match_data)
                    st.dataframe(df, width='stretch')
                
                # Show leagues found
                leagues = set()
                for m in matches:
                    if m.get('league_name'):
                        leagues.add(m.get('league_name'))
                    if m.get('short_code'):
                        leagues.add(m.get('short_code'))
                
                if leagues:
                    st.markdown("#### Leagues Found")
                    for league in sorted(leagues):
                        st.markdown(f"* {league}")
                
                st.markdown(f"""
                <div class="card" style="background: rgba(16, 185, 129, 0.2); border-color: #10b981;">
                    <h4 style="margin: 0 0 0.5rem 0; color: #10b981;">Data Saved</h4>
                    <p style="margin: 0; color: #94a3b8;">
                        File: <code>{filename}</code><br>
                        Matches: {len(matches)}<br>
                        With Results: {with_results}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # Show existing historical data files
    st.markdown("#### Existing Historical Data Files")
    
    # Find all historical data files
    historical_files = sorted(glob.glob('data/historical_matches_*.json'), reverse=True)
    
    if historical_files:
        file_data = []
        for f in historical_files:
            filename = os.path.basename(f)
            # Extract date from filename
            date_match = re.search(r'historical_matches_(\d{4}-\d{2}-\d{2})\.json', filename)
            if date_match:
                date_str = date_match.group(1)
                # Count matches in file
                try:
                    with open(f, 'r', encoding='utf-8') as fp:
                        data = json.load(fp)
                        with_results = sum(1 for m in data if m.get('has_result'))
                        file_data.append({
                            'Date': date_str,
                            'File': filename,
                            'Total': len(data),
                            'Results': with_results,
                            'Path': f
                        })
                except:
                    pass
        
        if file_data:
            df = pd.DataFrame(file_data)
            st.dataframe(df[['Date', 'Total', 'Results']], width='stretch')
            
            # Download buttons
            st.markdown("##### Download Files")
            cols = st.columns(3)
            for i, row in enumerate(file_data[:6]):
                with cols[i % 3]:
                    try:
                        with open(row['Path'], 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        st.download_button(
                            label=f"📥 {row['Date']}",
                            data=file_content,
                            file_name=row['File'],
                            mime='application/json'
                        )
                    except:
                        pass
    else:
        st.info("No historical data files found.")


# ════════════════════════════════════════════════════════════════════════════════
# SETTINGS TAB
# ════════════════════════════════════════════════════════════════════════════════

def display_settings():
    """Display settings."""
    st.markdown("### ⚙️ Settings")
    
    st.markdown("""
    <div class="card">
        <h4>📁 Data Directory</h4>
        <code style="background: rgba(99, 102, 241, 0.2); padding: 0.5rem; border-radius: 8px;">data/</code>
    </div>
    
    <div class="card">
        <h4>🔧 Available Commands</h4>
        <ul style="color: #94a3b8;">
            <li><code>fb <url></code> - Predict a match</li>
            <li><code>fb predict --url <url></code> - Predict with options</li>
            <li><code>fb result <url> <score></code> - Add result</li>
            <li><code>fb train</code> - Train model</li>
            <li><code>fb stats</code> - View statistics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h4>🌐 Deployment</h4>
        <p style="color: #94a3b8;">Deploy on <a href="https://share.streamlit.io" target="_blank">Streamlit Community Cloud</a> or <a href="https://huggingface.co/spaces" target="_blank">HuggingFace Spaces</a> for free.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
