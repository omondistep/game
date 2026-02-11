#!/usr/bin/env python3
"""
Football Prediction System - Modern Streamlit App
Deploy on: Streamlit Community Cloud or HuggingFace Spaces

Features:
- Match Predictions with ML & Statistical Analysis
- System Statistics & Training Data
- Model Training Interface
- Match Results Entry
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json

# Import prediction system
from football_prediction_system import FootballPredictionSystem

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
        st.markdown("""
        <div style="background: rgba(99, 102, 241, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #6366f1; margin: 1rem 0;">
            <h4 style="margin: 0 0 0.5rem 0; color: #6366f1;">📋 Injury Impact Analysis</h4>
            <p style="margin: 0; color: #94a3b8;">""" + injury_narrative + """</p>
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
    st.markdown("""
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
    tab_predict, tab_stats, tab_train, tab_results, tab_leagues, tab_settings = st.tabs([
        "🔮 Predict", 
        "📊 Statistics", 
        "🧠 Train Model", 
        "✅ Add Result", 
        "🏆 Leagues",
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
            predict_btn = st.button("🔮 Predict", use_container_width=True)
        
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
    # ═══════════════════════════════════════════════════════════════════════=
    with tab_leagues:
        display_leagues_tab()
    
    # ════════════════════════════════════════════════════════════════════════
    # SETTINGS TAB
    # ════════════════════════════════════════════════════════════════════════
    with tab_settings:
        display_settings()

# ════════════════════════════════════════════════════════════════════════
# PREDICTION DISPLAY
# ════════════════════════════════════════════════════════════════════════

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
        
        col_a, col_b = st.columns(2)
        with col_a:
            render_prediction_badge("Over 2.5", min(ou_conf, 100), "green" if ou_pred == 'Over' else "gray")
        with col_b:
            render_prediction_badge("Under 2.5", 100 - ou_conf, "yellow" if ou_pred == 'Under' else "gray")
        
        st.markdown(f"**Recommended:** {ou_pred} 2.5 • Confidence: {ou_conf:.0f}%")
    
    # Convergence Analysis
    st.markdown("### 🔗 Convergence Analysis")
    st.markdown("How well do ML and Statistical models agree?")
    
    ml_r = conv.get('ml_result', '?')
    w_r = conv.get('weighted_result', '?')
    ml_conf = result.get('ml_prediction', {}).get('result', {}).get('confidence', 0) * 100
    w_conf = result.get('weighted_prediction', {}).get('result', {}).get('confidence', 0) * 100
    
    r_match = conv.get('result_match', False)
    
    ml_o = conv.get('ml_ou', '?')
    w_o = conv.get('weighted_ou', '?')
    o_match = conv.get('ou_match', False)
    
    # Calculate convergence
    ml_probs = result.get('ml_prediction', {}).get('result', {}).get('probabilities', {})
    w_probs = result.get('weighted_prediction', {}).get('result', {}).get('probabilities', {})
    
    if ml_probs and w_probs:
        prob_diff = sum(abs(ml_probs.get(k, 0) - w_probs.get(k, 0)) for k in ['1', 'X', '2']) / 3 * 100
        r_deg = max(0, 100 - prob_diff)
    else:
        r_deg = 0
    
    ml_ou_probs = result.get('ml_prediction', {}).get('over_under', {}).get('probabilities', {})
    w_ou_probs = result.get('weighted_prediction', {}).get('over_under', {}).get('probabilities', {})
    
    if ml_ou_probs and w_ou_probs:
        ou_diff = sum(abs(ml_ou_probs.get(k, 0) - w_ou_probs.get(k, 0)) for k in ['Over', 'Under']) / 2 * 100
        o_deg = max(0, 100 - ou_diff)
    else:
        o_deg = 0
    
    c = sum([r_match, o_match])
    conf_level = "🟢 HIGH" if c == 2 else ("🟡 MEDIUM" if c == 1 else "🔴 LOW")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ML Model", f"{ml_r}", f"{ml_conf:.0f}% conf")
    with col2:
        st.metric("Statistical Model", f"{w_r}", f"{w_conf:.0f}% conf")
    with col3:
        st.metric("Agreement", "✅ Agree" if r_match else "⚠️ Disagree", f"{r_deg:.0f}%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ML O/U", ml_o)
    with col2:
        st.metric("Statistical O/U", w_o)
    with col3:
        st.metric("O/U Agreement", "✅ Agree" if o_match else "⚠️ Disagree", f"{o_deg:.0f}%")
    
    st.info(f"**Overall Confidence:** {conf_level}")
    
    # Injury Report
    injuries_data = result.get('injuries', {})
    if injuries_data:
        display_injury_info(home, away, injuries_data)
    
    # Standings
    st.markdown("### 📊 Standings")
    standings = md.get('standings', {})
    
    if isinstance(standings, dict):
        col_h, col_a = st.columns(2)
        
        with col_h:
            sh = standings.get('home', {})
            if isinstance(sh, dict):
                st.markdown(f"""
                <div class="card">
                    <h4 style="color: #6366f1;">#{sh.get('position', '?')} {home}</h4>
                    <p style="margin: 0;"><strong>{sh.get('points', '?')} pts</strong> | W:{sh.get('won','?')} D:{sh.get('drawn','?')} L:{sh.get('lost','?')}</p>
                    <p style="margin: 0; color: #94a3b8;">GF:{sh.get('gf','?')} GA:{sh.get('ga','?')} GD:{sh.get('gd','?')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col_a:
            sa = standings.get('away', {})
            if isinstance(sa, dict):
                st.markdown(f"""
                <div class="card">
                    <h4 style="color: #a855f7;">#{sa.get('position', '?')} {away}</h4>
                    <p style="margin: 0;"><strong>{sa.get('points', '?')} pts</strong> | W:{sa.get('won','?')} D:{sa.get('drawn','?')} L:{sa.get('lost','?')}</p>
                    <p style="margin: 0; color: #94a3b8;">GF:{sa.get('gf','?')} GA:{sa.get('ga','?')} GD:{sa.get('gd','?')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Head to Head
    h2h = md.get('head_to_head', {})
    if h2h.get('matches'):
        st.markdown("### 🤝 Head to Head")
        h2h_matches = h2h['matches'][:6]
        
        h2h_data = []
        for m in h2h_matches:
            h2h_data.append({
                'Date': m.get('date', '?'),
                'Home': m.get('home_team', '?'),
                'Score': f"{m.get('home_score','?')}-{m.get('away_score','?')}",
                'Away': m.get('away_team', '?'),
                'Comp': m.get('competition', '')
            })
        
        if h2h_data:
            df = pd.DataFrame(h2h_data)
            st.table(df)
        
        s = h2h.get('summary', {})
        if s:
            st.markdown(f"""
            <div style="display: flex; gap: 2rem; justify-content: center;">
                <span style="color: #6366f1;">{home}: {s.get('home_wins',0)} ({s.get('home_win_pct',0)}%)</span>
                <span style="color: #f59e0b;">Draws: {s.get('draws',0)} ({s.get('draw_pct',0)}%)</span>
                <span style="color: #a855f7;">{away}: {s.get('away_wins',0)} ({s.get('away_win_pct',0)}%)</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Key Factors
    if analysis.get('key_factors'):
        st.markdown("### 🔑 Key Factors")
        for f in analysis['key_factors']:
            st.write(f"• {f}")
    
    # Value Bets
    if analysis.get('value_bets'):
        st.markdown("### 💰 Value Bets")
        for vb in analysis['value_bets']:
            lbl = {'1': f'{home} Win', 'X': 'Draw', '2': f'{away} Win'}.get(vb['outcome'], vb['outcome'])
            st.success(f"✓ **{lbl}**: Market {vb['market']:.2f} vs Fair {vb['fair']:.2f} (+{vb['value_pct']:.1f}% value)")
    
    # Model info
    ml_pred_info = result.get('ml_prediction', {})
    if ml_pred_info.get('prediction_method') in ['ml', 'league_ml']:
        league = info.get('league', '')
        st.info(f"✓ **ML Model Trained** - Using learned patterns from {league}")

# ════════════════════════════════════════════════════════════════════════
# STATS DISPLAY
# ════════════════════════════════════════════════════════════════════════

def display_stats():
    """Display system statistics."""
    st.markdown("### 📊 System Statistics")
    
    try:
        stats = system.get_statistics()
        
        # Calculate leagues from training data
        training = system.storage.get_training_data()
        total_leagues = len(training)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Matches", stats.get('total_matches', 0))
        with col2:
            st.metric("Leagues", total_leagues)
        with col3:
            trained = "✅ Yes" if stats.get('model_trained') else "❌ No"
            st.metric("Model Trained", trained)
        with col4:
            st.metric("Predictions Made", stats.get('predictions_count', 0))
        
        # League distribution chart
        if training:
            st.markdown("### 📈 League Distribution")
            league_data = {}
            for entry in training:
                league_val = entry.get('league', 'Unknown')
                # Ensure league name is a string, not a dict
                if league_val is None:
                    league_name = 'Unknown'
                elif isinstance(league_val, str):
                    league_name = league_val
                else:
                    # If it's not a string or None, skip or use str()
                    league_name = str(league_val)
                examples = entry.get('examples', [])
                league_data[league_name] = league_data.get(league_name, 0) + len(examples)
            
            if league_data:
                df = pd.DataFrame(list(league_data.items()), columns=['League', 'Matches'])
                st.bar_chart(df.set_index('League'))
    
    except Exception as e:
        st.error(f"Error loading statistics: {e}")

# ════════════════════════════════════════════════════════════════════════
# TRAIN TAB
# ════════════════════════════════════════════════════════════════════════

def display_train():
    """Display model training interface."""
    st.markdown("### 🧠 Train Prediction Model")
    st.markdown("Train the ML model on historical match data for better predictions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>📋 Training Requirements</h4>
            <ul style="color: #94a3b8;">
                <li>Minimum 10 matches per league</li>
                <li>Historical data with known results</li>
                <li>League-specific patterns learned</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        try:
            stats = system.get_statistics()
            td = system.storage.get_league_training_data()
            st.metric("Available Training Data", len(td))
            
            if len(td) < 10:
                st.warning(f"⚠️ Need 10+ matches, have {len(td)}")
            else:
                st.success(f"✅ Ready to train ({len(td)} matches available)")
        except:
            st.error("Error loading training data")
    
    col1, col2 = st.columns(2)
    with col1:
        train_btn = st.button("🧠 Train Model", use_container_width=True)
    with col2:
        force_train = st.checkbox("Force retrain (overwrite existing model)")
    
    if train_btn:
        with st.spinner("Training model..."):
            result = system.train_model()
            
            if 'error' in result:
                st.error(f"❌ {result['error']}")
            else:
                st.success("✅ Model trained successfully!")
                st.json(result)

# ═══════════════════════════════════════════════════════════════════════=
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════=

def process_batch_results(urls):
    """Process multiple URLs to add match results."""
    results_summary = {
        'success': 0,
        'failed': 0,
        'not_played': 0,
        'details': []
    }
    
    progress_bar = st.progress(0)
    
    for i, url in enumerate(urls):
        progress_bar.progress((i + 1) / len(urls))
        
        with st.spinner(f"Processing {i+1}/{len(urls)}: {url[-20:] if len(url) > 20 else url}..."):
            result = system.add_match_result(url)
            
            if result:
                results_summary['success'] += 1
                results_summary['details'].append({'url': url, 'status': 'success'})
            else:
                # Check if match was already played or not
                results_summary['failed'] += 1
                results_summary['details'].append({'url': url, 'status': 'failed'})
    
    progress_bar.empty()
    
    # Display results
    st.markdown("### 📊 Batch Processing Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Success", results_summary['success'])
    with col2:
        st.metric("❌ Failed", results_summary['failed'])
    with col3:
        st.metric("📄 Total Processed", len(urls))
    
    if results_summary['success'] > 0:
        st.success(f"✅ Successfully added {results_summary['success']} results!")
    
    if results_summary['failed'] > 0:
        st.warning(f"⚠️ {results_summary['failed']} results could not be extracted")
    
    # Show details
    if results_summary['details']:
        with st.expander("View Details"):
            for detail in results_summary['details'][:20]:  # Show first 20
                status_emoji = "✅" if detail['status'] == 'success' else "❌"
                st.write(f"{status_emoji} {detail['url']}")

# ════════════════════════════════════════════════════════════════════════
# RESULTS TAB
# ════════════════════════════════════════════════════════════════════════

def display_results():
    """Display add result interface."""
    st.markdown("### ✅ Add Match Result")
    st.markdown("Update training data with actual match results from completed games.")
    
    # Single URL input
    st.markdown("#### Single Result")
    col1, col2 = st.columns([2, 1])
    with col1:
        url = st.text_input("Match URL", placeholder="https://www.forebet.com/...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        add_btn = st.button("✅ Add Result", use_container_width=True)
    
    if add_btn and url:
        with st.spinner("Adding result..."):
            result = system.add_match_result(url)
            if result:
                st.success("✅ Result added successfully!")
            else:
                st.error("❌ Could not extract result")
    
    st.markdown("---")
    
    # Batch upload
    st.markdown("#### Batch Upload")
    st.markdown("Upload a text file with one URL per line (e.g., results.txt) to add multiple results.")
    
    uploaded_file = st.file_uploader(
        "Choose a text file with URLs",
        type=['txt'],
        accept_multiple_files=False,
        key="batch_urls_file",
        help="Upload a file containing Forebet URLs, one per line"
    )
    
    if uploaded_file is not None:
        # Read URLs from file
        urls = uploaded_file.read().decode('utf-8').strip().split('\n')
        urls = [u.strip() for u in urls if u.strip() and u.strip().startswith('http')]
        
        if urls:
            st.info(f"📄 Found {len(urls)} URLs in uploaded file")
            
            # Process options
            col1, col2 = st.columns(2)
            with col1:
                max_urls = st.number_input("Max URLs to process", min_value=1, value=len(urls))
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                
            if st.button("🚀 Process Batch Results", use_container_width=True):
                process_batch_results(urls[:max_urls])

# ═══════════════════════════════════════════════════════════════════════=
# LEAGUES TAB
# ═══════════════════════════════════════════════════════════════════════=

def load_unknown_leagues():
    """Load unknown leagues from JSON file."""
    try:
        with open('data/unknown_leagues.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def load_league_mapping():
    """Load league mapping from JSON file."""
    try:
        with open('data/league_mapping.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_league(code, name):
    """Save a league name for a code."""
    # Update league_mapping.json
    mapping = load_league_mapping()
    if code not in mapping:
        mapping[code] = {
            'short_code': '',
            'league_name': name,
            'country': '',
            'source': 'manual'
        }
        with open('data/league_mapping.json', 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2)
    
    # Also update unknown_leagues.json for backwards compatibility
    unknown = load_unknown_leagues()
    unknown[code] = name
    with open('data/unknown_leagues.json', 'w') as f:
        json.dump(unknown, f, indent=2)
    return True

def upload_leagues_json(json_data):
    """Upload multiple leagues from JSON data."""
    mapping = load_league_mapping()
    unknown = load_unknown_leagues()
    count = 0
    
    for code, value in json_data.items():
        if isinstance(value, str):
            if code not in mapping:
                mapping[code] = {
                    'short_code': '',
                    'league_name': value,
                    'country': '',
                    'source': 'upload'
                }
            unknown[code] = value
            count += 1
        elif isinstance(value, dict):
            league_name = value.get('league_name') or value.get('suggested_name') or value.get('name')
            short_code = value.get('short_code', '')
            country = value.get('country', '')
            
            if league_name and code not in mapping:
                mapping[code] = {
                    'short_code': short_code,
                    'league_name': league_name,
                    'country': country,
                    'source': 'upload'
                }
            if league_name:
                unknown[code] = league_name
                count += 1
    
    with open('data/league_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)
    with open('data/unknown_leagues.json', 'w') as f:
        json.dump(unknown, f, indent=2)
    
    return count

def display_leagues_tab():
    """Display the leagues management tab."""
    st.markdown("### 🏆 League Management")
    st.markdown("Manage league names for codes that aren't recognized automatically.")
    st.markdown("League info is automatically extracted from match pages and saved to the database.")
    
    # Load current leagues from both sources
    mapping = load_league_mapping()
    unknown = load_unknown_leagues()
    
    # Combine all leagues
    all_leagues = {}
    
    # Add from mapping
    for code, info in mapping.items():
        if isinstance(info, dict):
            name = info.get('league_name') or info.get('short_code') or f"League {code}"
            country = info.get('country', '')
            source = info.get('source', 'unknown')
        else:
            name = info
            country = ''
            source = 'unknown'
        all_leagues[code] = {'name': name, 'country': country, 'source': source}
    
    # Add from unknown (not already in mapping)
    for code, value in unknown.items():
        if code not in all_leagues:
            if isinstance(value, str):
                all_leagues[code] = {'name': value, 'country': '', 'source': 'manual'}
            elif isinstance(value, dict):
                name = value.get('suggested_name') or value.get('name', f"League {code}")
                all_leagues[code] = {'name': name, 'country': '', 'source': 'manual'}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 View All Leagues")
        if all_leagues:
            # Convert to DataFrame for display
            league_list = []
            for code, info in all_leagues.items():
                league_list.append({
                    'Code': code, 
                    'Name': info['name'],
                    'Country': info.get('country', ''),
                    'Source': info.get('source', 'unknown')
                })
            
            if league_list:
                df = pd.DataFrame(league_list)
                st.dataframe(df, use_container_width=True)
                st.caption(f"Total: {len(league_list)} leagues")
        else:
            st.info("No leagues found! 🎉")
    
    with col2:
        st.markdown("#### ➕ Add Single League")
        with st.form("add_league_form"):
            code = st.text_input("League Code (5 digits)", placeholder="24236")
            name = st.text_input("League Name", placeholder="Bahrain Premier League")
            submit = st.form_submit_button("💾 Save League")
            
            if submit and code and name:
                if save_league(code.strip(), name.strip()):
                    st.success(f"✅ Saved: {name}")
                    st.rerun()
    
    st.markdown("---")
    
    st.markdown("#### 📤 Upload Multiple Leagues")
    st.markdown("Upload a JSON file with league code-to-name mappings.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Option 1: Upload JSON file**")
        league_file = st.file_uploader(
            "Choose a JSON file",
            type=['json'],
            key="league_json_file"
        )
        
        if league_file is not None:
            try:
                json_data = json.load(league_file)
                st.write(f"Found {len(json_data)} entries in file")
                
                if st.button("📤 Upload All Leagues"):
                    count = upload_leagues_json(json_data)
                    if count > 0:
                        st.success(f"✅ Uploaded {count} leagues!")
                        st.rerun()
                    else:
                        st.warning("No valid leagues found in file")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col2:
        st.markdown("**Option 2: Paste JSON directly**")
        json_text = st.text_area(
            "Paste JSON here",
            placeholder='{\n  "24236": "Bahrain Premier League",\n  "24206": "Kenya Premier League"\n}',
            height=150
        )
        
        if json_text:
            try:
                json_data = json.loads(json_text)
                st.write(f"Found {len(json_data)} entries")
                
                if st.button("📤 Upload Pasted JSON"):
                    count = upload_leagues_json(json_data)
                    if count > 0:
                        st.success(f"✅ Uploaded {count} leagues!")
                        st.rerun()
                    else:
                        st.warning("No valid leagues found")
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
    
    st.markdown("---")
    
    # Download current leagues
    st.markdown("#### 📥 Download League List")
    st.markdown("Download the complete league database (league_mapping.json)")
    
    if st.button("📥 Download League Database"):
        mapping = load_league_mapping()
        st.download_button(
            label="💾 Download league_mapping.json",
            data=json.dumps(mapping, indent=2, ensure_ascii=False),
            file_name="league_mapping.json",
            mime="application/json"
        )

# ════════════════════════════════════════════════════════════════════════
# SETTINGS TAB
# ════════════════════════════════════════════════════════════════════════

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
