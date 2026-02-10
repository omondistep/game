#!/usr/bin/env python3
"""
Football Prediction System - Web API for Vercel Serverless
Uses Vercel's Python Runtime with ASGI adapter
"""

import sys
import os
from datetime import datetime

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import prediction system
from football_prediction_system import FootballPredictionSystem
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Initialize system
system = FootballPredictionSystem()

# Create FastAPI app
app = FastAPI(
    title="Football Prediction System",
    description="ML-based football match predictions with web interface",
    version="1.0.0"
)

# Embedded HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Football Prediction System</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 40px;
        }
        h1 {
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { color: #888; margin-top: 10px; }
        
        .tabs { display: flex; gap: 10px; margin-bottom: 30px; flex-wrap: wrap; justify-content: center; }
        .tab {
            padding: 12px 24px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .tab:hover, .tab.active {
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            border-color: transparent;
        }
        
        .panel {
            display: none;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 30px;
            max-width: 800px;
            margin: 0 auto;
        }
        .panel.active { display: block; }
        
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; color: #aaa; }
        input, textarea {
            width: 100%;
            padding: 12px 16px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            color: #fff;
            font-size: 16px;
        }
        input:focus, textarea:focus { outline: none; border-color: #00d4ff; }
        
        .btn {
            padding: 14px 28px;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            border: none;
            border-radius: 8px;
            color: #fff;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
        }
        
        .result-box {
            margin-top: 30px;
            padding: 20px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            border-left: 4px solid #00d4ff;
        }
        .result-box.error { border-left-color: #ef4444; }
        .result-box.success { border-left-color: #22c55e; }
        
        pre {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 14px;
        }
        
        h3 { margin-top: 20px; color: #00d4ff; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>⚽ Football Prediction System</h1>
            <p class="subtitle">ML-based match predictions with weighted factor analysis</p>
        </header>
        
        <div class="tabs">
            <div class="tab active" onclick="showTab('predict')">🔮 Predict</div>
            <div class="tab" onclick="showTab('result')">📝 Result</div>
            <div class="tab" onclick="showTab('stats')">📊 Stats</div>
        </div>
        
        <div id="predict" class="panel active">
            <h2>🔮 Match Prediction</h2>
            <p style="color: #888; margin-bottom: 20px;">Enter a Forebet match URL</p>
            <form id="predictForm" onsubmit="submitPredict(event)">
                <div class="form-group">
                    <label>Forebet URL</label>
                    <input type="url" id="predict_url" placeholder="https://..." required>
                </div>
                <button type="submit" class="btn">🔮 Predict</button>
            </form>
            <div id="predictResult"></div>
        </div>
        
        <div id="result" class="panel">
            <h2>📝 Add Result</h2>
            <form id="resultForm" onsubmit="submitResult(event)">
                <div class="form-group">
                    <label>Forebet URL</label>
                    <input type="url" id="result_url" placeholder="https://..." required>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div class="form-group">
                        <label>Home</label>
                        <input type="number" id="home_score" min="0" required>
                    </div>
                    <div class="form-group">
                        <label>Away</label>
                        <input type="number" id="away_score" min="0" required>
                    </div>
                </div>
                <button type="submit" class="btn">📝 Save</button>
            </form>
            <div id="resultOutput"></div>
        </div>
        
        <div id="stats" class="panel">
            <h2>📊 Statistics</h2>
            <button class="btn" onclick="loadStats()" style="margin-bottom: 20px;">📊 Load</button>
            <div id="statsOutput"></div>
        </div>
    </div>
    
    <script>
        function showTab(tabId) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelector(`.tab[onclick="showTab('${tabId}')"]`).classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }
        
        async function submitPredict(e) {
            e.preventDefault();
            const url = document.getElementById('predict_url').value;
            try {
                const r = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url})
                });
                const data = await r.json();
                document.getElementById('predictResult').innerHTML = 
                    data.error 
                        ? `<div class="result-box error">❌ ${data.error}</div>`
                        : `<div class="result-box success"><pre>${JSON.stringify(data, null, 2)}</pre></div>`;
            } catch (err) {
                document.getElementById('predictResult').innerHTML = 
                    `<div class="result-box error">❌ ${err.message}</div>`;
            }
        }
        
        async function submitResult(e) {
            e.preventDefault();
            const url = document.getElementById('result_url').value;
            const home = parseInt(document.getElementById('home_score').value);
            const away = parseInt(document.getElementById('away_score').value);
            try {
                const r = await fetch('/api/result', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url, home_score: home, away_score: away})
                });
                const data = await r.json();
                document.getElementById('resultOutput').innerHTML = 
                    `<div class="result-box success">✅ ${data.message || 'Done'}</div>`;
            } catch (err) {
                document.getElementById('resultOutput').innerHTML = 
                    `<div class="result-box error">❌ ${err.message}</div>`;
            }
        }
        
        async function loadStats() {
            try {
                const r = await fetch('/api/stats');
                const data = await r.json();
                document.getElementById('statsOutput').innerHTML = 
                    `<div class="result-box"><pre>${JSON.stringify(data, null, 2)}</pre></div>`;
            } catch (err) {
                document.getElementById('statsOutput').innerHTML = 
                    `<div class="result-box error">❌ ${err.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""


@app.get("/")
async def home():
    return HTML_TEMPLATE


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


class PredictRequest(BaseModel):
    url: str
    save_data: bool = True


@app.post("/api/predict")
async def predict(request: PredictRequest):
    try:
        result = system.predict_match(request.url, save_data=request.save_data)
        if 'error' in result:
            return {"error": result['error']}
        return result
    except Exception as e:
        return {"error": str(e)}


class ResultRequest(BaseModel):
    url: str
    home_score: int
    away_score: int


@app.post("/api/result")
async def add_result(request: ResultRequest):
    try:
        success = system.record_result(request.url, request.home_score, request.away_score)
        return {"success": success, "message": "Result recorded" if success else "Failed"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/train")
async def train_model():
    try:
        success = system.train_models(force=True)
        return {"success": success, "message": "Training complete" if success else "Skipped"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/stats")
async def get_stats():
    try:
        return system.get_model_stats()
    except Exception as e:
        return {"error": str(e)}
