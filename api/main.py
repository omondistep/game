#!/usr/bin/env python3
"""
Football Prediction System - Web API
FastAPI-based web application with web UI
Deployable to Vercel Serverless Functions
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Import prediction system
import sys; sys.path.insert(0, "../"); from football_prediction_system import FootballPredictionSystem

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
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
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
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        h2, h3 { margin-bottom: 20px; }
        h3 { margin-top: 20px; color: #00d4ff; }
        
        .api-section { margin-bottom: 30px; }
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
            <div class="tab" onclick="showTab('train')">🧠 Train</div>
            <div class="tab" onclick="showTab('stats')">📊 Stats</div>
            <div class="tab" onclick="showTab('batch')">📚 Batch</div>
            <div class="tab" onclick="showTab('api')">🔗 API</div>
        </div>
        
        <!-- Predict Panel -->
        <div id="predict" class="panel active">
            <h2>🔮 Match Prediction</h2>
            <p style="color: #888; margin-bottom: 20px;">Enter a Forebet match URL to get a prediction</p>
            
            <form id="predictForm" onsubmit="submitPredict(event)">
                <div class="form-group">
                    <label>Forebet Match URL</label>
                    <input type="url" id="predict_url" placeholder="https://www.forebet.com/en/football/matches/..." required>
                </div>
                <button type="submit" class="btn" id="predictBtn">🔮 Get Prediction</button>
            </form>
            <div id="predictResult"></div>
        </div>
        
        <!-- Result Panel -->
        <div id="result" class="panel">
            <h2>📝 Add Match Result</h2>
            <p style="color: #888; margin-bottom: 20px;">Record an actual match result for training</p>
            
            <form id="resultForm" onsubmit="submitResult(event)">
                <div class="form-group">
                    <label>Forebet Match URL</label>
                    <input type="url" id="result_url" placeholder="https://www.forebet.com/..." required>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div class="form-group">
                        <label>Home Score</label>
                        <input type="number" id="home_score" min="0" required>
                    </div>
                    <div class="form-group">
                        <label>Away Score</label>
                        <input type="number" id="away_score" min="0" required>
                    </div>
                </div>
                <button type="submit" class="btn" id="resultBtn">📝 Save Result</button>
            </form>
            <div id="resultOutput"></div>
        </div>
        
        <!-- Train Panel -->
        <div id="train" class="panel">
            <h2>🧠 Train Prediction Model</h2>
            <p style="color: #888; margin-bottom: 20px;">Train the ML model with accumulated match data</p>
            
            <form id="trainForm" onsubmit="submitTrain(event)">
                <div class="form-group">
                    <label>
                        <input type="checkbox" id="force_train" style="width: auto;"> Force training
                    </label>
                </div>
                <button type="submit" class="btn" id="trainBtn">🧠 Start Training</button>
            </form>
            <div id="trainOutput"></div>
        </div>
        
        <!-- Stats Panel -->
        <div id="stats" class="panel">
            <h2>📊 Model Statistics</h2>
            <p style="color: #888; margin-bottom: 20px;">View model performance metrics</p>
            <button class="btn" onclick="loadStats()" style="margin-bottom: 20px;">📊 Load Statistics</button>
            <div id="statsOutput"></div>
        </div>
        
        <!-- Batch Panel -->
        <div id="batch" class="panel">
            <h2>📚 Batch Predictions</h2>
            <p style="color: #888; margin-bottom: 20px;">Process multiple match URLs at once</p>
            
            <form id="batchForm" onsubmit="submitBatch(event)">
                <div class="form-group">
                    <label>Match URLs (one per line)</label>
                    <textarea id="batch_urls" rows="8" placeholder="https://...&#10;https://..."></textarea>
                </div>
                <button type="submit" class="btn" id="batchBtn">📚 Process Batch</button>
            </form>
            <div id="batchOutput"></div>
        </div>
        
        <!-- API Panel -->
        <div id="api" class="panel">
            <h2>🔗 API Documentation</h2>
            
            <div class="api-section">
                <h3>POST /api/predict</h3>
                <pre>{"url": "https://...", "save_data": true}</pre>
            </div>
            
            <div class="api-section">
                <h3>POST /api/result</h3>
                <pre>{"url": "https://...", "home_score": 2, "away_score": 1}</pre>
            </div>
            
            <div class="api-section">
                <h3>POST /api/train</h3>
                <pre>{"force": false}</pre>
            </div>
            
            <div class="api-section">
                <h3>GET /api/stats</h3>
                <p style="color: #888;">Returns model statistics</p>
            </div>
            
            <div class="api-section">
                <h3>GET /health</h3>
                <p style="color: #888;">Health check endpoint</p>
            </div>
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
            const btn = document.getElementById('predictBtn');
            const url = document.getElementById('predict_url').value;
            
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span>';
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url, save_data: true})
                });
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('predictResult').innerHTML = 
                        `<div class="result-box error"><h3>❌ Error</h3><p>${data.error}</p></div>`;
                } else {
                    document.getElementById('predictResult').innerHTML = 
                        `<div class="result-box success"><pre>${JSON.stringify(data, null, 2)}</pre></div>`;
                }
            } catch (err) {
                document.getElementById('predictResult').innerHTML = 
                    `<div class="result-box error"><h3>❌ Error</h3><p>${</p></div>`;
            }
            
           err.message} btn.disabled = false;
            btn.inner🔮 Get Prediction';
HTML = '        }
        
        async function submitResult(e) {
            e.preventDefault();
            const btn = document.getElementById('resultBtn = document.getElement');
            const urlById('result_url').value;
            const home_score = parseInt(document.getElementById('home_score').value);
            const away_score = parseInt(document.getElementById('away_score').value);
            
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span>';
            
            try {
                const response = await fetch('/api/result', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url, home_score, away_score})
                });
                const data = await response.json();
                
                document.getElementById('resultOutput').innerHTML = 
                    `<div class="result-box success"><h3>✅ ${data.message || 'Done'}</h3></div>`;
            } catch (err) {
                document.getElementById('resultOutput').innerHTML = 
                    `<div class="result-box error"><h3>❌ Error</h3><p>${err.message}</p></div>`;
            }
            
            btn.disabled = false;
            btn.innerHTML = '📝 Save Result';
        }
        
        async function submitTrain(e) {
            e.preventDefault();
            const btn = document.getElementById('trainBtn');
            const force = document.getElementById('force_train').checked;
            
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span>';
            
            try {
                const response = await fetch('/api/train', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({force})
                });
                const data = await response.json();
                
                document.getElementById('trainOutput').innerHTML = 
                    `<div class="result-box success"><pre>${JSON.stringify(data, null, 2)}</pre></div>`;
            } catch (err) {
                document.getElementById('trainOutput').innerHTML = 
                    `<div class="result-box error"><h3>❌ Error</h3><p>${err.message}</p></div>`;
            }
            
            btn.disabled = false;
            btn.innerHTML = '🧠 Start Training';
        }
        
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                document.getElementById('statsOutput').innerHTML = 
                    `<div class="result-box"><pre>${JSON.stringify(data, null, 2)}</pre></div>`;
            } catch (err) {
                document.getElementById('statsOutput').innerHTML = 
                    `<div class="result-box error"><h3>❌ Error</h3><p>${err.message}</p></div>`;
            }
        }
        
        async function submitBatch(e) {
            e.preventDefault();
            const btn = document.getElementById('batchBtn');
            const urlsText = document.getElementById('batch_urls').value;
            const urls = urlsText.split('\\n').map(u => u.trim()).filter(u => u);
            
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span>';
            
            try {
                const response = await fetch('/api/batch', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({urls})
                });
                const data = await response.json();
                
                document.getElementById('batchOutput').innerHTML = 
                    `<div class="result-box"><pre>${JSON.stringify(data, null, 2)}</pre></div>`;
            } catch (err) {
                document.getElementById('batchOutput').innerHTML = 
                    `<div class="result-box error"><h3>❌ Error</h3><p>${err.message}</p></div>`;
            }
            
            btn.disabled = false;
            btn.innerHTML = '📚 Process Batch';
        }
    </script>
</body>
</html>
"""


# ======================================================================
# API Routes
# ======================================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web UI."""
    return HTML_TEMPLATE


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/predict")
async def predict(request: dict):
    """Get a prediction for a match URL."""
    try:
        url = request.get("url")
        save_data = request.get("save_data", True)
        
        if not url:
            return JSONResponse(status_code=400, content={"error": "URL is required"})
        
        result = system.predict_match(url, save_data=save_data)
        
        if 'error' in result:
            return JSONResponse(status_code=400, content=result)
        
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/result")
async def add_result(request: dict):
    """Record an actual match result."""
    try:
        url = request.get("url")
        home_score = request.get("home_score")
        away_score = request.get("away_score")
        
        if not url or home_score is None or away_score is None:
            return JSONResponse(status_code=400, content={"error": "Missing required fields"})
        
        success = system.record_result(url, home_score, away_score)
        
        if success:
            return {"success": True, "message": "Result recorded successfully"}
        else:
            return JSONResponse(status_code=400, content={"error": "Failed to record result"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/train")
async def train_model(request: dict):
    """Trigger model training."""
    try:
        force = request.get("force", False)
        success = system.train_models(force=force)
        
        if success:
            return {
                "success": True, 
                "message": "Model trained successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {"success": False, "message": "Training skipped (not enough time elapsed)"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/stats")
async def get_stats():
    """Get model statistics."""
    try:
        stats = system.get_model_stats()
        return stats
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/batch")
async def batch_predict(request: dict):
    """Process multiple URLs for batch predictions."""
    urls = request.get("urls", [])
    results = []
    
    for url in urls:
        try:
            result = system.predict_match(url, save_data=True)
            results.append({
                "url": url,
                "result": result,
                "success": "error" not in result
            })
        except Exception as e:
            results.append({
                "url": url,
                "result": {"error": str(e)},
                "success": False
            })
    
    return {
        "total": len(urls),
        "success": sum(1 for r in results if r["success"]),
        "results": results
    }


@app.get("/api/leagues")
async def get_leagues():
    """Get list of tracked leagues."""
    try:
        leagues = system.storage.get_leagues()
        return {"leagues": leagues}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ======================================================================
# Run locally
# ======================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
