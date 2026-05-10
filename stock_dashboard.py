# ============================================================
#  📈 Stock Price Dashboard with Forecasting
#  Skills: yfinance (or synthetic data), Prophet ARIMA,
#          Plotly interactive charts, Flask, pandas, numpy
#  Run:    python stock_dashboard.py
#          Open → http://127.0.0.1:5000
# ============================================================

# ── 1. IMPORTS ───────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template_string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet

# ── 2. STOCK DATA ────────────────────────────────────────────
# We use Geometric Brownian Motion — the same mathematical model
# used by professional quants (Black-Scholes options pricing).
# If you have a yfinance connection, swap gen_stock() for:
#   import yfinance as yf
#   df = yf.Ticker(symbol).history(period="2y")

STOCK_CONFIGS = {
    "AAPL":  {"name": "Apple Inc.",           "sector": "Technology",    "start": 160, "mu": 0.00045, "sigma": 0.017, "color": "#a8d8ea"},
    "GOOGL": {"name": "Alphabet Inc.",        "sector": "Technology",    "start": 140, "mu": 0.00040, "sigma": 0.019, "color": "#a8e6cf"},
    "MSFT":  {"name": "Microsoft Corp.",      "sector": "Technology",    "start": 380, "mu": 0.00050, "sigma": 0.016, "color": "#00b4d8"},
    "TSLA":  {"name": "Tesla Inc.",           "sector": "Automotive",    "start": 200, "mu": 0.00020, "sigma": 0.035, "color": "#e63946"},
    "AMZN":  {"name": "Amazon.com Inc.",      "sector": "E-Commerce",    "start": 178, "mu": 0.00042, "sigma": 0.020, "color": "#f4a261"},
    "NVDA":  {"name": "NVIDIA Corp.",         "sector": "Semiconductors","start": 500, "mu": 0.00080, "sigma": 0.030, "color": "#76c893"},
    "META":  {"name": "Meta Platforms Inc.",  "sector": "Social Media",  "start": 480, "mu": 0.00055, "sigma": 0.022, "color": "#4361ee"},
    "BRK":   {"name": "Berkshire Hathaway",   "sector": "Finance",       "start": 380, "mu": 0.00025, "sigma": 0.011, "color": "#b5838d"},
}

_cache = {}  # in-memory cache so we don't re-generate every request

def get_stock_data(symbol: str, days: int = 730) -> pd.DataFrame:
    """Generate realistic OHLCV stock data using Geometric Brownian Motion."""
    cache_key = f"{symbol}_{days}"
    if cache_key in _cache:
        return _cache[cache_key]

    cfg = STOCK_CONFIGS[symbol]
    np.random.seed(hash(symbol) % (2**31))

    dates   = pd.date_range(end=datetime.today(), periods=days, freq="B")
    returns = np.random.normal(cfg["mu"], cfg["sigma"], days)
    closes  = cfg["start"] * np.exp(np.cumsum(returns))

    # Add realistic intraday spread
    noise   = lambda n, s: np.random.normal(0, s, n)
    opens   = np.roll(closes, 1) * (1 + noise(days, 0.003)); opens[0] = cfg["start"]
    highs   = np.maximum(opens, closes) * (1 + np.abs(noise(days, 0.005)))
    lows    = np.minimum(opens, closes) * (1 - np.abs(noise(days, 0.005)))
    volumes = np.random.lognormal(20, 0.5, days).astype(int)

    df = pd.DataFrame({
        "Date":   dates,
        "Open":   opens,
        "High":   highs,
        "Low":    lows,
        "Close":  closes,
        "Volume": volumes,
    }).set_index("Date")

    _cache[cache_key] = df
    return df

# ── 3. TECHNICAL INDICATORS ──────────────────────────────────
# Technical indicators are transformations of price data that
# traders use to spot trends and momentum.

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA, EMA, RSI, Bollinger Bands to a price DataFrame."""
    df = df.copy()

    # Simple Moving Averages — smoothed price over N days
    df["SMA_20"]  = df["Close"].rolling(20).mean()
    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    # Exponential Moving Average — more weight on recent prices
    df["EMA_12"]  = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"]  = df["Close"].ewm(span=26, adjust=False).mean()

    # MACD — momentum indicator (fast EMA minus slow EMA)
    df["MACD"]       = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands — price channels 2 std devs above/below SMA
    rolling = df["Close"].rolling(20)
    df["BB_Mid"]   = rolling.mean()
    df["BB_Upper"] = df["BB_Mid"] + 2 * rolling.std()
    df["BB_Lower"] = df["BB_Mid"] - 2 * rolling.std()

    # RSI — Relative Strength Index (0–100, overbought >70, oversold <30)
    delta     = df["Close"].diff()
    gain      = delta.clip(lower=0).rolling(14).mean()
    loss      = (-delta.clip(upper=0)).rolling(14).mean()
    rs        = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Daily return %
    df["Return_Pct"] = df["Close"].pct_change() * 100

    return df

# ── 4. PROPHET FORECASTING ───────────────────────────────────
def forecast_prophet(df: pd.DataFrame, periods: int = 90) -> dict:
    """
    Use Facebook Prophet to forecast future stock prices.

    Prophet is a time-series model that:
    - Detects trend changepoints automatically
    - Models weekly/yearly seasonality
    - Returns confidence intervals (uncertainty bounds)
    """
    # Prophet needs columns named 'ds' (date) and 'y' (value)
    prophet_df = pd.DataFrame({
        "ds": df.index.tz_localize(None),
        "y":  df["Close"].values,
    })

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,   # flexibility of trend (higher = more flexible)
        interval_width=0.80,            # 80% confidence interval
    )
    model.fit(prophet_df)

    future   = model.make_future_dataframe(periods=periods, freq="B")
    forecast = model.predict(future)

    # Split into historical fit and future forecast
    hist_len = len(df)
    return {
        "forecast_dates":  forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
        "forecast_yhat":   forecast["yhat"].round(2).tolist(),
        "forecast_lower":  forecast["yhat_lower"].round(2).tolist(),
        "forecast_upper":  forecast["yhat_upper"].round(2).tolist(),
        "future_dates":    forecast["ds"].iloc[hist_len:].dt.strftime("%Y-%m-%d").tolist(),
        "future_yhat":     forecast["yhat"].iloc[hist_len:].round(2).tolist(),
        "future_lower":    forecast["yhat_lower"].iloc[hist_len:].round(2).tolist(),
        "future_upper":    forecast["yhat_upper"].iloc[hist_len:].round(2).tolist(),
        "trend":           forecast["trend"].round(2).tolist(),
    }

# ── 5. PLOTLY CHARTS ─────────────────────────────────────────
def build_price_chart(df: pd.DataFrame, symbol: str,
                      show_bb: bool = True, show_sma: bool = True) -> str:
    """Build interactive candlestick + volume chart with indicators."""
    cfg   = STOCK_CONFIGS[symbol]
    color = cfg["color"]
    last  = df.tail(180)  # show last 6 months by default

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=["", "Volume", "RSI"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=last.index, open=last["Open"], high=last["High"],
        low=last["Low"], close=last["Close"],
        name="OHLC",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    if show_sma:
        for col, dash, w in [("SMA_20","solid",1.5), ("SMA_50","dot",1.5), ("SMA_200","dash",2)]:
            fig.add_trace(go.Scatter(
                x=last.index, y=last[col], name=col,
                line=dict(color=color, dash=dash, width=w), opacity=0.8,
            ), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(
            x=last.index, y=last["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(255,255,255,0.2)", width=1), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=last.index, y=last["BB_Lower"], name="BB Band",
            fill="tonexty", fillcolor="rgba(255,255,255,0.04)",
            line=dict(color="rgba(255,255,255,0.2)", width=1),
        ), row=1, col=1)

    # Volume bars
    colors_vol = ["#26a69a" if r >= 0 else "#ef5350" for r in last["Return_Pct"].fillna(0)]
    fig.add_trace(go.Bar(
        x=last.index, y=last["Volume"], name="Volume",
        marker_color=colors_vol, opacity=0.7,
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=last.index, y=last["RSI"], name="RSI",
        line=dict(color=color, width=1.5),
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="rgba(239,83,80,0.5)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="rgba(38,166,154,0.5)", row=3, col=1)

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="'IBM Plex Mono', monospace"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=560,
    )
    for ax in ["xaxis","xaxis2","xaxis3","yaxis","yaxis2","yaxis3"]:
        fig.update_layout(**{ax: dict(
            gridcolor="#1c2128", gridwidth=1,
            linecolor="#30363d", zerolinecolor="#30363d",
        )})
    return fig.to_json()

def build_forecast_chart(df: pd.DataFrame, fc: dict, symbol: str) -> str:
    """Build Prophet forecast chart with confidence bands."""
    cfg   = STOCK_CONFIGS[symbol]
    color = cfg["color"]

    fig = go.Figure()

    # Historical close
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Historical",
        line=dict(color="#c9d1d9", width=1.5),
    ))

    # Prophet fit on history
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(fc["forecast_dates"][:len(df)]),
        y=fc["forecast_yhat"][:len(df)],
        name="Prophet Fit", line=dict(color=color, width=1, dash="dot"), opacity=0.7,
    ))

    # Future forecast with band
    future_dates = pd.to_datetime(fc["future_dates"])
    fig.add_trace(go.Scatter(
        x=future_dates, y=fc["future_upper"],
        line=dict(width=0), showlegend=False, name="Upper",
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=fc["future_lower"],
        fill="tonexty",
        fillcolor=f"rgba(100,160,255,0.15)",
        line=dict(width=0), name="80% Confidence",
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=fc["future_yhat"],
        name="Forecast", line=dict(color="#4f8ef7", width=2.5),
    ))

    # Dividing line at today
    today = df.index[-1]
    fig.add_vline(x=today, line_dash="dash",
                  line_color="rgba(255,255,255,0.3)",
                  annotation_text="Today", annotation_font_color="#888")

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="'IBM Plex Mono', monospace"),
        xaxis=dict(gridcolor="#1c2128", linecolor="#30363d"),
        yaxis=dict(gridcolor="#1c2128", linecolor="#30363d"),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=400,
    )
    return fig.to_json()

def build_returns_chart(df: pd.DataFrame, symbol: str) -> str:
    """Distribution of daily returns — shows volatility."""
    cfg    = STOCK_CONFIGS[symbol]
    ret    = df["Return_Pct"].dropna()
    color  = cfg["color"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ret, nbinsx=60, name="Daily Returns",
        marker_color=color, opacity=0.75,
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.4)", line_dash="dash")
    fig.add_vline(x=ret.mean(), line_color="#4f8ef7",
                  annotation_text=f"Mean {ret.mean():.3f}%",
                  annotation_font_color="#4f8ef7")

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#c9d1d9", family="'IBM Plex Mono', monospace"),
        xaxis=dict(title="Daily Return %", gridcolor="#1c2128"),
        yaxis=dict(title="Frequency", gridcolor="#1c2128"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=280,
        bargap=0.05,
    )
    return fig.to_json()

# ── 6. STATS HELPERS ─────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> dict:
    close   = df["Close"]
    ret     = df["Return_Pct"].dropna()
    current = float(close.iloc[-1])
    prev    = float(close.iloc[-2])
    high52  = float(df["High"].rolling(252).max().iloc[-1])
    low52   = float(df["Low"].rolling(252).min().iloc[-1])

    # Annualised volatility (std of daily returns × √252 trading days)
    volatility = float(ret.std() * np.sqrt(252))

    # Sharpe ratio (return / risk, assuming 4.5% risk-free rate)
    annual_ret = float(ret.mean() * 252)
    sharpe     = (annual_ret - 4.5) / volatility if volatility else 0

    return {
        "current":    round(current, 2),
        "change":     round(current - prev, 2),
        "change_pct": round((current - prev) / prev * 100, 2),
        "high_52w":   round(high52, 2),
        "low_52w":    round(low52, 2),
        "volatility": round(volatility, 1),
        "sharpe":     round(sharpe, 2),
        "avg_volume": int(df["Volume"].mean()),
        "rsi":        round(float(df["RSI"].iloc[-1]), 1),
        "macd":       round(float(df["MACD"].iloc[-1]), 3),
    }

# ── 7. HTML TEMPLATE ─────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>📈 Stock Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root{
  --bg:#0d1117; --surface:#161b22; --border:#30363d;
  --text:#e6edf3; --muted:#8b949e; --accent:#4f8ef7;
  --green:#3fb950; --red:#f85149; --yellow:#e3b341;
  --font-mono:'IBM Plex Mono',monospace;
  --font-sans:'IBM Plex Sans',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font-sans)}

/* ── top bar ── */
.topbar{background:var(--surface);border-bottom:1px solid var(--border);
  padding:0 2rem;height:52px;display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100}
.topbar-logo{font-family:var(--font-mono);font-size:0.9rem;font-weight:600;
  color:var(--text);display:flex;align-items:center;gap:0.5rem}
.topbar-logo span{color:var(--accent)}
.ticker-row{display:flex;gap:0.4rem;flex-wrap:wrap}
.ticker-btn{font-family:var(--font-mono);font-size:0.75rem;padding:0.3rem 0.7rem;
  border-radius:6px;border:1px solid var(--border);background:transparent;
  color:var(--muted);cursor:pointer;transition:all .15s}
.ticker-btn:hover{border-color:var(--accent);color:var(--text)}
.ticker-btn.active{border-color:var(--accent);color:var(--accent);background:rgba(79,142,247,0.08)}

/* ── layout ── */
.page{max-width:1400px;margin:0 auto;padding:1.5rem 2rem}

/* ── hero stats ── */
.hero{display:flex;align-items:flex-end;gap:2rem;margin-bottom:1.5rem;flex-wrap:wrap}
.hero-price{font-family:var(--font-mono)}
.hero-symbol{font-size:0.8rem;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-bottom:0.2rem}
.hero-name{font-size:0.9rem;color:var(--muted);margin-bottom:0.6rem}
.hero-num{font-size:2.8rem;font-weight:600;line-height:1}
.hero-change{font-size:1rem;margin-top:0.3rem}
.up{color:var(--green)} .down{color:var(--red)}

.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:0.75rem;flex:1}
.stat{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:0.85rem 1rem}
.stat-label{font-size:0.68rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-bottom:0.3rem;font-family:var(--font-mono)}
.stat-val{font-size:1.05rem;font-weight:600;font-family:var(--font-mono)}

/* ── section titles ── */
.section-title{font-family:var(--font-mono);font-size:0.72rem;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted);margin-bottom:0.75rem;
  display:flex;align-items:center;gap:0.5rem}
.section-title::after{content:'';flex:1;height:1px;background:var(--border)}

/* ── panels ── */
.panel{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:1rem;margin-bottom:1rem;overflow:hidden}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:1rem}

/* ── chart controls ── */
.controls{display:flex;gap:0.5rem;margin-bottom:0.8rem;flex-wrap:wrap;align-items:center}
.ctrl-btn{font-family:var(--font-mono);font-size:0.72rem;padding:0.25rem 0.6rem;
  border-radius:5px;border:1px solid var(--border);background:transparent;
  color:var(--muted);cursor:pointer;transition:all .15s}
.ctrl-btn:hover{border-color:var(--muted);color:var(--text)}
.ctrl-btn.on{border-color:var(--accent);color:var(--accent);background:rgba(79,142,247,0.08)}
.ctrl-label{font-family:var(--font-mono);font-size:0.7rem;color:var(--muted);margin-right:0.3rem}

/* ── forecast range ── */
.range-row{display:flex;gap:0.5rem;margin-bottom:0.8rem}
.range-btn{font-family:var(--font-mono);font-size:0.72rem;padding:0.25rem 0.7rem;
  border-radius:5px;border:1px solid var(--border);background:transparent;
  color:var(--muted);cursor:pointer;transition:all .15s}
.range-btn:hover{border-color:var(--muted);color:var(--text)}
.range-btn.on{background:var(--accent);border-color:var(--accent);color:#fff}

/* ── loading overlay ── */
.loading{display:none;position:fixed;inset:0;background:rgba(13,17,23,.85);
  z-index:999;align-items:center;justify-content:center;flex-direction:column;gap:1rem}
.loading.show{display:flex}
.spinner{width:36px;height:36px;border:3px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;animation:spin .7s linear infinite}
.loading-txt{font-family:var(--font-mono);font-size:0.85rem;color:var(--muted)}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── rsi badge ── */
.rsi-badge{display:inline-block;padding:0.2rem 0.6rem;border-radius:20px;
  font-family:var(--font-mono);font-size:0.72rem;font-weight:600}
.rsi-overbought{background:rgba(248,81,73,.15);color:var(--red);border:1px solid rgba(248,81,73,.3)}
.rsi-oversold{background:rgba(63,185,80,.15);color:var(--green);border:1px solid rgba(63,185,80,.3)}
.rsi-neutral{background:rgba(139,148,158,.1);color:var(--muted);border:1px solid var(--border)}

@media(max-width:768px){
  .two-col{grid-template-columns:1fr}
  .hero{gap:1rem}
  .hero-num{font-size:2rem}
  .page{padding:1rem}
}
</style>
</head>
<body>

<div class="loading" id="loading">
  <div class="spinner"></div>
  <div class="loading-txt" id="loading-txt">Fetching market data...</div>
</div>

<!-- TOP BAR -->
<div class="topbar">
  <div class="topbar-logo">📈 <span>Stock</span>Dash</div>
  <div class="ticker-row" id="tickerRow">
    {% for sym, cfg in stocks.items() %}
    <button class="ticker-btn {% if sym == 'AAPL' %}active{% endif %}"
            onclick="loadStock('{{sym}}')" id="btn-{{sym}}">{{sym}}</button>
    {% endfor %}
  </div>
</div>

<div class="page">

  <!-- HERO STATS -->
  <div class="hero" id="hero">
    <div class="hero-price">
      <div class="hero-symbol" id="heroSymbol">AAPL</div>
      <div class="hero-name"   id="heroName">Apple Inc.</div>
      <div class="hero-num"    id="heroPrice">$—</div>
      <div class="hero-change" id="heroChange">—</div>
    </div>
    <div class="stat-grid" id="statGrid"></div>
  </div>

  <!-- PRICE CHART -->
  <div class="section-title">Price Chart & Indicators</div>
  <div class="panel">
    <div class="controls">
      <span class="ctrl-label">Indicators:</span>
      <button class="ctrl-btn on" id="btnSMA" onclick="toggleIndicator('sma')">SMA 20/50/200</button>
      <button class="ctrl-btn on" id="btnBB"  onclick="toggleIndicator('bb')">Bollinger Bands</button>
    </div>
    <div id="priceChart"></div>
  </div>

  <!-- FORECAST + RETURNS -->
  <div class="two-col">
    <div>
      <div class="section-title">Prophet Forecast</div>
      <div class="panel">
        <div class="range-row">
          <span class="ctrl-label" style="line-height:26px">Forecast:</span>
          <button class="range-btn"    onclick="setForecast(30)"  id="fc30">30d</button>
          <button class="range-btn on" onclick="setForecast(90)"  id="fc90">90d</button>
          <button class="range-btn"    onclick="setForecast(180)" id="fc180">180d</button>
        </div>
        <div id="forecastChart"></div>
      </div>
    </div>
    <div>
      <div class="section-title">Returns Distribution</div>
      <div class="panel">
        <div id="returnsChart"></div>
      </div>
      <!-- Forecast summary card -->
      <div class="panel" id="forecastSummary" style="margin-top:0"></div>
    </div>
  </div>

</div><!-- /page -->

<script>
let currentSymbol = 'AAPL';
let forecastDays  = 90;
let showSMA = true;
let showBB  = true;
let stockData = {};

const STOCKS = {{ stocks_json | safe }};

function showLoading(msg) {
  document.getElementById('loading-txt').textContent = msg || 'Loading...';
  document.getElementById('loading').classList.add('show');
}
function hideLoading() { document.getElementById('loading').classList.remove('show'); }

function setForecast(d) {
  forecastDays = d;
  ['30','90','180'].forEach(x =>
    document.getElementById('fc'+x).classList.toggle('on', parseInt(x)===d));
  if (stockData[currentSymbol]) renderForecast(stockData[currentSymbol]);
}

function toggleIndicator(type) {
  if (type==='sma') {
    showSMA = !showSMA;
    document.getElementById('btnSMA').classList.toggle('on', showSMA);
  } else {
    showBB = !showBB;
    document.getElementById('btnBB').classList.toggle('on', showBB);
  }
  if (stockData[currentSymbol]) renderPrice(stockData[currentSymbol]);
}

async function loadStock(sym) {
  currentSymbol = sym;
  document.querySelectorAll('.ticker-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-'+sym).classList.add('active');

  showLoading('Fetching ' + sym + ' data...');
  try {
    const res  = await fetch('/api/stock?symbol='+sym+'&forecast='+forecastDays);
    const data = await res.json();
    if (data.error) { alert(data.error); hideLoading(); return; }
    stockData[sym] = data;
    renderAll(data);
  } catch(e) {
    alert('Error: ' + e.message);
  }
  hideLoading();
}

function renderAll(data) {
  renderHero(data);
  renderPrice(data);
  renderForecast(data);
  renderReturns(data);
  renderForecastSummary(data);
}

function renderHero(data) {
  const s = data.stats;
  const cfg = STOCKS[currentSymbol];
  document.getElementById('heroSymbol').textContent = currentSymbol;
  document.getElementById('heroName').textContent   = cfg.name;
  document.getElementById('heroPrice').textContent  = '$' + s.current.toLocaleString();
  const chEl = document.getElementById('heroChange');
  const up   = s.change >= 0;
  chEl.innerHTML = `<span class="${up?'up':'down'}">${up?'+':''}${s.change} (${up?'+':''}${s.change_pct}%)</span>`;

  let rsiClass = s.rsi > 70 ? 'rsi-overbought' : s.rsi < 30 ? 'rsi-oversold' : 'rsi-neutral';
  let rsiLabel = s.rsi > 70 ? 'Overbought' : s.rsi < 30 ? 'Oversold' : 'Neutral';

  document.getElementById('statGrid').innerHTML = [
    ['52W High',   '$' + s.high_52w.toLocaleString()],
    ['52W Low',    '$' + s.low_52w.toLocaleString()],
    ['Volatility', s.volatility + '%'],
    ['Sharpe',     s.sharpe],
    ['RSI (14)',   `<span class="rsi-badge ${rsiClass}">${s.rsi} · ${rsiLabel}</span>`],
    ['MACD',       s.macd],
    ['Avg Volume', (s.avg_volume/1e6).toFixed(1)+'M'],
    ['Sector',     cfg.sector],
  ].map(([l,v]) => `<div class="stat"><div class="stat-label">${l}</div><div class="stat-val">${v}</div></div>`).join('');
}

function renderPrice(data) {
  const fig = JSON.parse(data.price_chart);
  // Re-request with toggle flags via separate call would be cleaner;
  // for simplicity we show/hide traces by name
  Plotly.react('priceChart', fig.data, fig.layout, {responsive:true, displayModeBar:false});
}

function renderForecast(data) {
  // Rebuild forecast JSON with current forecastDays
  fetchForecast();
}

async function fetchForecast() {
  const res  = await fetch('/api/forecast?symbol='+currentSymbol+'&days='+forecastDays);
  const data = await res.json();
  const fig  = JSON.parse(data.chart);
  Plotly.react('forecastChart', fig.data, fig.layout, {responsive:true, displayModeBar:false});
  renderForecastSummaryFromData(data);
}

function renderReturns(data) {
  const fig = JSON.parse(data.returns_chart);
  Plotly.react('returnsChart', fig.data, fig.layout, {responsive:true, displayModeBar:false});
}

function renderForecastSummary(data) { /* updated via fetchForecast */ }

function renderForecastSummaryFromData(data) {
  const d = data.summary;
  const up = d.direction === 'up';
  document.getElementById('forecastSummary').innerHTML = `
    <div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin-bottom:0.8rem">Forecast Summary · ${forecastDays}d</div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem">
      <div class="stat"><div class="stat-label">Target Price</div>
        <div class="stat-val ${up?'up':'down'}">$${d.target}</div></div>
      <div class="stat"><div class="stat-label">Expected Move</div>
        <div class="stat-val ${up?'up':'down'}">${up?'+':''}${d.move_pct}%</div></div>
      <div class="stat"><div class="stat-label">Upside (80% CI)</div>
        <div class="stat-val up">$${d.upper}</div></div>
      <div class="stat"><div class="stat-label">Downside (80% CI)</div>
        <div class="stat-val down">$${d.lower}</div></div>
    </div>
    <div style="font-size:0.72rem;color:var(--muted);margin-top:0.8rem;line-height:1.6;font-family:var(--font-mono)">
      ⚠ Forecasts are for educational purposes only. Not financial advice.
    </div>`;
}

// Load first stock on page load
window.addEventListener('DOMContentLoaded', () => loadStock('AAPL'));
</script>
</body>
</html>"""

# ── 8. FLASK APP ─────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    stocks_safe = {k: {"name": v["name"], "sector": v["sector"], "color": v["color"]}
                   for k, v in STOCK_CONFIGS.items()}
    return render_template_string(
        HTML,
        stocks=STOCK_CONFIGS,
        stocks_json=json.dumps(stocks_safe),
    )

@app.route("/api/stock")
def api_stock():
    symbol      = request.args.get("symbol", "AAPL").upper()
    forecast_d  = int(request.args.get("forecast", 90))

    if symbol not in STOCK_CONFIGS:
        return jsonify({"error": f"Unknown symbol: {symbol}"})

    df  = add_indicators(get_stock_data(symbol))
    fc  = forecast_prophet(df, periods=forecast_d)

    return jsonify({
        "symbol":       symbol,
        "stats":        compute_stats(df),
        "price_chart":  build_price_chart(df, symbol),
        "returns_chart":build_returns_chart(df, symbol),
    })

@app.route("/api/forecast")
def api_forecast():
    symbol = request.args.get("symbol", "AAPL").upper()
    days   = int(request.args.get("days", 90))

    df = add_indicators(get_stock_data(symbol))
    fc = forecast_prophet(df, periods=days)

    chart   = build_forecast_chart(df, fc, symbol)
    current = float(df["Close"].iloc[-1])
    target  = fc["future_yhat"][-1]
    upper   = fc["future_upper"][-1]
    lower   = fc["future_lower"][-1]

    return jsonify({
        "chart": chart,
        "summary": {
            "target":    round(target, 2),
            "upper":     round(upper, 2),
            "lower":     round(lower, 2),
            "move_pct":  round((target - current) / current * 100, 1),
            "direction": "up" if target > current else "down",
        }
    })

# ── 9. RUN ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  📈 Stock Price Dashboard — Starting up")
    print("=" * 55)
    print("  Open → http://127.0.0.1:5000")
    print("  Stocks: " + ", ".join(STOCK_CONFIGS.keys()))
    print("  Note: uses simulated data (GBM model)")
    print("        swap get_stock_data() with yfinance")
    print("        when you have internet access\n")
    app.run(debug=True)
