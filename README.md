[# Stock-Price-Dashboard-with-Forecasting
An intermediate Python ML project that visualises stock price history, computes financial technical indicators, and forecasts future prices using Facebook Prophet. Features a full interactive dark-themed dashboard built with Plotly and Flask.

🖥️ Dashboard Preview

Run the app → open http://127.0.0.1:5000


8 stocks — AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, BRK
Live candlestick chart with SMA 20/50/200 and Bollinger Bands
Volume bars and RSI (Relative Strength Index) panel
Prophet forecast — 30 / 90 / 180-day horizon with 80% confidence bands
Returns distribution — histogram of daily % returns showing volatility
Key stats — 52-week high/low, Sharpe ratio, volatility, MACD


🛠️ Tech Stack
ToolPurposeyfinancePull real live stock data from Yahoo Financepandas + numpyTime-series data manipulationprophetFacebook's time-series forecasting modelplotlyInteractive candlestick + forecast chartsflaskWeb server + REST APIGBM (Geometric Brownian Motion)Fallback synthetic data model (same math as Black-Scholes)

🚀 How to Run
1. Clone the repo
bashgit clone https://github.com/YOUR_USERNAME/stock-dashboard.git
cd stock-dashboard
2. Install dependencies
bashpip install yfinance plotly prophet flask pandas numpy
3. Run the app
bashpython stock_dashboard.py
4. Open your browser
http://127.0.0.1:5000

Note: If yfinance is blocked on your network, the app automatically uses synthetic data generated with Geometric Brownian Motion — the same mathematical model used in professional options pricing.


📁 Project Structure
stock-dashboard/
│
├── stock_dashboard.py    # Full app — data, indicators, forecasting, Flask UI
└── README.md

📐 Technical Indicators Explained
IndicatorWhat It MeasuresSMA 20/50/200Simple Moving Average — smoothed price trend over N daysBollinger BandsPrice channels 2 standard deviations above/below SMA — volatilityRSI (14)Relative Strength Index — momentum (>70 = overbought, <30 = oversold)MACDMoving Average Convergence Divergence — trend momentum signalEMA 12/26Exponential Moving Average — more weight on recent prices

🔮 How Prophet Forecasting Works
Facebook Prophet is a time-series model that:

Detects trend changepoints automatically — where the price direction shifted
Models weekly/yearly seasonality — patterns that repeat on a calendar
Returns confidence intervals — a range of plausible future values (80% CI)
Handles missing data and outliers gracefully

pythonmodel = Prophet(
    weekly_seasonality=True,
    changepoint_prior_scale=0.05,   # how flexible the trend can be
    interval_width=0.80,            # 80% confidence band
)
model.fit(df)  # df has columns: ds (date), y (price)
forecast = model.predict(future)
# Returns: yhat (prediction), yhat_lower, yhat_upper

📊 Financial Concepts Implemented
python# Annualised Volatility
volatility = daily_returns.std() * np.sqrt(252)  # 252 trading days/year

# Sharpe Ratio (risk-adjusted return)
sharpe = (annual_return - risk_free_rate) / volatility

# Geometric Brownian Motion (stock price simulation)
returns = np.random.normal(mu, sigma, days)
prices  = start_price * np.exp(np.cumsum(returns))

🔌 Switching to Live Data
The app uses synthetic data by default. To use real live data, replace get_stock_data():
pythonimport yfinance as yf

def get_stock_data(symbol: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2y")
    return df   # already has Open, High, Low, Close, Volume

💡 Key Insights From the Dashboard

NVDA shows the highest annualised volatility (~30%) — high risk, high return
BRK has the tightest Bollinger Bands — most stable price action
RSI crossings of 70/30 are visible signals in the indicator panel
MACD histogram flips from green to red at trend reversals
Prophet's confidence bands widen significantly beyond 90 days — uncertainty compounds


🧠 What I Learned

How Geometric Brownian Motion models stock price randomness
How to compute financial technical indicators from raw OHLCV data
What Prophet is and how to use it for time-series forecasting
How to build interactive Plotly charts (candlestick, subplots, fills)
How the Sharpe ratio and annualised volatility are calculated
How to structure a multi-endpoint Flask API for a data dashboard


This is Intermediate Project 1 of 5 in the Python Data/ML learning roadmap.

🗺️ Full Learning Roadmap
Level#ProjectBeginner1EDA DashboardBeginner2House Price PredictorBeginner3Movie RecommenderBeginner4Sentiment AnalyzerBeginner5Study Buddy AI ChatbotIntermediate1Stock Price Dashboard ← You a](https://github.com/AahanKotian/Stock-Price-Dashboard-with-Forecasting)re hereIntermediate2Image Classifier (Deep Learning)Intermediate3RAG Document ChatbotIntermediate4Fake News DetectorIntermediate5Real-Time Data Pipeline
