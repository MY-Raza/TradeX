# 🚀 TradeX – Quantitative Crypto Trading Engine

TradeX is an end-to-end **quantitative trading platform** built for crypto futures — covering everything from **data ingestion → feature engineering → ML/DL modeling → backtesting → live execution**.

---

## 🌐 Platforms Integrated
Binance • Bybit • Kraken • MetaTrader 5  

✔️ Multi-exchange data pipelines  
✔️ Unified high-frequency OHLCV dataset  
✔️ Cross-platform strategy validation  

---

## ⚙️ Core System Architecture

📥 Data Ingestion  
→ Fetches high-frequency futures data (1m candles) from multiple exchanges  

🧹 Data Processing  
→ Cleaning, normalization, timestamp alignment, missing value handling  

🗄️ Storage  
→ PostgreSQL + TimescaleDB optimized for time-series workloads  

📊 Feature Engineering  
→ Technical indicators (RSI, MACD, ATR, BBANDS, EMA, etc.)  
→ NLP sentiment features from Reddit (crypto discussions)  

🧠 Modeling Layer  
→ ML Models: Random Forest, XGBoost  
→ DL Models: LSTM, GRU, TCN, Transformers, TFT  
→ Forecasting: ARIMA, N-BEATS  

⚡ Optimization  
→ Optuna for hyperparameter tuning  
→ PnL-driven model selection  

📈 Backtesting Engine  
→ Multi-timeframe signals (1h / 15m / 5m)  
→ Risk management (TP/SL)  
→ Strategy evaluation using real trading metrics  

💰 Live Execution  
→ Binance Futures Testnet deployment  
→ Real-time signal generation & order execution  
→ PnL tracking and logging  

---

## 🧠 NLP Sentiment Pipeline

✔️ Extracts data from crypto subreddits  
✔️ Uses FinBERT for sentiment classification  
✔️ Cleans & deduplicates posts (SimHash)  
✔️ Aggregates sentiment into time-based features  
✔️ Aligns sentiment with OHLCV data  

👉 Enhances trading signals by combining **market data + crowd sentiment**

---

## 📊 Strategy Intelligence

✔️ Indicator-based signals  
✔️ ML/DL-driven predictions  
✔️ Hybrid signal system (ensemble voting)  
✔️ PnL-based model ranking  

📌 Custom scoring:
`Score = (PnL × Sharpe Ratio) / |Max Drawdown|`

---

## 🔄 End-to-End Pipeline

```
Raw Market Data + Reddit Data
        ↓
Data Cleaning & Processing
        ↓
Feature Engineering
        ↓
ML/DL Modeling + Optimization
        ↓
Signal Generation
        ↓
Backtesting (PnL Evaluation)
        ↓
Live Execution
```

---

## 🛠️ Tech Stack

**Languages & Tools**  
Python • Pandas • NumPy • SQLAlchemy  

**Databases**  
PostgreSQL • TimescaleDB  

**ML/DL**  
Scikit-learn • XGBoost • PyTorch • Darts  

**Optimization**  
Optuna  

**APIs**  
Binance • Bybit • Kraken • MT5  

**NLP**  
FinBERT • Transformers  

---

## 💡 Key Highlights

✨ Built a **modular, production-ready quant system**  
✨ Shifted from **accuracy → profit-driven modeling**  
✨ Integrated **NLP sentiment with market data**  
✨ Designed **risk-aware strategy evaluation**  
✨ Achieved **real-time trading execution pipeline**  

---

## 📌 Key Insight

> In quantitative trading,  
> the best model isn’t the most accurate —  
> it’s the one that delivers consistent, risk-adjusted returns.

---


## 🤝 Feedback

Open to suggestions, collaborations, and ideas!  
Let’s build smarter trading systems 🚀# TradeX
