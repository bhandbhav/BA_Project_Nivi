"""
Nivi Oracle Engine — Walk-Forward Backtest & SHAP Weight Derivation
====================================================================
Trains an XGBoost classifier on historical Group A technicals.
Uses Walk-Forward validation (TimeSeriesSplit) to prevent look-ahead bias.
Extracts SHAP (SHapley Additive exPlanations) values to dynamically 
assign percentage weights to each technical indicator.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

from utils import fetch_prices, get_logger

log = get_logger("backtest_engine")
WEIGHTS_FILE = Path("data/static/dynamic_weights.json")

def prepare_ml_dataset(tickers: list, period: str = "3y") -> pd.DataFrame:
    """
    Fetches historical data and calculates Group A features.
    Creates a target variable: 1 if the 21-day forward return is positive.
    """
    print(f"📥 Fetching {period} of data for SHAP derivation...")
    prices = fetch_prices(tickers, period=period)
    
    if prices.empty:
        raise ValueError("No price data fetched for ML dataset.")

    dataset = []
    
    print("⚙️ Calculating Group A Technical Features...")
    for ticker in prices.columns:
        df = pd.DataFrame({'close': prices[ticker]}).dropna()
        if len(df) < 250: continue
            
        # Group A Features
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        df['trend_50'] = (df['close'] / df['sma_50']) - 1
        df['trend_200'] = (df['close'] / df['sma_200']) - 1
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # BB Width
        std = df['close'].rolling(20).std()
        sma_20 = df['close'].rolling(20).mean()
        df['bb_width'] = (4 * std) / sma_20
        
        # MACD Histogram
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd - signal
        
        # 1M Momentum (21 days)
        df['mom_1m'] = df['close'].pct_change(21)
        
        # Target Variable: 21-Day Forward Return
        # Shift -21 pulls future prices back to today's row
        df['fwd_ret_21d'] = df['close'].shift(-21) / df['close'] - 1
        df['target'] = (df['fwd_ret_21d'] > 0).astype(int)
        
        # Drop NaNs created by rolling windows and shifting
        df = df.dropna()
        dataset.append(df)

    if not dataset:
        raise ValueError("Feature calculation resulted in empty dataset.")
        
    master_df = pd.concat(dataset)
    return master_df

def run_walk_forward_shap() -> dict:
    """
    Runs TimeSeriesSplit XGBoost, extracts SHAP values, and normalizes
    them into base-100 percentage weights for the Technical Engine.
    """
    # Use top liquid Nifty 50 stocks for clean ML data
    training_tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
        "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS", "BAJFINANCE.NS"
    ]
    
    df = prepare_ml_dataset(training_tickers)
    
    features = ['trend_50', 'trend_200', 'rsi', 'bb_width', 'macd_hist', 'mom_1m']
    X = df[features]
    y = df['target']
    
    print("🤖 Running Walk-Forward Cross Validation (XGBoost)...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=3, 
        learning_rate=0.05, 
        random_state=42,
        eval_metric='logloss'
    )
    
    accuracies = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))
        
    avg_acc = np.mean(accuracies)
    print(f"   ↳ Walk-Forward Accuracy: {avg_acc*100:.1f}%")
    
    print("🧠 Extracting SHAP Feature Importance...")
    # Train on full dataset for final SHAP extraction
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Normalize to 100%
    total_shap = np.sum(mean_abs_shap)
    weights = (mean_abs_shap / total_shap) * 100
    
    # THE FIX: Force native Python floats and strings so JSON doesn't panic
    weight_dict = {str(feat): float(round(w, 1)) for feat, w in zip(features, weights)}
    
    # Save to static file for the technical_engine to use
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(weight_dict, f, indent=4)
        
    print(f"💾 Dynamic weights saved to {WEIGHTS_FILE}")
    return weight_dict

if __name__ == "__main__":
    print("\n🔬 INITIALIZING NIVI QUANTITATIVE BACKTESTER...\n")
    weights = run_walk_forward_shap()
    
    print("\n" + "="*50)
    print(" 🎯 DERIVED SHAP PARAMETER WEIGHTS")
    print("="*50)
    
    sorted_weights = dict(sorted(weights.items(), key=lambda item: item[1], reverse=True))
    for feat, w in sorted_weights.items():
        bar = "█" * int(w // 2)
        print(f"   {feat:<12} | {w:>4.1f}% | {bar}")
        
    print("="*50 + "\n✅ Backtest Complete.")