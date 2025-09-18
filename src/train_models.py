import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Try importing XGBoost safely
try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception as e:
    print("XGBoost not available:", e)
    xgb_available = False


def engineer_features(df):
    print("Engineering features...")
    df['Return'] = df['Close'].pct_change()
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['VolumeChange'] = df['Volume'].pct_change()
    df['Target'] = np.where(df['Return'].shift(-1) > 0, 1, 0)
    df = df.dropna()
    print("Features engineered:", df.shape)
    return df


def load_data(ticker, start, end):
    os.makedirs("data", exist_ok=True)
    cache_file = f"data/{ticker}.csv"

    if os.path.exists(cache_file):
        print(f"Loading cached data for {ticker}...")
        data = pd.read_csv(cache_file, index_col=0)
    else:
        print(f"Fetching data for {ticker} from {start} to {end}...")
        data = yf.download(ticker, start=start, end=end, interval="1d")
        if data.empty:
            raise ValueError("No data returned. Check ticker symbol or date range.")
        data.to_csv(cache_file)
        print(f"Data downloaded and cached: {data.shape}")

    # Fix date parsing explicitly
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.dropna(subset=["Close"])  # drop invalid rows

    # Ensure numeric dtype for key columns
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    return data


def main(ticker, start, end, out_path):
    data = load_data(ticker, start, end)
    data = engineer_features(data)

    X = data[['Return', 'SMA5', 'SMA10', 'VolumeChange']]
    y = data['Target']
    print("Training dataset prepared:", X.shape, "features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print("Train size:", X_train.shape, "Test size:", X_test.shape)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    if xgb_available:
        models["XGBoost"] = XGBClassifier(
            eval_metric="logloss",
            random_state=42
        )

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1": round(f1, 3)
        })

        print(f"{name} Accuracy: {acc:.2f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
        print("Classification Report:\n", classification_report(y_test, preds))

    # === Save JSON results ===
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_json = {
        "ticker": ticker,
        "start": start,
        "end": end,
        "results": results
    }
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # === Save README snippet ===
    df_results = pd.DataFrame(results)
    os.makedirs("plots", exist_ok=True)
    snippet_path = f"plots/{ticker}_README_snippet.md"
    with open(snippet_path, "w") as f:
        f.write(f"## Results for {ticker}\n\n")
        f.write("Below are the performance metrics for all models trained:\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n\n---\n\n")
        f.write(f"![Price and Volume Plot](plots/{ticker}_price_plot.png)\n")
    print(f"README snippet saved to {snippet_path}")

    # --- Visualization ---
    print("\nGenerating plots...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Price + Moving Averages
    ax1.plot(data.index, data['Close'], label="Close Price", linewidth=1.5)
    ax1.plot(data.index, data['SMA5'], label="SMA5", linestyle="--")
    ax1.plot(data.index, data['SMA10'], label="SMA10", linestyle="--")
    ax1.set_title(f"{ticker} Price with Moving Averages")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()

    # Volume subplot
    ax2.bar(data.index, data['Volume'], width=1.0, color="gray")
    ax2.set_ylabel("Volume")

    # Date formatting
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    out_path_plot = f"plots/{ticker}_price_plot.png"
    plt.savefig(out_path_plot)
    plt.close()
    print(f"Plot saved to {out_path_plot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock/Crypto Price Prediction with ML")
    parser.add_argument("--ticker", type=str, required=True,
                        help="Ticker symbol (e.g., AAPL for stock, BTC-USD for crypto)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--out", type=str, required=True,
                        help="Path to output JSON file (e.g., results/btc_results.json)")
    args = parser.parse_args()
    main(args.ticker, args.start, args.end, args.out)
