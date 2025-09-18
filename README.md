# Stock & Crypto Price Prediction with Machine Learning  

This project applies **Python (pandas, scikit-learn, XGBoost)** to predict short-term stock and cryptocurrency price movements.  
The main goal is to demonstrate how **data preprocessing, feature engineering, and ML models** can be combined into a reproducible pipeline.  

By default, the program fetches data using **Yahoo Finance**, engineers technical indicators, and trains multiple models to classify whether the next day’s return will be **positive or negative**.  

---

## What it does  

- **Fetches data** from Yahoo Finance (`yfinance`) and caches it locally.  
- **Feature engineering**: returns, moving averages, volume change.  
- **Trains multiple ML models** side by side:  
  - Logistic Regression  
  - Random Forest  
  - Gradient Boosting  
  - XGBoost (if available)  
- **Evaluates models** with accuracy, precision, recall, and F1-score.  
- **Outputs comparison tables** (CSV + Markdown) and **plots** for easy analysis.  

---

## How to run  

Make sure you have Python 3.9+ and the required packages installed:  

    pip install -r requirements.txt

Then run the program:  

    python src/train_models.py --ticker BTC-USD --start 2022-01-01 --end 2023-01-01

---

## Options  

- **`--ticker SYMBOL`**  
  Stock or crypto symbol (e.g., `AAPL`, `TSLA`, `BTC-USD`).  

- **`--start YYYY-MM-DD`**  
  Start date for historical data.  

- **`--end YYYY-MM-DD`**  
  End date for historical data.  

**Examples:**  

    python src/train_models.py --ticker BTC-USD --start 2022-01-01 --end 2023-01-01  
    python src/train_models.py --ticker AAPL --start 2023-01-01 --end 2023-12-31

---

## Example output  

**Console summary:**  

    === Model Comparison Table ===
                  Model  Accuracy  Precision  Recall    F1
    Logistic Regression     0.542      0.400   0.062  0.108
          Random Forest     0.403      0.400   0.688  0.506
      Gradient Boosting     0.417      0.419   0.812  0.553
                XGBoost     0.444      0.431   0.781  0.556

**Plots (saved in `plots/`):**

- `BTC-USD_price_plot.png`  

**Markdown table (saved in `plots/`):**

    | Model               | Accuracy | Precision | Recall | F1   |
    |----------------------|----------|-----------|--------|------|
    | Logistic Regression  | 0.542    | 0.400     | 0.062  | 0.108|
    | Random Forest        | 0.403    | 0.400     | 0.688  | 0.506|
    | Gradient Boosting    | 0.417    | 0.419     | 0.812  | 0.553|
    | XGBoost              | 0.444    | 0.431     | 0.781  | 0.556|

---

## Project structure  

    sc-prediction-ml/
    ├── data/                  # cached Yahoo Finance data (ignored in git)
    │   └── BTC-USD.csv
    ├── plots/                 # results and visuals
    │   ├── BTC-USD_price_plot.png
    │   ├── BTC-USD_model_results.md
    │   └── Example_README_snippet.md
    ├── results/               # (optional) structured JSON logs
    │   └── Example_results.json
    ├── src/                   # source code
    │   └── train_models.py
    ├── requirements.txt       # Python dependencies
    └── README.md              # project documentation

---

## Notes  

- Results vary across tickers and timeframes — financial prediction is inherently noisy.  
- Logistic Regression often provides the most stable baseline.  
- XGBoost can capture more non-linear patterns, but requires careful tuning for real gains.  
- The pipeline is designed to be **reproducible** and **extensible** (add new features, try new models).  
