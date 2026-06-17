# Optiver - Trading at the Close: Closing Price Prediction
This project replicates the core workflow of the Kaggle competition "Optiver - Trading at the Close". 
I built a complete pipeline to predict US stock closing price movements using order book and closing auction data. 
Key techniques include strict time-series validation, 30+ financial features (imbalance, auction pressure, volatility), feature selection, and automated hyperparameter tuning with Optuna.
The final LightGBM model achieves a validation MAE of 6.0652, outperforming the naive historical-mean baseline.
## Project Overview
This project replicates the core workflow of the [Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close) Kaggle competition. The goal is to predict the 60-second future price movement of Nasdaq stocks using order book and closing auction data.

## Key Results
- **Baseline Model**: LightGBM with 32 features → Validation MAE: 6.0689
- **After Feature Selection & Tuning**: 15 selected features + Optuna hyperparameter optimization → Validation MAE: **6.0652**
- **Improvement**: Achieved a slight but consistent improvement over the naive historical-mean baseline.

## Technical Highlights
- **Strict Time-Series Split**: Data split by `date_id` to prevent future information leakage.
- **Feature Engineering**: Built 30+ features covering:
  - Price dynamics (mid price, spread, WAP deviation)
  - Order book imbalance (bid/ask volume ratio, rolling statistics)
  - Auction pressure (imbalance size, far-near spread, change acceleration)
  - Multi-scale realized volatility
  - Stock-level normalization
- **Model Optimization**: Used Optuna Bayesian search to tune LightGBM hyperparameters.
- **Memory Optimization**: Used `float32`/`int16` dtypes to handle 5M+ rows efficiently.

## Repository Structure
- `optiver-baseline.ipynb`: Main notebook containing data loading, feature engineering, model training, and evaluation.
- `README.md`: Project documentation.

## Environment
- Kaggle Notebook (Python 3.10+)
- Key libraries: `pandas`, `numpy`, `lightgbm`, `optuna`

## How to Run
1. Download the dataset from the [competition page](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data).
2. Upload the notebook to Kaggle or run locally with the required libraries.
3. Execute cells in order.

## Learnings & Skills
- Financial market microstructure (order book, auction book, bid/ask dynamics)
- Time-series feature engineering for high-frequency data
- Experimental iteration and logging in quantitative research
- End-to-end machine learning pipeline for real-world prediction tasks

## Acknowledgements
Inspired by discussions and public notebooks from the Kaggle community, especially the 1st place solution.

| Experiment | Features | MAE | Notes |
|:---|:---|:---|:---|
| Baseline | Raw fields | 6.0689 | Nearly equal to naive baseline |
| Exp1 | Remove time feature | 6.0680 | Time feature not useful |
| Exp2 | Add interaction features | 6.0681 | No improvement |
| Exp3 | Top-15 features + simplified model | 6.0667 | Reduction helps |
| Exp4 | Stock-level normalization + Optuna | **6.0652** | First solid win over baseline |
| Champion-lite | Multi-scale imbalance & vol | 6.0657 | Marginal returns diminish |

## Data
The data is from the [Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close/data) competition on Kaggle. You must accept the competition rules to download it.
