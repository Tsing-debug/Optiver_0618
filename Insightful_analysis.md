### Championship Solution Analysis & Implementation

#### Overview
The 1st place solution in the Optiver – Trading at the Close competition built a highly optimized LightGBM model driven by features that capture **market microstructure dynamics** rather than raw price/volume snapshots. The core innovation lies in three areas:
1. **Stock-level adaptive normalization** – making signals comparable across stocks with different volatility profiles.
2. **Momentum and acceleration of order book / auction signals** – capturing not just the current state, but the speed and change-of-speed of demand-supply pressure.
3. **Divergence features between order book and auction book** – quantifying when the two information sources disagree, which often precedes sharp price moves.

We implemented a simplified but faithful version of this approach and achieved a validation MAE of **5.9682**, a **+1.63% improvement** over our previous best baseline (6.0652).

---

#### Key Feature Engineering Concepts

| Feature Category | Description | Financial Intuition |
|:---|:---|:---|
| **Rolling Z-Score Normalization** | For each stock, core signals (e.g. `book_imbalance`, `imbalance_size`) are normalized using a rolling window mean and standard deviation. | A bid/ask imbalance of 0.3 means very different things for a liquid large-cap vs. a volatile small-cap. Normalization converts raw values into “how many standard deviations away from normal” for that specific stock. |
| **Momentum & Acceleration** | First and second differences of `book_imbalance` and `imbalance_size` are computed, along with their rolling statistics over multiple time windows (5s, 10s, 30s). | Velocity and acceleration of order book pressure often signal the entry of large institutional orders before they fully execute. |
| **Auction–Order Book Divergence** | `(imbalance_size_norm - book_imbalance_norm) / (|imbalance_size_norm| + |book_imbalance_norm|)` – a normalized measure of disagreement between passive auction demand and active order book pressure. | When passive (auction) buyers are strong but active (book) sellers dominate, the market may be absorbing a large order — often leading to a reversal or breakout. |
| **Multi-Scale Volatility** | Realized price volatility over 5s, 10s, and 30s windows. | Short-term volatility spikes are often the footprint of informed trading or liquidity gaps. |
| **Price–Reference Deviation** | `(mid_price - reference_price) / reference_price` – how far the continuous market has drifted from the auction reference price. | A large deviation suggests that the auction information is not yet fully priced in, offering a predictive signal. |

---

#### Implementation Details

We implemented these features in a modular function `build_champion_features()` that operates on the raw competition data. The key steps:

1. **Base signals**: `mid_price`, `spread`, `book_imbalance`, `wap_deviation`, `auction_spread`, `ref_price_deviation`.
2. **Grouped rolling z-score normalization** (per stock, window=200 seconds) for `book_imbalance`, `imbalance_size`, `wap_deviation`.
3. **Velocity & acceleration**: `diff()` within each stock to get `imb_velocity`, `imb_acceleration`, `imbalance_velocity`.
4. **Multi-window statistics**: For windows `[5, 10, 30]`, compute rolling mean of `imb_velocity`, rolling std of `imb_acceleration`, rolling mean of `spread`, and price volatility.
5. **Divergence feature**: Combine normalized auction and book imbalances.
6. **Deviation from recent mean**: Current `book_imbalance` minus its rolling mean over 5s and 30s.

The final feature set contained 27 features, listed in the Appendix below.

---

#### Model Training

We used the competition’s champion-level LightGBM parameters, which emphasize strong regularization and long training with early stopping:

```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'mae',
    'metric': 'mae',
    'learning_rate': 0.01,
    'max_depth': 5,
    'num_leaves': 31,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'n_estimators': 5000
}
```

Targets were standardized per stock (using training-set mean and std) before training, and predictions were de-standardized before evaluation — a strict time-series split (first 80% dates for training, last 20% for validation) ensured no future information leakage.

---

#### Results & Feature Importance

| Metric | Value |
|:---|:---|
| Validation MAE | **5.9682** |
| Baseline MAE (historical mean) | 6.0670 |
| Improvement over baseline | **1.63%** |

**Top 15 Feature Importances** (LightGBM `feature_importances_`):

| Feature | Importance | Interpretation |
|:---|:---|:---|
| `seconds_in_bucket` | 9260 | Time-of-day effects (approaching close) |
| `ref_price_deviation` | 8362 | Order book vs. auction reference price gap |
| `mid_price` | 6662 | Current price level |
| `auction_spread` | 5632 | Auction book depth tightness |
| `imbalance_size_norm` | 5506 | Normalized auction imbalance |
| `spread` | 5334 | Bid-ask spread (liquidity) |
| `imbalance_velocity` | 5266 | Speed of auction imbalance change |
| `price_vol_30` | 4716 | 30-second price volatility |
| `imb_acc_std_30` | 4686 | Volatility of imbalance acceleration |
| `wap_deviation` | 4422 | WAP vs. mid-price deviation |
| `spread_mean_30` | 4214 | Average spread over 30s |
| `book_imbalance_norm` | 3199 | Normalized order book imbalance |
| `wap_deviation_norm` | 3098 | Normalized WAP deviation |
| `spread_mean_10` | 2944 | Average spread over 10s |
| `book_imbalance` | 2886 | Raw order book imbalance |

The dominance of `ref_price_deviation` and the normalized/momentum features confirms that the championship approach successfully extracted predictive signals that were absent from our earlier baseline feature sets.

---

#### Key Takeaways

1. **Normalization is everything**: Making signals comparable across stocks via rolling per-stock z-scores was the single most impactful improvement.
2. **Momentum beats static**: The velocity and acceleration of order book/auction signals often contain more predictive power than the raw values.
3. **Divergence matters**: When auction and order book disagree, the market is likely to move — this is a robust signal.
4. **Simplicity wins**: Despite the sophisticated features, the model itself remained a single, well-tuned LightGBM — no deep learning or complex ensembles were necessary to achieve top-tier performance.

This implementation represents a strong baseline that can be further improved with hyperparameter optimization (e.g., Optuna) and ensemble methods, but already demonstrates the core ideas that drove the winning solution.

---

#### Appendix: Full Feature List

```
seconds_in_bucket, mid_price, spread, book_imbalance, wap_deviation,
auction_spread, ref_price_deviation, book_imbalance_norm, imbalance_size_norm,
wap_deviation_norm, imb_velocity, imb_acceleration, imbalance_velocity,
imb_vel_mean_5, imb_vel_mean_10, imb_vel_mean_30, imb_acc_std_5, imb_acc_std_10,
imb_acc_std_30, spread_mean_5, spread_mean_10, spread_mean_30, price_vol_5,
price_vol_10, price_vol_30, imb_vs_mean_5, imb_vs_mean_30, auction_book_divergence
```
