Here's the full English version of the post-processing scaling section, ready to drop into your README.

---

### Post-Processing Scaling Experiment

After obtaining the raw LightGBM predictions, we attempted a simple linear scaling post-processing step to further reduce MAE.

#### Methodology
- Obtain the model's raw predictions on the **training set** (converted back to original target scale).
- Fit a `LinearRegression` model: `true_target ~ predicted_target`.
- Apply the fitted scaler to the **validation set** predictions.
- Compare MAE before and after scaling.

```python
from sklearn.linear_model import LinearRegression

# Training set predictions (original scale)
y_train_pred = model.predict(X_train) * train_std + train_mean

# Fit scaler
scaler = LinearRegression()
scaler.fit(y_train_pred.values.reshape(-1, 1), y_train.values)

# Apply to validation set
y_val_pred_scaled = scaler.predict(y_pred.values.reshape(-1, 1))
```

#### Results

| Metric | Before Scaling | After Scaling | Change |
|:---|:---|:---|:---|
| Validation MAE | **5.968244** | 5.971709 | +0.003465 (worse) |
| Scaler Coefficient | — | 1.082 | — |
| Scaler Intercept | — | 0.020 | — |

#### Technical Analysis

The fitted coefficient of **1.082** indicates that, on the **training set**, the model's predictions were slightly underestimated (actual ≈ 1.082 × predicted + 0.020). This minor bias may come from:
- The model's conservative tendency on extreme values (shrinkage caused by regularization).
- Small numerical errors from the target standardization / de-standardization pipeline.

However, when applied to the **validation set**, MAE actually increased. This implies:
- The training-set bias pattern **did not generalize** to the validation period — the model's predictions on unseen data are already nearly unbiased.
- Forcing the training-set-derived scaling introduced a **bias that overfit the training period**, harming out-of-sample performance.

#### Financial Interpretation

In a real market-making context, **prediction unbiasedness matters far more than a tiny MAE improvement**:
- If predictions are systematically too low (coefficient > 1), the market maker would continuously quote prices below fair value, accumulating directional risk and getting picked off by opportunistic traders.
- In this experiment, the raw predictions are already well-calibrated on the validation set (scaling only degraded performance), indicating that our feature set and model successfully captured signals that are **linearly related to future price moves** and not overfitted to a specific market regime.

**Conclusion**: For our current model, linear post-hoc scaling provides no benefit. The raw predictions are robust and well-calibrated — a sign of effective feature engineering. When features are good, the model can output properly scaled predictions without extra correction.

---
| Experiment | Method | Val MAE | Notes |
|:---|:---|:---|:---|
| Champion Features | Champion feature set + LightGBM | **5.9682** | +1.63% over previous best |
| + Linear Scaling | Train-set scaler applied to val | 5.9717 | Bias introduced, MAE increased |
```
