# DA / Intraday Energy Market Forecasting

Production-style forecasting framework for German day-ahead and intraday electricity markets.

The project currently focuses on forecasting German **DA–ID price relationships** for EPEX intraday auctions:

- `IDA1`
- `IDA2`
- `IDA3`
- optional future support for continuous intraday prices

Two target definitions are tested:

- **Direct spread forecast**
  - `y = ID price - DA price`
  - example: `IDA1 - DA`
- **Direct price forecast**
  - `y = ID price`
  - implied spread signal: `predicted ID price - known DA price`

The trading interpretation is:

- forecast the expected ID–DA spread;
- trade only when the predicted spread is large enough;
- evaluate both forecast quality and thresholded trading performance.

---

## Data

### Market data

- **EPEX Spot**
  - German/Luxembourg intraday auction prices:
    - `IDA1`
    - `IDA2`
    - `IDA3`
  - partial support for continuous intraday weighted-average prices

### Fundamentals and forecasts

- **ENTSO-E / German TSO data**
  - day-ahead prices
  - load forecasts
  - solar forecasts
  - wind onshore forecasts
  - wind offshore forecasts
  - renewable forecast revisions
  - actual generation and load where available

### Master dataset

The modelling master file is:

```text
data/master/germany_ida_master_15min.parquet
```

Current setup:

- UTC-indexed master table via `timestamp_utc`
- 15-minute resolution where available
- target-specific feature building with lookahead checks
- ENTSO-E backfill for EPEX-imported IDA2/IDA3 periods

Approximate current data situation:

- `IDA1`: longest and most stable history
- `IDA2`: shorter but usable after historical EPEX import and ENTSO-E backfill
- `IDA3`: shortest and currently most unstable modelling target
- continuous intraday: available only partially and not yet a primary target

---

## Feature logic

Feature families currently include:

- calendar features:
  - hour
  - quarter-hour
  - weekday
  - month
  - weekend flag
- own-market history:
  - product price lags
  - product spread lags
  - same-quarter-hour rolling means / medians / volatility
- DA price features:
  - current known DA price
  - DA price lags / rolling features
- forecast-level features:
  - DA renewable forecasts
  - ID renewable forecasts where available
  - load and residual load forecasts
- forecast-revision features:
  - ID forecast minus DA forecast
  - wind revision
  - solar revision
  - residual load revision
- lagged forecast-error features:
  - actual generation/load minus previous forecasts
  - only lagged or rolling versions are allowed
- regime indicators:
  - rolling volatility
  - short-vs-long volatility ratio
  - mean-shift indicators

Price targets require non-null DA price because trading evaluation uses implied spread:

```text
predicted ID price - known DA price
```

---

## Models

Current model set:

- **Seasonal naive**
  - same-quarter-hour persistence baseline
  - important benchmark because ID spreads have strong local seasonality
- **ElasticNet**
  - regularized linear model
  - useful as a conservative ML baseline
  - good for handling many collinear features
- **XGBoost**
  - decision tree (nonlinear) model
  - feature selection and Optuna tuning are stored as reusable artifacts

Model artifacts are stored under:

```text
model_artifacts/
```

---

## Evaluation

Backtests are walk-forward and currently use expanding windows for the latest full sweep.

Forecast metrics:

- RMSE
- MAE
- Pearson correlation
- Spearman correlation
- directional accuracy

Trading-style metrics:

- thresholded PnL
- trade fraction
- hit rate conditional on traded signals
- threshold estimated from train-fold prediction quantiles

Supported window modes:

- `expanding`
- `rolling_gap_aware`
- `rolling_contiguous`

---

## Current intermediate results

These results are preliminary research outputs, not production trading results.

### IDA1 spread

| Model | Folds | RMSE | MAE | Directional acc. | Signal Spearman | Total PnL | Traded hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgboost | 18 | 16.42 | 8.96 | 0.769 | 0.707 | 341,021 | 0.911 |
| elasticnet | 18 | 16.47 | 9.06 | 0.769 | 0.705 | 345,002 | 0.914 |
| seasonal_naive | 18 | 16.72 | 9.26 | 0.770 | 0.703 | 343,314 | 0.908 |

Notes:

- all three models are close;
- XGBoost and ElasticNet slightly improve RMSE/MAE versus seasonal naive;
- trading-style metrics remain strong across all models;
- `IDA1_spread` is currently one of the most promising direct-spread targets.

### IDA1 price

| Model | Folds | RMSE | MAE | Directional acc. | Signal Spearman | Total PnL | Traded hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| elasticnet | 18 | 16.93 | 9.54 | 0.756 | 0.686 | 329,007 | 0.895 |
| xgboost | 18 | 21.57 | 12.29 | 0.665 | 0.447 | 235,160 | 0.741 |
| seasonal_naive | 18 | 47.33 | 31.07 | 0.602 | 0.264 | 117,100 | 0.620 |

Notes:

- ElasticNet currently dominates this target by RMSE/MAE and trading metrics;
- XGBoost is useful but weaker than ElasticNet in this run;
- direct price forecasting works, but performance should always be judged through implied spread quality too.

### IDA2 spread

| Model | Folds | RMSE | MAE | Directional acc. | Signal Spearman | Total PnL | Traded hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| seasonal_naive | 8 | 17.50 | 10.95 | 0.748 | 0.659 | 169,616 | 0.880 |
| xgboost | 8 | 17.80 | 11.06 | 0.743 | 0.640 | 174,620 | 0.869 |
| elasticnet | 8 | 17.84 | 10.83 | 0.737 | 0.638 | 171,860 | 0.879 |

Notes:

- seasonal naive remains very competitive;
- XGBoost has the highest total PnL in this run but does not improve RMSE;
- `IDA2_spread` needs more robustness checks before choosing a champion model.

### IDA2 price

| Model | Folds | RMSE | MAE | Directional acc. | Signal Spearman | Total PnL | Traded hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| elasticnet | 8 | 18.08 | 10.95 | 0.738 | 0.644 | 166,633 | 0.877 |
| xgboost | 8 | 20.76 | 12.40 | 0.711 | 0.589 | 150,738 | 0.838 |
| seasonal_naive | 8 | 42.43 | 27.53 | 0.628 | 0.302 | 53,279 | 0.638 |

Notes:

- ElasticNet is currently the strongest model;
- XGBoost is second-best and beats seasonal naive clearly;
- price target looks more promising than spread target for `IDA2` in this run.

### IDA3 spread

| Model | Folds | RMSE | MAE | Directional acc. | Signal Spearman | Total PnL | Traded hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| xgboost | 9 | 28.12 | 16.38 | 0.524 | 0.044 | 7,890 | 0.470 |
| seasonal_naive | 9 | 28.19 | 16.79 | 0.579 | 0.133 | 13,083 | 0.617 |
| elasticnet | 9 | 28.39 | 16.68 | 0.524 | 0.109 | 4,187 | 0.555 |

Notes:

- seasonal naive is still hard to beat;
- ML models do not yet provide a stable improvement;
- `IDA3_spread` is currently noisy and should be treated as exploratory.

### IDA3 price

| Model | Folds | RMSE | MAE | Directional acc. | Signal Spearman | Total PnL | Traded hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| elasticnet | 9 | 34.31 | 21.21 | 0.545 | 0.211 | 2,501 | 0.608 |
| xgboost | 9 | 40.05 | 23.25 | 0.520 | 0.137 | 2,137 | 0.528 |
| seasonal_naive | 9 | 52.20 | 35.37 | 0.562 | 0.188 | -3,015 | 0.597 |

Notes:

- ElasticNet has the best RMSE/MAE and positive PnL in this snapshot;
- XGBoost is weaker than ElasticNet in this run;
- `IDA3_price` remains unstable because the usable history is short.

---

## Current conclusions

- `IDA1` is the most mature target family.
- `IDA1_spread` and `IDA1_price` both look usable for continued development.
- `IDA2_price` is currently more promising than `IDA2_spread`.
- `IDA3` needs more work:
  - short history;
  - higher noise;
  - unstable ML improvement over baseline.
- ElasticNet is surprisingly competitive and should remain in the model set.
- XGBoost benefits from Optuna tuning but is not automatically superior.
- Seasonal naive remains a strong benchmark and should not be removed.

---

## Immediate next steps

- rerun full model sweep after every major data backfill;
- verify all selected features for lookahead safety;
- add daily model training / forecast generation pipeline;
- store generated forecasts as timestamped artifacts;
- evaluate realized forecasts once actual auction outcomes arrive;
- test ensemble combinations:
  - seasonal naive + ElasticNet
  - ElasticNet + XGBoost
- investigate whether `IDA3` needs:
  - different target definition;
  - shorter horizon;
  - auction-specific features;
  - stronger regime separation;
- add simple plots:
  - model comparison;
  - cumulative PnL;
  - realized vs predicted spread;
  - feature importance / selected feature stability.

---

## Status

Work in progress. Current repository snapshot shows:

- data ingestion and master parquet maintenance;
- ENTSO-E/EPEX coverage checks and backfill logic;
- target-specific feature engineering;
- feature selection;
- Optuna tuning for XGBoost;
- walk-forward evaluation;
- first cross-target modelling comparison.
