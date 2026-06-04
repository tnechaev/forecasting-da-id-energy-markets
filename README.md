## DA / ID forecasting framework in energy markets

This projects aims to forecast, as a first step, ID - DA price spreads, for ID auctions 1,2 and 3, for German energy market. Features used are lags related to prices themselves, renewable forecast DA - ID spreads, residual load forecast error etc. Fundamentals forecasts are from German TSOs and are retrieved from ENTSO-e. These forecasts are then to be used in a trading model. Forecasting is to be done via baseline and possibly 2 ML models like XGBoost and Elasticnet, with an ensemble model, possibly, to reduce systematic error.

Possible extension is to forecast generation and load as well, and use own forecasts.

Data used: 
- EPEX spot: ID auctions (A1/2/3) and continuous (weighted avg) prices.
- ENTSO-e: generation/load/renewable forecasts, actuals, DA prices

German market only right now, extendable to other markets.

**Production-style**. Research and large re-optimization bits can be done locally, data fetching and re-training and daily forecasting runs on Github cloud. Visualization and forecast data to be available soon.

### Work in progress (locally, research stage), more details and results to come shortly!
