# Limitations

While working on this Retail Demand Forecasting and Replenishment
project, I identified a few limitations in the current implementation.

## 1. Dataset Limitations

-   The model is currently trained on a synthetic dataset, not actual
    retail sales data. So, the results may differ when it is used with
    real-world data.
-   The dataset contains only 3 stores and 6 products. It may not
    represent the complexity of a larger retail business.
-   The dataset covers a limited period, so it may not capture long-term
    changes in customer demand.

## 2. Forecasting Limitations

-   The model mainly depends on historical sales data. It may not
    predict sudden changes in demand accurately.
-   The forecast is generated recursively. This means the prediction for
    one day is used to generate the prediction for the next day. Because
    of this, errors may accumulate over the 7-day forecast.
-   The current forecasting logic assumes that the latest product price
    and promotion details will continue for the next 7 days.
-   The model does not currently consider external factors such as
    weather, festivals, holidays, competitor prices, or local events.

## 3. Inventory Limitations

-   The current model predicts units sold, but it does not yet
    distinguish actual demand from lost sales caused by products being
    out of stock.
-   The replenishment recommendation logic is not yet implemented.
-   Factors such as supplier lead time, minimum order quantity, storage
    capacity, and safety stock are not currently included.

## 4. Model Limitations

-   I have currently implemented a Random Forest Regressor. I have not
    yet compared it with other forecasting models.
-   Hyperparameter tuning has not been performed.
-   The current evaluation is based on a chronological train-test split.
    More validation across different time periods is needed before
    relying on the results.
-   The model does not currently provide prediction intervals or
    uncertainty estimates.
-   Feature importance shows which features are useful for prediction,
    but it does not explain cause-and-effect relationships.

## 5. Future Improvements

I plan to improve the project by:

-   Testing the model with real retail sales data.
-   Comparing Random Forest with other models such as Linear Regression
    and Gradient Boosting.
-   Adding external factors such as holidays, festivals, and weather.
-   Improving the forecasting logic and validating predictions across
    multiple time periods.
-   Implementing replenishment recommendations using forecast demand,
    current stock, safety stock, and supplier lead time.
-   Adding model monitoring and retraining when new sales data becomes
    available.

## Conclusion

The current implementation provides an initial workflow for retail
demand forecasting. However, it still needs further testing with
real-world data and additional inventory-related logic before it can be
considered suitable for actual retail operations.
