# Retail Demand Forecasting & Replenishment

## Model Evaluation and Project Report

## 1. Project Overview

The objective of this project is to develop a machine learning workflow
that forecasts daily retail product demand across multiple stores.

The workflow covers data loading, preprocessing, exploratory data
analysis, time-series feature engineering, baseline modelling, Random
Forest training, hyperparameter tuning, model evaluation, visualisation,
and seven-day demand forecasting.

The project currently focuses on demand prediction. It does not yet
implement a complete inventory optimisation or automated replenishment
system.

## 2. Dataset Overview

The project uses `retail_daily_demand_dataset.csv`.

  Attribute           Description
  ------------------- --------------------------
  Number of records   13,140
  Date coverage       2024-01-01 to 2025-12-30
  Stores              3
  Products            6
  Frequency           Daily

The dataset contains the following columns:

-   `date`
-   `store_id`
-   `product_id`
-   `category`
-   `unit_price`
-   `promotion`
-   `promotion_discount`
-   `stock_on_hand`
-   `units_sold`

The target variable is `units_sold`.

## 3. Exploratory Data Analysis

Exploratory Data Analysis was performed in `eda.ipynb`.

The notebook is used to inspect the dataset, understand its
distributions, and explore demand patterns across dates, stores,
products, prices, and promotions.

The EDA work forms the basis for the feature engineering and modelling
stages.

## 4. Data Preprocessing

The preprocessing pipeline is implemented in `file/preprocessing.py`.

The pipeline performs these operations:

1.  Loads the CSV dataset using Pandas.
2.  Converts `date` to datetime format.
3.  Removes duplicate rows.
4.  Removes rows containing missing values.
5.  Sorts records by date.
6.  Resets the DataFrame index.

The current implementation drops rows with missing values rather than
imputing them during preprocessing.

## 5. Feature Engineering

Feature engineering is implemented in `file/feature_engineering.py`.

Calendar features:

  Feature         Description
  --------------- ------------------------------------------
  `day`           Day of the month
  `month`         Month of the year
  `day_of_week`   Day of the week, represented from 0 to 6
  `is_weekend`    1 for Saturday/Sunday, otherwise 0

Historical demand features:

  Feature             Description
  ------------------- ------------------------------------------------
  `lag_1`             Previous observation's sales
  `lag_7`             Sales seven observations earlier
  `lag_14`            Sales fourteen observations earlier
  `lag_28`            Sales twenty-eight observations earlier
  `rolling_mean_7`    Mean of the previous seven observations
  `rolling_mean_28`   Mean of the previous twenty-eight observations

Lag and rolling features are calculated separately for each
store-product combination. Rolling means use a one-period shift so that
the current target value is not included in its own rolling feature.

Rows without sufficient historical observations for the required lag and
rolling features are removed.

## 6. Models Developed

### 6.1 Baseline Model

The baseline is a simple seven-period lag approach. It predicts demand
using `lag_7`, the sales value from seven observations earlier.

This provides a benchmark for measuring whether machine learning adds
predictive value.

### 6.2 Original Random Forest Regressor

The original model is a Scikit-learn pipeline containing:

-   A `ColumnTransformer` for preprocessing.
-   Most-frequent imputation and one-hot encoding for categorical
    columns.
-   Median imputation for numerical columns.
-   A `RandomForestRegressor` with 150 trees and `random_state=42`.

The model uses store, product, category, price, promotion, calendar,
lag, and rolling-mean features.

### 6.3 Tuned Random Forest Regressor

Hyperparameter tuning is implemented in `file/tuning.py` using
`RandomizedSearchCV`.

The search considers:

-   Number of trees (`n_estimators`).
-   Maximum tree depth (`max_depth`).
-   Minimum samples required to split a node (`min_samples_split`).
-   Minimum samples required at a leaf (`min_samples_leaf`).
-   Number of features considered at each split (`max_features`).

Chronological validation folds are created using `TimeSeriesSplit` over
unique training dates. All rows belonging to a given date are assigned
to the same fold.

The selected estimator is saved as `outputs/tuned_retail_model.joblib`.

## 7. Train-Test and Evaluation Methodology

The modelling workflow uses a chronological split:

-   The first 80% of unique dates are used for training.
-   The remaining 20% of unique dates are held out for testing.

The original and tuned Random Forest models are compared on the same
holdout period.

The baseline uses the same test rows.

### Evaluation metrics

  -----------------------------------------------------------------------
  Metric                  Meaning                 Preferred direction
  ----------------------- ----------------------- -----------------------
  MAE                     Mean Absolute Error;    Lower
                          average absolute        
                          difference between      
                          actual and predicted    
                          demand                  

  RMSE                    Root Mean Squared       Lower
                          Error; gives greater    
                          weight to larger errors 

  R²                      Coefficient of          Higher
                          determination; measures 
                          explained variation     
                          relative to a           
                          mean-prediction         
                          reference               

  WAPE                    Weighted Absolute       Lower
                          Percentage Error; total 
                          absolute error divided  
                          by total absolute       
                          actual demand           
  -----------------------------------------------------------------------

Predictions are clipped at zero because negative unit demand is not
meaningful.

## 8. Model Evaluation Results

The following results were obtained from `outputs/model_comparison.csv`.

  Metric       Baseline   Original Random Forest   Tuned Random Forest
  ---------- ---------- ------------------------ ---------------------
  MAE             8.967                    5.810             **5.639**
  RMSE           12.972                    8.372             **7.911**
  R²              0.367                    0.736             **0.764**
  WAPE (%)       33.170                   21.493            **20.861**

Values are rounded to three decimal places.

## 9. Results Interpretation

The original Random Forest outperformed the baseline across all four
metrics.

Compared with the original Random Forest, the tuned Random Forest
achieved:

  Metric       Original RF   Tuned RF        Change
  ---------- ------------- ---------- -------------
  MAE                5.810      5.639   2.94% lower
  RMSE               8.372      7.911   5.51% lower
  R²                 0.736      0.764        +0.028
  WAPE (%)          21.493     20.861   2.94% lower

The tuned model has the lowest MAE, RMSE, and WAPE, and the highest R²
among the three evaluated approaches.

The reduction in MAE and WAPE is modest, while the reduction in RMSE is
larger. This indicates that tuning improved overall test-set
performance, with a relatively stronger reduction in the metric that
penalises larger errors.

These results describe performance on this particular holdout dataset.
They do not guarantee the same performance on future or real-world
retail data.

## 10. Final Model Selection

The Tuned Random Forest Regressor is selected as the final model because
it achieved the best measured result across all four evaluation metrics.

The selected model is saved at:

`outputs/tuned_retail_model.joblib`

The original model is retained at:

`outputs/retail_model.joblib`

Keeping both models makes the comparison reproducible and preserves the
untuned benchmark.

## 11. Forecast Generation

The forecasting function is implemented in `file/forecasting.py`.

It generates a seven-day forecast for each store-product combination.

The process is:

1.  Retrieve the historical records for a store-product combination.
2.  Sort the records by date.
3.  Build the calendar, lag, and rolling-mean features for the next day.
4.  Predict demand using the selected trained model.
5.  Append the prediction to the sales history.
6.  Repeat the process recursively for the following forecast days.

The forecast is saved to:

`outputs/next_7_days_forecast.csv`

The output includes:

-   `date`
-   `store_id`
-   `product_id`
-   `predicted_units_sold`

The forecasting implementation carries forward the latest observed
category, price, promotion, and discount values. This is an assumption
about future conditions, not a known future schedule.

## 12. Visualisation Outputs

The visualisation module generates:

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `outputs/actual_vs_predicted.png`   Compares daily aggregate actual and
                                      predicted demand

  `outputs/prediction_scatter.png`    Compares individual actual and
                                      predicted values

  `outputs/feature_importance.png`    Displays Random Forest feature
                                      importance
  -----------------------------------------------------------------------

These plots provide additional ways to inspect model behaviour beyond
the aggregate evaluation metrics.

## 13. Project Limitations

### 13.1 Synthetic data

The project dataset is synthetic. Results on this dataset may not
represent performance on actual retail transactions or operational store
data.

### 13.2 Limited store and product coverage

The dataset contains three stores and six products. A larger and more
diverse dataset would be needed to assess generalisation across more
locations and product categories.

### 13.3 External demand drivers

The current feature set does not include all possible external drivers,
such as public holidays, weather, local events, competitor pricing, or
broader market conditions.

### 13.4 Stock availability and lost sales

Recorded units sold may not equal true customer demand when products are
out of stock. The current modelling pipeline does not explicitly
estimate lost sales caused by stockouts.

### 13.5 Future price and promotion assumptions

The seven-day forecasting function carries forward the latest observed
price, promotion flag, and discount. If these values change in the
future, forecast accuracy may be affected.

### 13.6 Recursive forecasting errors

Predictions are appended to the history and reused to create later
forecast features. Errors on earlier forecast days can therefore
influence predictions on subsequent days.

### 13.7 Point forecasts only

The model produces point predictions. It does not currently provide
prediction intervals or a quantified measure of forecast uncertainty.

### 13.8 No complete replenishment optimisation

The project forecasts demand but does not yet calculate safety stock,
reorder points, order quantities, supplier lead-time requirements, or
inventory costs. Therefore, it should be described as a demand
forecasting project with replenishment as a potential extension, rather
than a complete replenishment optimisation system.

### 13.9 Limited validation history

The reported metrics are based on one chronological holdout period.
Additional rolling-origin backtesting across multiple periods would
provide stronger evidence of performance stability.

## 14. Future Improvements

Potential next steps include:

1.  Validate the model on real retail sales data.
2.  Add holiday, event, weather, and other external features where
    reliable data is available.
3.  Perform rolling-origin backtesting across multiple test windows.
4.  Compare Random Forest against additional forecasting approaches.
5.  Estimate prediction intervals or forecast uncertainty.
6.  Incorporate future planned prices and promotions.
7.  Add stockout and lost-sales handling.
8.  Build replenishment logic using forecast demand, lead time, safety
    stock, and service-level targets.
9.  Monitor model performance as new sales data becomes available.

## 15. Generated Project Artifacts

  Artifact                    Location
  --------------------------- ----------------------------------------
  Exploratory Data Analysis   `eda.ipynb`
  Dataset                     `data/retail_daily_demand_dataset.csv`
  Preprocessing               `file/preprocessing.py`
  Feature Engineering         `file/feature_engineering.py`
  Baseline Model              `file/baseline.py`
  Model Training              `file/train.py`
  Hyperparameter Tuning       `file/tuning.py`
  Model Evaluation            `file/evaluation.py`
  Model Comparison            `file/comparison.py`
  Forecasting                 `file/forecasting.py`
  Visualisation               `file/visualization.py`
  Original Model              `outputs/retail_model.joblib`
  Tuned Model                 `outputs/tuned_retail_model.joblib`
  Best Parameters             `outputs/best_parameters.json`
  Model Comparison CSV        `outputs/model_comparison.csv`
  Evaluation Metrics          `outputs/evaluation_metrics.json`
  Test Predictions            `outputs/test_predictions.csv`
  Seven-Day Forecast          `outputs/next_7_days_forecast.csv`
  Actual vs Predicted Plot    `outputs/actual_vs_predicted.png`
  Prediction Scatter Plot     `outputs/prediction_scatter.png`
  Feature Importance Plot     `outputs/feature_importance.png`

## 16. Conclusion

This project implements an end-to-end retail demand forecasting
workflow, from data preprocessing and exploratory analysis through
time-series feature engineering, baseline modelling, Random Forest
training, hyperparameter tuning, evaluation, visualisation, and
seven-day forecasting.

The baseline model provides a simple benchmark. Both Random Forest
models improve on that benchmark, and the tuned Random Forest achieves
the best measured performance on the chronological holdout dataset.

The tuned Random Forest is therefore selected as the final forecasting
model for this project. The results should be interpreted in light of
the synthetic dataset, limited coverage, future-condition assumptions,
and the absence of a complete inventory replenishment optimisation
component.
