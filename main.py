
import os
import json
import numpy as np
import pandas as pd

from file.preprocessing import load_data, clean_data
from file.feature_engineering import create_features
from file.train import train_model
from file.tuning import tune_model
from file.comparison import compare_models
from file.forecasting import forecast_next_7_days
from file.visualization import visualize_predictions


os.makedirs("outputs", exist_ok=True)


print("\n1. Loading dataset...")
df = load_data("data/retail_daily_demand_dataset.csv")
print("Dataset shape:", df.shape)


print("\n2. Cleaning data...")
df = clean_data(df)
print("Cleaned dataset shape:", df.shape)


print("\n3. Feature engineering...")
featured_df = create_features(df)
print("Featured dataset shape:", featured_df.shape)


print("\n4. Training original Random Forest...")
original_model, original_test = train_model(
    featured_df,
    model_path="outputs/retail_model.joblib"
)

print("Original model training completed.")


print("\n5. Tuning Random Forest...")
tuned_model, tuned_test, best_params = tune_model(
    featured_df,
    model_path="outputs/tuned_retail_model.joblib"
)

print("Tuned model training completed.")


print("\n6. Comparing models...")

# Both models must be evaluated on the same chronological test period.
original_test = original_test.sort_values(
    ["date", "store_id", "product_id"]
).reset_index(drop=True)

tuned_test = tuned_test.sort_values(
    ["date", "store_id", "product_id"]
).reset_index(drop=True)

if not original_test[
    ["date", "store_id", "product_id"]
].equals(
    tuned_test[["date", "store_id", "product_id"]]
):
    raise ValueError("Original and tuned models have different test datasets.")

test_df = original_test

comparison = compare_models(
    original_model,
    tuned_model,
    test_df
)

comparison.to_csv("outputs/model_comparison.csv")


print("\n7. Saving evaluation results...")

with open("outputs/evaluation_metrics.json", "w") as file:
    json.dump(comparison.to_dict(orient="index"), file, indent=4)

with open("outputs/best_parameters.json", "w") as file:
    json.dump(best_params, file, indent=4)


print("\n8. Saving test predictions...")

from file.train import FEATURES, TARGET

predictions = np.maximum(
    0,
    tuned_model.predict(test_df[FEATURES])
)

prediction_df = test_df[
    ["date", "store_id", "product_id", TARGET]
].copy()

prediction_df["predicted_units_sold"] = predictions

prediction_df.to_csv(
    "outputs/test_predictions.csv",
    index=False
)


print("\n9. Forecasting next 7 days...")

forecast_df = forecast_next_7_days(df, tuned_model)

forecast_df.to_csv(
    "outputs/next_7_days_forecast.csv",
    index=False
)

print(forecast_df.head(15))
print("Total forecast records:", len(forecast_df))


print("\n10. Generating visualisations...")

visualize_predictions(tuned_model, test_df)


print("\nPROJECT COMPLETED SUCCESSFULLY!")

print("\nGenerated files:")

for filename in sorted(os.listdir("outputs")):
    print("-", filename)
