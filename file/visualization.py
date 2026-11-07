
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .train import FEATURES, TARGET


def visualize_predictions(model, test_df):

    os.makedirs("outputs", exist_ok=True)

    actual = test_df[TARGET]
    predicted = model.predict(test_df[FEATURES])

    results = test_df[["date"]].copy()
    results["actual"] = actual.values
    results["predicted"] = predicted

    # 1. Actual vs predicted graph
    daily = results.groupby("date")[["actual", "predicted"]].sum()

    plt.figure(figsize=(12, 5))
    plt.plot(daily.index, daily["actual"], label="Actual")
    plt.plot(daily.index, daily["predicted"], label="Predicted")

    plt.title("Actual vs Predicted Demand")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png")
    plt.close()

    # 2. Prediction scatter plot
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.5)

    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    plt.plot([minimum, maximum], [minimum, maximum], "--")

    plt.title("Actual vs Predicted Values")
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.tight_layout()
    plt.savefig("outputs/prediction_scatter.png")
    plt.close()

    # 3. Feature importance
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["regressor"]

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": estimator.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=True
    )

    plt.figure(figsize=(10, 8))
    plt.barh(
        importance["feature"],
        importance["importance"]
    )

    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()

    print("\nTop 10 Important Features:")
    print(
        importance.sort_values(
            "importance",
            ascending=False
        ).head(10)
    )

    print("\nAll visualisations saved successfully.")


