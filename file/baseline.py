
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from .train import TARGET


def evaluate_baseline(test_df):

    # Use sales from 7 days ago as prediction
    actual = test_df[TARGET]
    predicted = np.maximum(0, test_df["lag_7"])

    # Calculate evaluation metrics
    metrics = {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "WAPE_percent": (
            np.abs(actual - predicted).sum()
            / np.abs(actual).sum() * 100
        )
    }

    print("\nBaseline Model Results")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

    return metrics
