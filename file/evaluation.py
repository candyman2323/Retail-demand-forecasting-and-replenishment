import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, test_df):
    from .train import FEATURES, TARGET

    actual = test_df[TARGET]
    predicted = np.maximum(0, model.predict(test_df[FEATURES]))

    results = {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
        "WAPE_percent": (
            np.abs(actual - predicted).sum() / np.abs(actual).sum() * 100
        )
    }
    for metric, value in results.items():
        print(f"{metric}: {value:.3f}")
    return results
