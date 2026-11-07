
import pandas as pd

from .baseline import evaluate_baseline
from .evaluation import evaluate_model


def compare_models(original_model, tuned_model, test_df):

    print("\nBASELINE MODEL")
    baseline_metrics = evaluate_baseline(test_df)

    print("\nORIGINAL RANDOM FOREST")
    original_metrics = evaluate_model(original_model, test_df)

    print("\nTUNED RANDOM FOREST")
    tuned_metrics = evaluate_model(tuned_model, test_df)

    comparison = pd.DataFrame([
        {"Model": "Baseline", **baseline_metrics},
        {"Model": "Original Random Forest", **original_metrics},
        {"Model": "Tuned Random Forest", **tuned_metrics}
    ])

    comparison = comparison.set_index("Model")

    print("\nMODEL COMPARISON")
    print(comparison.round(3))

    comparison.to_csv("outputs/model_comparison.csv")

    return comparison
