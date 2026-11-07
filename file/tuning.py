
import os
import joblib
import numpy as np

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from .train import FEATURES, TARGET, CATEGORICAL, NUMERIC


def tune_model(df, model_path="outputs/tuned_retail_model.joblib"):

    df = df.sort_values("date").reset_index(drop=True)

    dates = np.array(sorted(df["date"].unique()))
    split_date = dates[int(len(dates) * 0.8)]

    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()

    unique_train_dates = np.array(sorted(train["date"].unique()))

    time_split = TimeSeriesSplit(n_splits=3)

    cv_splits = []

    for train_date_idx, val_date_idx in time_split.split(unique_train_dates):

        fold_train_dates = unique_train_dates[train_date_idx]
        fold_val_dates = unique_train_dates[val_date_idx]

        train_idx = np.flatnonzero(
            train["date"].isin(fold_train_dates).to_numpy()
        )

        val_idx = np.flatnonzero(
            train["date"].isin(fold_val_dates).to_numpy()
        )

        cv_splits.append((train_idx, val_idx))

    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL),

        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), NUMERIC)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            random_state=42,
            n_jobs=-1
        ))
    ])

    parameters = {
        "regressor__n_estimators": [100, 150, 200],
        "regressor__max_depth": [None, 10, 20, 30],
        "regressor__min_samples_split": [2, 5, 10],
        "regressor__min_samples_leaf": [1, 2, 4],
        "regressor__max_features": ["sqrt", 0.7, 1.0]
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=10,
        cv=cv_splits,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    print("Starting hyperparameter tuning...")

    search.fit(train[FEATURES], train[TARGET])

    best_model = search.best_estimator_

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(best_model, model_path)

    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest Cross-Validation MAE:")
    print(round(-search.best_score_, 3))

    print("\nTuned model saved:", model_path)

    return best_model, test, search.best_params_
