import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "units_sold"
FEATURES = [
    "store_id", "product_id", "category", "unit_price",
    "promotion", "promotion_discount", "day", "month",
    "day_of_week", "is_weekend", "lag_1", "lag_7",
    "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28"
]
CATEGORICAL = ["store_id", "product_id", "category"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

def train_model(df, model_path="../models/retail_model.joblib"):
    dates = sorted(df["date"].unique())
    split_date = dates[int(len(dates) * 0.8)]
    train = df[df["date"] < split_date]
    test = df[df["date"] >= split_date]

    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), CATEGORICAL),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), NUMERIC)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=150, random_state=42, n_jobs=-1
        ))
    ])

    model.fit(train[FEATURES], train[TARGET])

    import os
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(model, model_path)
    return model, test
