def create_features(df):
    df = df.copy()
    df = df.sort_values(["store_id", "product_id", "date"])

    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    group = df.groupby(["store_id", "product_id"])["units_sold"]
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = group.shift(lag)

    df["rolling_mean_7"] = group.transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    df["rolling_mean_28"] = group.transform(
        lambda x: x.shift(1).rolling(28).mean()
    )

    df = df.dropna().reset_index(drop=True)
    return df
