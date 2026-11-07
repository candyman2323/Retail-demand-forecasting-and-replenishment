
import pandas as pd

from .train import FEATURES


def forecast_next_7_days(df, model):

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    last_date = df["date"].max()

    combinations = df[
        ["store_id", "product_id"]
    ].drop_duplicates()

    forecasts = []

    for _, combination in combinations.iterrows():

        store_id = combination["store_id"]
        product_id = combination["product_id"]

        history = df[
            (df["store_id"] == store_id) &
            (df["product_id"] == product_id)
        ].sort_values("date").copy()

        latest = history.iloc[-1]

        sales = history["units_sold"].tolist()

        for day in range(1, 8):

            forecast_date = last_date + pd.Timedelta(days=day)

            day_of_week = forecast_date.day_of_week

            # Create the exact features used during training
            input_data = pd.DataFrame([{
                "store_id": store_id,
                "product_id": product_id,
                "category": latest["category"],
                "unit_price": latest["unit_price"],
                "promotion": latest["promotion"],
                "promotion_discount": latest["promotion_discount"],
                "day": forecast_date.day,
                "month": forecast_date.month,
                "day_of_week": day_of_week,
                "is_weekend": int(day_of_week >= 5),
                "lag_1": sales[-1],
                "lag_7": sales[-7],
                "lag_14": sales[-14],
                "lag_28": sales[-28],
                "rolling_mean_7": sum(sales[-7:]) / 7,
                "rolling_mean_28": sum(sales[-28:]) / 28
            }])

            # Maintain the training feature order
            input_data = input_data[FEATURES]

            prediction = model.predict(input_data)[0]

            prediction = max(0, prediction)

            forecasts.append({
                "date": forecast_date,
                "store_id": store_id,
                "product_id": product_id,
                "predicted_units_sold": round(prediction, 2)
            })

            # Use prediction for the next day's lag features
            sales.append(prediction)

    forecast_df = pd.DataFrame(forecasts)

    return forecast_df
