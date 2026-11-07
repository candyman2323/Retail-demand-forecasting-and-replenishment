import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()

    # Sort by date
    df = df.sort_values(by="date")

    # Reset index
    df = df.reset_index(drop=True)

    return df