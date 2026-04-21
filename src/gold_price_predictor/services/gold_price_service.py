from datetime import datetime

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.preprocessing import PolynomialFeatures  # type: ignore


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_cols = {"Date", "Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, found: {df.columns.tolist()}"
        )

    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date", "Close"], inplace=True)

    df = df.sort_values("Date").reset_index(drop=True)
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days
    return df, df["Date"].min()


def train_model(df):
    x = df[["Days"]]
    y = df["Close"]
    train_size = int(0.8 * len(df))
    x_train = x[:train_size]
    y_train = y[:train_size]

    model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    model.fit(x_train, y_train)
    return model


def compute_metrics(model, df):
    x = df[["Days"]]
    y = df["Close"]
    pred = model.predict(x)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    return {
        "r2": f"{r2:.2f}",
        "mse": f"{mse:.2f}",
        "rmse": f"{np.sqrt(mse):.2f}",
    }


def build_context(csv_path):
    df, min_date = load_and_prepare_data(csv_path)
    model = train_model(df)
    metrics = compute_metrics(model, df)
    today_date = datetime.now().strftime("%Y-%m-%d")
    return df, min_date, model, metrics, today_date
