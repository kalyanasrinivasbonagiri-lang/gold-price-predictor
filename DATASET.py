import yfinance as yf
import pandas as pd
from pathlib import Path


def download_gold_prices():
    ticker = "GC=F"
    cache_dir = Path(".yfinance_cache")
    cache_dir.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir.resolve()))

    df = yf.download(ticker, start="2010-01-01", interval="1d", auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[expected_columns].copy()

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df.dropna(subset=["Date", "Close"], inplace=True)
    df = df.sort_values("Date").reset_index(drop=True)

    df.to_csv("GOLD_prices_2010_to_today.csv", index=False)

    print("Gold prices updated successfully")
    print(f"Latest date: {df['Date'].iloc[-1].date()}")


if __name__ == "__main__":
    download_gold_prices()
