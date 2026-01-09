import yfinance as yf  # type: ignore
import pandas as pd  # type: ignore
import os
from datetime import timedelta

# ================= CONFIG =================
TICKER = "GC=F"
CSV_FILE = "GOLD_prices_2010_to_today.csv"
START_DATE = "2010-01-01"
REFRESH_DAYS = 1   # check daily
# =========================================

print("Gold Price Dataset Manager (SAFE APPEND MODE)")

def clean_and_lock_schema(csv_file: str) -> pd.DataFrame:
    """
    Ensures schema:
    Date, Open, High, Low, Close, Volume
    WITHOUT deleting historical data
    """
    print("Cleaning & locking CSV schema...")

    df = pd.read_csv(csv_file)

    # Fix Date column
    if df.columns[0] != "Date":
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)

    # Normalize column names (Yahoo quirks)
    rename_map = {}
    for col in df.columns:
        c = str(col)
        if "Open" in c:
            rename_map[col] = "Open"
        elif "High" in c:
            rename_map[col] = "High"
        elif "Low" in c:
            rename_map[col] = "Low"
        elif "Close" in c:
            rename_map[col] = "Close"
        elif "Volume" in c:
            rename_map[col] = "Volume"

    df.rename(columns=rename_map, inplace=True)

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    df = df[required_cols]

    # Remove duplicates safely
    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)

    df.sort_values("Date", inplace=True)
    df.to_csv(csv_file, index=False)

    print("Schema verified & locked.")
    return df


def update_dataset():
    """
    Appends ONLY new data to existing CSV.
    Never deletes old data.
    """
    if os.path.exists(CSV_FILE):
        print("Existing dataset found.")
        df_existing = pd.read_csv(CSV_FILE, parse_dates=["Date"])

        last_date = df_existing["Date"].max()
        print(f"Last available date: {last_date.date()}")

        start_fetch_date = last_date + timedelta(days=1)
    else:
        print("No dataset found. Creating from scratch...")
        start_fetch_date = pd.to_datetime(START_DATE)
        df_existing = pd.DataFrame()

    print(f"Fetching data from {start_fetch_date.date()} to today...")

    df_new = yf.download(
        TICKER,
        start=start_fetch_date.strftime("%Y-%m-%d"),
        progress=False
    )

    if df_new.empty:
        print("No new data available. Dataset already up-to-date.")
        return

    df_new.reset_index(inplace=True)

    # Append safely
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Final clean & save
    df_combined.to_csv(CSV_FILE, index=False)
    clean_and_lock_schema(CSV_FILE)

    print("Dataset updated successfully (data appended safely).")


if __name__ == "__main__":
    update_dataset()