import yfinance as yf  # type: ignore
import pandas as pd  # type: ignore
import os
from datetime import timedelta
import time

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
    Includes proper error handling for yfinance issues.
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

    # Retry logic for rate limiting
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df_new = yf.download(
                TICKER,
                start=start_fetch_date.strftime("%Y-%m-%d"),
                progress=False
            )
            
            if df_new.empty:
                print("No new data available. Dataset already up-to-date.")
                return
            
            # Handle MultiIndex columns if present
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            
            # Reset index to make Date a column
            df_new.reset_index(inplace=True)
            
            # Clean column names and select required ones
            df_new.columns = df_new.columns.str.strip()
            df_new = df_new[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
            # Append safely
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['Date'], keep='last')
            df_combined = df_combined.sort_values('Date').reset_index(drop=True)
            
            # Save
            df_combined.to_csv(CSV_FILE, index=False)
            
            print(f"✅ Dataset updated successfully! Added {len(df_new)} new records")
            print(f"   Latest date: {df_combined['Date'].max().date()}")
            return
                
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = (attempt + 1) * 30
                print(f"Rate limited (HTTP 429). Waiting {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    print("Failed to fetch data after retries.")
                    return
                time.sleep(5)
    
    print("Update complete.")


if __name__ == "__main__":
    update_dataset()