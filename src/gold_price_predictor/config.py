from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey123")
    DATA_FILE = BASE_DIR / "GOLD_prices_2010_to_today.csv"
    TEMPLATE_DIR = BASE_DIR / "templates"
    STATIC_DIR = BASE_DIR / "static"
    MPL_CONFIG_DIR = BASE_DIR / ".mpl_config"
    FALLBACK_USD_TO_INR = float(os.getenv("FALLBACK_USD_TO_INR", "90.0"))
