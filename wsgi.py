from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gold_price_predictor.app_factory import create_app


app = create_app()
