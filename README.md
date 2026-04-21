# Gold Price Predictor

A Flask dashboard that predicts future gold prices from historical data, converts the estimate from USD to INR, and visualizes the result with a chart.

## Features

- Predict future gold prices using polynomial regression
- Convert predicted USD prices to INR with a live exchange-rate API and fallback rate
- View model metrics such as R-squared, MSE, and RMSE
- Compare yearly average gold prices
- Track prediction history in the browser session
- Refresh the dataset from Yahoo Finance

## Tech Stack

- Python 3.11+
- Flask
- pandas
- scikit-learn
- matplotlib
- requests
- yfinance

## Project Structure

```text
gold-price-predictor/
|-- app.py
|-- DATASET.py
|-- dataset.py
|-- GOLD_prices_2010_to_today.csv
|-- pyproject.toml
|-- uv.lock
|-- wsgi.py
|-- scripts/
|   `-- update_gold_dataset.py
|-- src/
|   `-- gold_price_predictor/
|       |-- __init__.py
|       |-- app_factory.py
|       |-- config.py
|       |-- services/
|       |   |-- currency_service.py
|       |   `-- gold_price_service.py
|       `-- utils/
|           `-- plotting.py
|-- templates/
|   `-- index_dashboard.html
`-- static/
    `-- .gitkeep
```

## How It Works

1. The app loads `GOLD_prices_2010_to_today.csv`.
2. It prepares a `Days` feature from the historical date range.
3. It trains a degree-3 polynomial regression model.
4. A future date entered by the user is converted into a day offset.
5. The predicted USD price is converted to INR per gram and per 10 grams.
6. The dashboard renders the prediction, metrics, and chart.

## Installation

### Option 1: using `uv`

```bash
uv sync
```

### Option 2: using `pip`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Run the App

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Update the Dataset

Use either of these:

```bash
python dataset.py
```

or

```bash
python scripts/update_gold_dataset.py
```

## Environment Variables

You can create a local `.env` file based on `.env.example`.

Supported settings:

- `SECRET_KEY`
- `FALLBACK_USD_TO_INR`

## Routes

- `/` : prediction dashboard
- `/trends` : historical chart view
- `/compare` : compare average prices between two years
- `/history` : session prediction history
- `/clear_history` : clear saved session history
- `/about` : project and model info
- `/usd_to_inr` : JSON endpoint for the current USD to INR rate

## Notes

- The current model is intentionally simple and meant for learning and demo purposes.
- Predictions are based only on historical price patterns.
- This project should not be used for real investment or financial decisions.

## Deployment

The repo includes [wsgi.py](./wsgi.py) for WSGI-based deployment.

A typical production entry point is:

```bash
gunicorn wsgi:app
```

## License

This project is available under the MIT License.
