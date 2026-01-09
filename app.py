from flask import Flask, render_template, request, session, redirect, url_for, jsonify  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from datetime import datetime
import os
import tempfile
import requests
# Ensure matplotlib uses a non-GUI backend and does not try to read user config
# Set environment variable before importing matplotlib to avoid hangs on import
os.environ.setdefault('MPLBACKEND', 'Agg')
# Ensure matplotlib config directory is local and writable to avoid issues
mpl_cfg_dir = os.path.join(os.getcwd(), '.mpl_config')
os.makedirs(mpl_cfg_dir, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', mpl_cfg_dir)
import io
import base64
import warnings
from sklearn.preprocessing import PolynomialFeatures  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore
from sklearn.pipeline import make_pipeline  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
import pickle
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates')
app.secret_key = "supersecretkey123"

# Set Flask to trust proxy headers (needed for Render)
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)

# USD to INR conversion rate (you can update this as needed)
USD_TO_INR = 89.82


# ---------------- DATA AND MODEL ---------------- #
def load_and_prepare_data():
    df = pd.read_csv('GOLD_prices_2010_to_today.csv')

    # Strip column spaces (safety)
    df.columns = df.columns.str.strip()

    # Validate required columns
    required_cols = {'Date', 'Close'}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, found: {df.columns.tolist()}"
        )

    df = df[['Date', 'Close']].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date', 'Close'], inplace=True)

    df = df.sort_values('Date').reset_index(drop=True)
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days

    return df, df['Date'].min()

def train_model(df):
    X = df[['Days']]
    y = df['Close']
    train_size = int(0.8 * len(df))
    X_train = X[:train_size]
    y_train = y[:train_size]

    # Train the model
    model = make_pipeline(PolynomialFeatures(3), LinearRegression())
    model.fit(X_train, y_train)

    # Save the trained model to a pickle file
    with open("gold_price_model.pkl", "wb") as file:
        pickle.dump(model, file)
    print("Model trained and saved as gold_price_model.pkl")

    return model

USD_INR_CACHE = {
    "rate": None,
    "last_updated": None
}
from datetime import datetime
import requests

USD_INR_CACHE = {
    "rate": None,
    "last_updated": None
}

FALLBACK_USD_TO_INR = 83.0

def get_usd_to_inr():
    try:
        # Cache for 1 hour
        if USD_INR_CACHE["rate"] and USD_INR_CACHE["last_updated"]:
            if (datetime.now() - USD_INR_CACHE["last_updated"]).seconds < 3600:
                print("Using cached USD to INR rate.")
                return USD_INR_CACHE["rate"]

        url = "https://open.er-api.com/v6/latest/USD"
        response = requests.get(url, timeout=5)
        data = response.json()

        print("API Response:", data)

        rate = data["rates"]["INR"]

        USD_INR_CACHE["rate"] = rate
        USD_INR_CACHE["last_updated"] = datetime.now()

        print("Fetched USD to INR rate from open.er-api.com.")
        return rate

    except Exception as e:
        print("Using fallback USD to INR rate. Error:", e)
        return FALLBACK_USD_TO_INR


def calculate_inr_price(usd_price):
    usd_to_inr = (usd_price * get_usd_to_inr()) / 31.103
    usd_to_inr_with_6_percent = usd_to_inr * 1.06
    final_price_in_inr = usd_to_inr_with_6_percent * 1.03
    final_price_in_inr_for_10grams = final_price_in_inr * 10
    return {
        'original_usd': usd_price,
        'after_6_percent': usd_to_inr_with_6_percent,
        'after_3_percent': final_price_in_inr,
        'inr_price': final_price_in_inr,
        'inr_price_for_10grams': final_price_in_inr_for_10grams,
    }

def create_plot(df, prediction_value=None, date_to_predict=None):
    # Import matplotlib lazily to avoid blocking server startup (font cache building)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Set dark theme for matplotlib
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1a1f3a')
    ax.set_facecolor('#0a0e27')

    # Plot historical data with gold color
    ax.plot(df['Date'], df['Close'], label='Historical Prices',
            color='#ffd700', linewidth=2, alpha=0.8)

    # Plot prediction point if exists
    if prediction_value is not None and date_to_predict:
        future_date = pd.to_datetime(date_to_predict)
        ax.scatter([future_date], [prediction_value],
                   color='#ff7f0e', label='Prediction', s=150,
                   edgecolors='white', linewidth=2, zorder=5)

    ax.set_title('Gold Price History', fontsize=16, color='#ffd700', fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12, color='#8b92b0')
    ax.set_ylabel('Price (USD)', fontsize=12, color='#8b92b0')
    ax.grid(True, alpha=0.2, color='#4c9aff', linestyle='--')
    ax.tick_params(colors='#8b92b0')
    plt.xticks(rotation=45)

    # Style legend
    legend = ax.legend(facecolor='#1a1f3a', edgecolor='#ffd700',
                       labelcolor='#e8eaf0', framealpha=0.9)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200, facecolor='#1a1f3a')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    return img_base64


def compute_metrics(model, df):
    X = df[['Days']]
    y = df['Close']
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    return {'r2': f"{r2:.2f}", 'mse': f"{mse:.2f}", 'rmse': f"{np.sqrt(mse):.2f}"}


def build_context():
    df, min_date = load_and_prepare_data()
    model = train_model(df)
    metrics = compute_metrics(model, df)
    today_date = datetime.now().strftime('%Y-%m-%d')
    return df, min_date, model, metrics, today_date


# ---------------- ROUTES ---------------- #

@app.route('/', methods=['GET', 'POST'])
def index():
    df, min_date, model, metrics, today_date = build_context()
    prediction = None
    inr_calculation = None
    plot_image = None
    error = None

    # Initialize history in session if not exists
    if 'history' not in session:
        session['history'] = []

    if request.method == 'POST':
        date_str = request.form.get('date')
        try:
            future_date = pd.to_datetime(date_str)
            today = pd.to_datetime(today_date)

            if future_date <= today:
                error = "Please select a future date"
            else:
                future_day = (future_date - min_date).days
                # model.predict expects 2D array-like
                prediction = float(model.predict([[future_day]])[0])
                
                # Calculate INR price with the formula
                inr_calculation = calculate_inr_price(prediction)
                
                plot_image = create_plot(df, prediction, date_str)

                # Add to history with timestamp
                session['history'].append({
                    'date': date_str,
                    'prediction': round(prediction, 2),
                    'inr_price': round(inr_calculation['inr_price'], 2),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                session.modified = True

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('index_dashboard.html',
                           prediction=prediction,
                           inr_calculation=inr_calculation,
                           plot_image=plot_image,
                           error=error,
                           today_date=today_date,
                           model_metrics=metrics,
                           history=session.get('history', []),
                           section='prediction')


@app.route('/trends')
def trends():
    df, _, _, metrics, today_date = build_context()
    plot_image = create_plot(df)
    return render_template('index_dashboard.html',
                           plot_image=plot_image,
                           model_metrics=metrics,
                           today_date=today_date,
                           history=session.get('history', []),
                           section='trends')


@app.route('/compare', methods=['GET', 'POST'])
def compare():
    df, _, _, metrics, today_date = build_context()
    result = None
    years = sorted(df['Date'].dt.year.unique())
    year1, year2 = None, None

    if request.method == 'POST':
        try:
            year1 = int(request.form.get('year1'))
            year2 = int(request.form.get('year2'))
            df1 = df[df['Date'].dt.year == year1]
            df2 = df[df['Date'].dt.year == year2]
            avg1 = df1['Close'].mean() if not df1.empty else 0
            avg2 = df2['Close'].mean() if not df2.empty else 0
            diff = avg2 - avg1
            result = f"{year1} Avg: ${avg1:.2f}, {year2} Avg: ${avg2:.2f} → Difference: ${diff:.2f}"
        except Exception as e:
            result = f"Error: Invalid years selected"

    return render_template('index_dashboard.html',
                           model_metrics=metrics,
                           today_date=today_date,
                           history=session.get('history', []),
                           section='compare',
                           years=years,
                           result=result,
                           year1=year1,
                           year2=year2)


@app.route('/history')
def history():
    _, _, _, metrics, today_date = build_context()
    return render_template('index_dashboard.html',
                           model_metrics=metrics,
                           today_date=today_date,
                           history=session.get('history', []),
                           section='history')


@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['history'] = []
    session.modified = True
    return redirect(url_for('history'))


@app.route('/about')
def about():
    _, _, _, metrics, today_date = build_context()
    return render_template('index_dashboard.html',
                           model_metrics=metrics,
                           today_date=today_date,
                           history=session.get('history', []),
                           section='about')


@app.route('/usd_to_inr', methods=['GET'])
def usd_to_inr():
    try:
        rate = get_usd_to_inr()
        return jsonify({"USD_to_INR_rate": rate})
    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    # Disable the auto-reloader to avoid restarts caused by changes
    # in site-packages (which can trigger connection resets in browser)
    app.run(debug=True, use_reloader=False)