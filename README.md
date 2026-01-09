# 🏆 Gold Price Predictor Using Polynomial Regression

<div align="center">

![Gold Price Predictor](https://img.shields.io/badge/Gold-Price%20Predictor-FFD700?style=for-the-badge&logo=bitcoin&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![ML](https://img.shields.io/badge/Machine%20Learning-Polynomial%20Regression-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

**AI-powered gold price forecasting with real-time USD to INR conversion**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-integration) • [Model](#-model-details)

</div>

---

## 📋 Overview

Gold Price Predictor is an intelligent web application that leverages **Polynomial Regression (degree 3)** machine learning to forecast future gold prices. Built with Flask and scikit-learn, it provides accurate predictions with automatic USD to INR conversion using live exchange rates.

### 🎯 Key Highlights

- 🔮 **AI-Powered Predictions** - Polynomial regression model trained on historical data
- 💱 **Real-time Currency Conversion** - Live USD to INR rates via API
- 📊 **Interactive Dashboard** - Beautiful dark-themed UI with animations
- 📈 **Trend Visualization** - Historical price charts with matplotlib
- 📜 **Prediction History** - Track and compare all your forecasts
- ⚖️ **Year Comparison** - Analyze gold price changes across years
- 📱 **Responsive Design** - Works seamlessly on mobile and desktop

---

## ✨ Features

### 🔮 Price Prediction
- Select any future date to get predicted gold prices
- Automatic calculation breakdown showing:
  - Original USD price per ounce
  - After 6% import duty
  - After 3% GST
  - Final INR price per gram and per 10 grams

### 📊 Analytics Dashboard
- **Model Metrics Display**: R-squared, MSE, and RMSE values
- **Historical Trends**: Interactive charts showing price movements over time
- **Year-over-Year Comparison**: Compare average prices between any two years
- **Prediction History**: Session-based tracking of all predictions with timestamps

### 💰 Currency Integration
- Live USD to INR exchange rates from [open.er-api.com](https://open.er-api.com)
- 1-hour caching to optimize API calls
- Fallback rate (₹83.0) for API failures
- Automatic conversion per troy ounce to grams

### 🎨 User Interface
- Modern dark theme with gold accents
- Smooth animations and hover effects
- Mobile-responsive with slide-out navigation
- Real-time embedded gold price widget from dpgold.com

---

## 🚀 Demo

### Dashboard Preview
```
┌─────────────────────────────────────────────────┐
│  🏆 Gold Analytics AI Predictor                 │
├─────────────────────────────────────────────────┤
│  📊 Metrics                                     │
│  • R-Squared: 0.95  • MSE: 234.56  • RMSE: 15.3│
├─────────────────────────────────────────────────┤
│  🔮 Make a Prediction                           │
│  Select Date: [2026-06-15] [Predict Price]     │
│                                                 │
│  Predicted Gold Price: $2,450.00               │
│  ₹ Final Price: ₹6,892.45 per gram             │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/gold-price-predictor.git
cd gold-price-predictor
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset
Ensure `GOLD_prices_2010_to_today.csv` is in the root directory with columns:
- `Date` (YYYY-MM-DD format)
- `Close` (USD price)

### Step 5: Run the Application
```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

---

## 📦 Dependencies

```txt
flask==2.3.0
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.2.2
matplotlib==3.7.1
requests==2.31.0
werkzeug==2.3.0
```

Create a `requirements.txt` file with the above content.

---

## 📁 Project Structure

```
gold-price-predictor/
│
├── app.py                          # Main Flask application
├── GOLD_prices_2010_to_today.csv  # Historical gold price dataset
├── gold_price_model.pkl           # Trained ML model (auto-generated)
├── requirements.txt               # Python dependencies
│
├── templates/
│   └── index_dashboard.html       # Main dashboard template
│
├── static/                        # (Optional) Static assets
│
└── README.md                      # Project documentation
```

---

## 🎓 Model Details

### Algorithm: Polynomial Regression (Degree 3)

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(3), LinearRegression())
```

### Training Process
1. **Data Loading**: Historical gold prices from 2010 to present
2. **Feature Engineering**: Days since earliest date as predictor variable
3. **Train-Test Split**: 80% training, 20% validation
4. **Model Training**: Polynomial features transformed, then linear regression
5. **Model Persistence**: Saved as `gold_price_model.pkl` using pickle

### Prediction Formula
```
y = β₀ + β₁x + β₂x² + β₃x³
```
Where `x` = days since the earliest date in the dataset

### Model Evaluation
- **R-Squared (R²)**: Measures goodness of fit
- **MSE (Mean Squared Error)**: Average squared prediction error
- **RMSE (Root Mean Squared Error)**: Standard deviation of residuals

---

## 🌐 API Integration

### USD to INR Exchange Rate API

**Endpoint**: `https://open.er-api.com/v6/latest/USD`

**Features**:
- Free tier with no authentication required
- Hourly rate limiting
- 1-hour cache implementation
- Automatic fallback to ₹83.0

**Implementation**:
```python
def get_usd_to_inr():
    # Check cache (1-hour validity)
    if cache_is_valid():
        return cached_rate
    
    # Fetch live rate
    response = requests.get("https://open.er-api.com/v6/latest/USD")
    rate = response.json()["rates"]["INR"]
    
    # Update cache
    cache_rate(rate)
    return rate
```

---

## 💡 Usage Guide

### Making Predictions

1. **Navigate to Predict Page**
   - Click "🔮 Predict" in the sidebar

2. **Select Future Date**
   - Choose any date after today
   - Click "Predict Price"

3. **View Results**
   - USD price prediction
   - Detailed INR calculation breakdown
   - Updated price chart

### Viewing History

1. **Access History Page**
   - Click "📜 History" in sidebar

2. **Review Predictions**
   - See all predictions with timestamps
   - USD and INR prices displayed
   - Clear history option available

### Comparing Years

1. **Navigate to Compare**
   - Click "⚖ Compare" in sidebar

2. **Select Years**
   - Choose two years from dropdowns
   - Click "Compare Years"

3. **Analyze Results**
   - Average price per year
   - Price difference calculation

---

## 🔧 Configuration

### Adjusting Conversion Rates

Edit the calculation in `app.py`:

```python
def calculate_inr_price(usd_price):
    usd_to_inr = (usd_price * get_usd_to_inr()) / 31.103  # Troy ounce to grams
    usd_to_inr_with_6_percent = usd_to_inr * 1.06        # Import duty
    final_price_in_inr = usd_to_inr_with_6_percent * 1.03 # GST
    final_price_in_inr_for_10grams = final_price_in_inr * 10
    
    return {
        'inr_price': final_price_in_inr,
        'inr_price_for_10grams': final_price_in_inr_for_10grams
    }
```

### Changing Model Degree

Modify the polynomial degree in `train_model()`:

```python
model = make_pipeline(PolynomialFeatures(3), LinearRegression())  # Change 3 to desired degree
```

---

## 🚀 Deployment

### Deploy on Render

1. Create `render.yaml`:
```yaml
services:
  - type: web
    name: gold-predictor
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
```

2. Add `gunicorn` to requirements:
```bash
echo "gunicorn==20.1.0" >> requirements.txt
```

3. Push to GitHub and connect to Render

### Deploy on Heroku

1. Create `Procfile`:
```
web: gunicorn app:app
```

2. Deploy:
```bash
heroku create gold-price-predictor
git push heroku main
```

---

## ⚠️ Disclaimer

**This application is for educational and demonstration purposes only.**

- Predictions are based on historical patterns and statistical modeling
- NOT intended for financial advice or investment decisions
- Gold prices are influenced by numerous real-world factors not captured by this model
- Always consult financial professionals for investment guidance

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Ideas
- Add more ML models (LSTM, ARIMA, Prophet)
- Implement user authentication
- Add database persistence
- Include more financial metrics
- Multi-currency support

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@kalyanasrinivasbonagiri]([https://github.com/kalyanasrinivasbonagiri](https://github.com/kalyanasrinivasbonagiri-lang))
- LinkedIn: [kalyanasrinivasbonagiri](www.linkedin.com/in/kalyanasrinivas-bonagiri-a33709322)
- Email: kalyanasrinivasbonagiri@gmaail.com

---

## 🙏 Acknowledgments

- Historical gold price data from [source]
- Exchange rate API by [open.er-api.com](https://open.er-api.com)
- Scikit-learn for machine learning capabilities
- Flask framework for web application
- Live gold price widget by [dpgold.com](https://www.dpgold.com)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ and 🐍 Python

</div>
