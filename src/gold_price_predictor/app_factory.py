import os
import warnings

import pandas as pd  # type: ignore
from flask import Flask, jsonify, redirect, render_template, request, session, url_for  # type: ignore
from werkzeug.middleware.proxy_fix import ProxyFix

from gold_price_predictor.config import Config
from gold_price_predictor.services.currency_service import calculate_inr_price, get_usd_to_inr
from gold_price_predictor.services.gold_price_service import build_context
from gold_price_predictor.utils.plotting import create_plot


warnings.filterwarnings("ignore")


def create_app():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.makedirs(Config.MPL_CONFIG_DIR, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Config.MPL_CONFIG_DIR))

    app = Flask(
        __name__,
        template_folder=str(Config.TEMPLATE_DIR),
        static_folder=str(Config.STATIC_DIR),
    )
    app.config.from_object(Config)
    app.wsgi_app = ProxyFix(app.wsgi_app)

    register_routes(app)
    return app


def register_routes(app):
    @app.route("/", methods=["GET", "POST"])
    def index():
        df, min_date, model, metrics, today_date = build_context(app.config["DATA_FILE"])
        prediction = None
        inr_calculation = None
        plot_image = None
        error = None

        if "history" not in session:
            session["history"] = []

        if request.method == "POST":
            date_str = request.form.get("date")
            try:
                future_date = pd.to_datetime(date_str)
                today = pd.to_datetime(today_date)

                if future_date <= today:
                    error = "Please select a future date"
                else:
                    future_day = (future_date - min_date).days
                    prediction = float(model.predict([[future_day]])[0])
                    inr_calculation = calculate_inr_price(
                        prediction,
                        app.config["FALLBACK_USD_TO_INR"],
                    )
                    plot_image = create_plot(df, prediction, date_str)

                    session["history"].append(
                        {
                            "date": date_str,
                            "prediction": round(prediction, 2),
                            "inr_price": round(inr_calculation["inr_price"], 2),
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                    session.modified = True
            except Exception as exc:
                error = f"Error: {exc}"

        return render_template(
            "index_dashboard.html",
            prediction=prediction,
            inr_calculation=inr_calculation,
            plot_image=plot_image,
            error=error,
            today_date=today_date,
            model_metrics=metrics,
            history=session.get("history", []),
            section="prediction",
        )

    @app.route("/trends")
    def trends():
        df, _, _, metrics, today_date = build_context(app.config["DATA_FILE"])
        plot_image = create_plot(df)
        return render_template(
            "index_dashboard.html",
            plot_image=plot_image,
            model_metrics=metrics,
            today_date=today_date,
            history=session.get("history", []),
            section="trends",
        )

    @app.route("/compare", methods=["GET", "POST"])
    def compare():
        df, _, _, metrics, today_date = build_context(app.config["DATA_FILE"])
        result = None
        years = sorted(df["Date"].dt.year.unique())
        year1, year2 = None, None

        if request.method == "POST":
            try:
                year1 = int(request.form.get("year1"))
                year2 = int(request.form.get("year2"))
                df1 = df[df["Date"].dt.year == year1]
                df2 = df[df["Date"].dt.year == year2]
                avg1 = df1["Close"].mean() if not df1.empty else 0
                avg2 = df2["Close"].mean() if not df2.empty else 0
                diff = avg2 - avg1
                result = (
                    f"{year1} Avg: ${avg1:.2f}, {year2} Avg: ${avg2:.2f} "
                    f"-> Difference: ${diff:.2f}"
                )
            except Exception:
                result = "Error: Invalid years selected"

        return render_template(
            "index_dashboard.html",
            model_metrics=metrics,
            today_date=today_date,
            history=session.get("history", []),
            section="compare",
            years=years,
            result=result,
            year1=year1,
            year2=year2,
        )

    @app.route("/history")
    def history():
        _, _, _, metrics, today_date = build_context(app.config["DATA_FILE"])
        return render_template(
            "index_dashboard.html",
            model_metrics=metrics,
            today_date=today_date,
            history=session.get("history", []),
            section="history",
        )

    @app.route("/clear_history", methods=["POST"])
    def clear_history():
        session["history"] = []
        session.modified = True
        return redirect(url_for("history"))

    @app.route("/about")
    def about():
        _, _, _, metrics, today_date = build_context(app.config["DATA_FILE"])
        return render_template(
            "index_dashboard.html",
            model_metrics=metrics,
            today_date=today_date,
            history=session.get("history", []),
            section="about",
        )

    @app.route("/usd_to_inr", methods=["GET"])
    def usd_to_inr():
        try:
            rate = get_usd_to_inr(app.config["FALLBACK_USD_TO_INR"])
            return jsonify({"USD_to_INR_rate": rate})
        except Exception as exc:
            return jsonify({"error": str(exc)})
