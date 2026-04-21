import base64
import io

import pandas as pd  # type: ignore


def create_plot(df, prediction_value=None, date_to_predict=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1f3a")
    ax.set_facecolor("#0a0e27")

    ax.plot(
        df["Date"],
        df["Close"],
        label="Historical Prices",
        color="#ffd700",
        linewidth=2,
        alpha=0.8,
    )

    if prediction_value is not None and date_to_predict:
        future_date = pd.to_datetime(date_to_predict)
        ax.scatter(
            [future_date],
            [prediction_value],
            color="#ff7f0e",
            label="Prediction",
            s=150,
            edgecolors="white",
            linewidth=2,
            zorder=5,
        )

    ax.set_title(
        "Gold Price History",
        fontsize=16,
        color="#ffd700",
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Date", fontsize=12, color="#8b92b0")
    ax.set_ylabel("Price (USD)", fontsize=12, color="#8b92b0")
    ax.grid(True, alpha=0.2, color="#4c9aff", linestyle="--")
    ax.tick_params(colors="#8b92b0")
    plt.xticks(rotation=45)

    ax.legend(
        facecolor="#1a1f3a",
        edgecolor="#ffd700",
        labelcolor="#e8eaf0",
        framealpha=0.9,
    )

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=200, facecolor="#1a1f3a")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return img_base64
