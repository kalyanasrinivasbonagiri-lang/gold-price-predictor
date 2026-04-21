from datetime import datetime

import requests


USD_INR_CACHE = {"rate": None, "last_updated": None}


def get_usd_to_inr(fallback_rate):
    try:
        if USD_INR_CACHE["rate"] and USD_INR_CACHE["last_updated"]:
            if (datetime.now() - USD_INR_CACHE["last_updated"]).seconds < 3600:
                return USD_INR_CACHE["rate"]

        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=5)
        data = response.json()
        rate = data["rates"]["INR"]

        USD_INR_CACHE["rate"] = rate
        USD_INR_CACHE["last_updated"] = datetime.now()
        return rate
    except Exception:
        return fallback_rate


def calculate_inr_price(usd_price, fallback_rate):
    usd_to_inr = (usd_price * get_usd_to_inr(fallback_rate)) / 31.103
    usd_to_inr_with_6_percent = usd_to_inr * 1.06
    final_price_in_inr = usd_to_inr_with_6_percent * 1.03
    final_price_in_inr_for_10grams = final_price_in_inr * 10
    return {
        "original_usd": usd_price,
        "after_6_percent": usd_to_inr_with_6_percent,
        "after_3_percent": final_price_in_inr,
        "inr_price": final_price_in_inr,
        "inr_price_for_10grams": final_price_in_inr_for_10grams,
    }
