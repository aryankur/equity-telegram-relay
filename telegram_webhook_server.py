"""
TradingView -> Telegram Alert Relay
------------------------------------
Receives webhook alerts from TradingView (based on your Equity Tool
Pine Script's alertcondition() calls) and forwards them as Telegram
messages.
"""

import os
import logging
from flask import Flask, request, jsonify
import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tv-telegram-relay")

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
# Optional shared secret so random internet traffic can't spam your bot.
# Add ?token=YOUR_SECRET to the webhook URL you give TradingView.
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# Simple in-memory cooldown so the same signal on the same ticker
# doesn't spam you if TradingView retries or re-evaluates.
_last_sent = {}
COOLDOWN_SECONDS = 60


def send_telegram_message(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID env vars")
        return False
    try:
        resp = requests.post(
            TELEGRAM_API_URL,
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error("Failed to send Telegram message: %s", e)
        return False


def format_alert(data: dict) -> str:
    signal = str(data.get("signal", "ALERT")).upper()
    ticker = data.get("ticker", "?")
    price = data.get("price", "?")
    time = data.get("time", "")

    emoji = {"BUY": "\U0001F7E2", "SELL": "\U0001F534", "CLOSE": "\u26AA"}.get(signal, "\U0001F514")

    return (
        f"{emoji} <b>{signal}</b> signal\n"
        f"Ticker: <b>{ticker}</b>\n"
        f"Price: {price}\n"
        f"Time: {time}"
    )


@app.route("/webhook", methods=["POST"])
def webhook():
    if WEBHOOK_SECRET and request.args.get("token") != WEBHOOK_SECRET:
        return jsonify({"error": "unauthorized"}), 401

    # TradingView sends either raw text or JSON depending on how you
    # configured the alert message. Handle both.
    data = request.get_json(silent=True)
    if data is None:
        raw = request.get_data(as_text=True)
        data = {"signal": raw.strip() or "ALERT", "ticker": "", "price": "", "time": ""}

    key = f"{data.get('ticker')}|{data.get('signal')}"
    import time as _time
    now = _time.time()
    if key in _last_sent and now - _last_sent[key] < COOLDOWN_SECONDS:
        log.info("Skipping duplicate alert within cooldown: %s", key)
        return jsonify({"status": "skipped_duplicate"}), 200
    _last_sent[key] = now

    message = format_alert(data)
    ok = send_telegram_message(message)

    return jsonify({"status": "sent" if ok else "failed"}), (200 if ok else 500)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
