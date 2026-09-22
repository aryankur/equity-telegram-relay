"""
TradingView -> Telegram Alert Relay (v3)
------------------------------------------
Builds on v2 (logging, retry-on-failure, multi-recipient). New in v3:

1. PORTFOLIO-LEVEL RISK CAP ACROSS SYMBOLS
   This server sees every alert from every chart/symbol you've wired up,
   which is the one place that actually CAN enforce a portfolio-wide rule
   (a single Pine script only ever sees its own chart). It tracks which
   symbols currently have an open BUY/SELL position (based on the alerts
   it has received) and will SUPPRESS a new BUY/SELL alert - not forward
   it to Telegram - if you're already at your configured position/risk
   cap. A CLOSE alert for a symbol frees up its slot.

   Configure via env vars:
     MAX_CONCURRENT_POSITIONS = 3   (simple cap: max N symbols open at once)
   Position state is kept in a local JSON file (positions_state.json) so
   it survives the server sleeping/waking, but NOT a fresh Render deploy
   (that wipes the disk). If you redeploy while positions are open,
   manually check/clear positions_state.json or send a CLOSE for each
   open symbol first.

2. ML CONFIRMING FILTER HOOK
   Pine Script cannot run a trained model - there's no ML runtime inside
   TradingView. This server CAN, by calling out to wherever you host a
   scoring model. This is a working INTEGRATION POINT, not a model - you
   still need to build/host the actual scorer (e.g. a small FastAPI app
   serving your XGBoost model's predict_proba). Point ML_FILTER_URL at
   it and this relay will POST the alert to it and only forward to
   Telegram if it approves. If ML_FILTER_URL is unset, this gate is
   skipped entirely and everything behaves like v2.

   Expected contract for your scoring endpoint:
     POST <ML_FILTER_URL>  body: {"ticker":..., "signal":..., "price":...}
     response: {"approve": true/false, "confidence": 0.0-1.0}  (confidence optional)
"""

import os
import json
import time as time_module
import logging
from flask import Flask, request, jsonify
import requests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tv-telegram-relay")

app = Flask(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
_raw_chat_ids = os.environ.get("TELEGRAM_CHAT_IDS") or os.environ.get("TELEGRAM_CHAT_ID", "")
TELEGRAM_CHAT_IDS = [c.strip() for c in _raw_chat_ids.split(",") if c.strip()]
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")
RETRY_DELAY_SECONDS = float(os.environ.get("RETRY_DELAY_SECONDS", "3"))
LOG_FILE = os.environ.get("ALERT_LOG_FILE", "alert_log.jsonl")

MAX_CONCURRENT_POSITIONS = int(os.environ.get("MAX_CONCURRENT_POSITIONS", "3"))
POSITIONS_FILE = os.environ.get("POSITIONS_FILE", "positions_state.json")

ML_FILTER_URL = os.environ.get("ML_FILTER_URL", "").strip()
ML_FILTER_TIMEOUT = float(os.environ.get("ML_FILTER_TIMEOUT_SECONDS", "5"))

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

_last_sent = {}
COOLDOWN_SECONDS = 60


# --------- Position state (portfolio risk cap) ---------

def load_positions() -> dict:
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_positions(positions: dict) -> None:
    try:
        with open(POSITIONS_FILE, "w") as f:
            json.dump(positions, f)
    except OSError as e:
        log.warning("Could not persist positions state: %s", e)


# --------- Logging ---------

def log_alert(record: dict) -> None:
    try:
        record["logged_at"] = time_module.time()
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        log.warning("Could not write to alert log: %s", e)


# --------- Telegram delivery ---------

def send_telegram_message(chat_id: str, text: str, attempt: int = 1) -> bool:
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        log.error("Missing TELEGRAM_BOT_TOKEN or chat_id")
        return False
    try:
        resp = requests.post(
            TELEGRAM_API_URL,
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error("Telegram send failed (attempt %d) for chat %s: %s", attempt, chat_id, e)
        if attempt == 1:
            time_module.sleep(RETRY_DELAY_SECONDS)
            return send_telegram_message(chat_id, text, attempt=2)
        return False


def format_alert(data: dict, note: str = "") -> str:
    signal = str(data.get("signal", "ALERT")).upper()
    ticker = data.get("ticker", "?")
    price = data.get("price", "?")
    qty = data.get("qty", "")
    time_str = data.get("time", "")

    emoji = {"BUY": "\U0001F7E2", "SELL": "\U0001F534", "CLOSE": "\u26AA"}.get(signal, "\U0001F514")

    lines = [
        f"{emoji} <b>{signal}</b> signal",
        f"Ticker: <b>{ticker}</b>",
        f"Price: {price}",
    ]
    if qty:
        lines.append(f"Suggested qty: {qty}")
    lines.append(f"Time: {time_str}")
    if note:
        lines.append(f"\n<i>{note}</i>")
    return "\n".join(lines)


# --------- ML confirming filter (integration point) ---------

def check_ml_filter(data: dict) -> tuple[bool, str]:
    """Returns (approved, note). If ML_FILTER_URL isn't configured, always
    approves and skips the check entirely."""
    if not ML_FILTER_URL:
        return True, ""
    try:
        resp = requests.post(ML_FILTER_URL, json=data, timeout=ML_FILTER_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()
        approve = bool(result.get("approve", False))
        confidence = result.get("confidence")
        note = f"ML filter: {'approved' if approve else 'rejected'}"
        if confidence is not None:
            note += f" (confidence {confidence})"
        return approve, note
    except (requests.RequestException, ValueError) as e:
        log.error("ML filter check failed, failing OPEN (not blocking trade) - error: %s", e)
        # Fails open: if your scorer is down, we don't want to silently
        # block every trade. Change to `return False, ...` if you'd
        # rather fail closed (block trades when the scorer is unreachable).
        return True, "ML filter unreachable - passed through"


# --------- Webhook ---------

@app.route("/webhook", methods=["POST"])
def webhook():
    if WEBHOOK_SECRET and request.args.get("token") != WEBHOOK_SECRET:
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True)
    if data is None:
        raw = request.get_data(as_text=True)
        data = {"signal": raw.strip() or "ALERT", "ticker": "", "price": "", "time": ""}

    signal = str(data.get("signal", "")).upper()
    ticker = data.get("ticker", "unknown")

    key = f"{ticker}|{signal}"
    now = time_module.time()
    if key in _last_sent and now - _last_sent[key] < COOLDOWN_SECONDS:
        log.info("Skipping duplicate alert within cooldown: %s", key)
        log_alert({"event": "skipped_duplicate", "data": data})
        return jsonify({"status": "skipped_duplicate"}), 200
    _last_sent[key] = now

    positions = load_positions()

    # --------- Portfolio risk cap check (BUY/SELL only) ---------
    if signal in ("BUY", "SELL"):
        already_open = ticker in positions
        if not already_open and len(positions) >= MAX_CONCURRENT_POSITIONS:
            note = f"Suppressed: portfolio cap reached ({len(positions)}/{MAX_CONCURRENT_POSITIONS} symbols open: {', '.join(positions.keys())})"
            log.info(note)
            log_alert({"event": "suppressed_portfolio_cap", "data": data, "open_positions": list(positions.keys())})
            for chat_id in TELEGRAM_CHAT_IDS:
                send_telegram_message(chat_id, format_alert(data, note=note))
            return jsonify({"status": "suppressed_portfolio_cap", "open_positions": list(positions.keys())}), 200

        # --------- ML confirming filter check ---------
        approved, ml_note = check_ml_filter(data)
        if not approved:
            log.info("Suppressed by ML filter: %s", data)
            log_alert({"event": "suppressed_ml_filter", "data": data})
            for chat_id in TELEGRAM_CHAT_IDS:
                send_telegram_message(chat_id, format_alert(data, note=f"Suppressed: {ml_note}"))
            return jsonify({"status": "suppressed_ml_filter"}), 200

        positions[ticker] = {"signal": signal, "opened_at": now}
        save_positions(positions)

    elif signal == "CLOSE":
        if ticker in positions:
            del positions[ticker]
            save_positions(positions)
        ml_note = ""

    else:
        ml_note = ""

    if not TELEGRAM_CHAT_IDS:
        log.error("No recipients configured")
        log_alert({"event": "failed_no_recipients", "data": data})
        return jsonify({"status": "failed", "reason": "no_recipients"}), 500

    message = format_alert(data, note=ml_note if signal in ("BUY", "SELL") else "")
    results = {chat_id: send_telegram_message(chat_id, message) for chat_id in TELEGRAM_CHAT_IDS}
    all_ok = all(results.values())

    log_alert({"event": "sent" if all_ok else "partial_or_failed", "data": data, "results": results})
    return jsonify({"status": "sent" if all_ok else "partial_failure", "results": results}), (200 if all_ok else 500)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/positions", methods=["GET"])
def positions_view():
    """See what the portfolio cap currently thinks is open."""
    return jsonify({"open_positions": load_positions(), "max_concurrent": MAX_CONCURRENT_POSITIONS}), 200


@app.route("/positions/clear", methods=["POST"])
def positions_clear():
    """Manually reset tracked positions - use if state gets out of sync
    with reality (e.g. after a redeploy, or a missed CLOSE alert)."""
    if WEBHOOK_SECRET and request.args.get("token") != WEBHOOK_SECRET:
        return jsonify({"error": "unauthorized"}), 401
    save_positions({})
    return jsonify({"status": "cleared"}), 200


@app.route("/recent-alerts", methods=["GET"])
def recent_alerts():
    n = int(request.args.get("n", 20))
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()[-n:]
        records = [json.loads(line) for line in lines]
        return jsonify({"count": len(records), "alerts": records}), 200
    except FileNotFoundError:
        return jsonify({"count": 0, "alerts": []}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
