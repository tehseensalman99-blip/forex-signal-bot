"""
Forex / Gold Signal Bot  (GitHub Actions + Telegram)
----------------------------------------------------
Strategy (testing ke liye):
  - Timeframe: 15 minute
  - LONG  : EMA 9 neeche se EMA 21 ko cross kare UPAR  + price EMA 200 se upar ho
  - SHORT : EMA 9 upar se EMA 21 ko cross kare NEECHE  + price EMA 200 se neeche ho
  - SL / TP : entry price ka percentage (har symbol ka alag neeche set hai)

Data: Twelve Data (free API key)
Alert: ntfy app (push notification) / Gmail / Telegram — jo secrets daaloge wohi chalenge
"""

import datetime as dt
import json
import os
import smtplib
import sys
import urllib.parse
import urllib.request
from email.mime.text import MIMEText

# ===================== SETTINGS (yahan change karo) =====================
SYMBOLS = {
    #  symbol      SL %          TP %          price decimals
    "XAU/USD": {"sl_pct": 0.40, "tp_pct": 0.80, "digits": 2},
    "EUR/USD": {"sl_pct": 0.20, "tp_pct": 0.40, "digits": 5},
}
INTERVAL = "15min"
CANDLE_MINUTES = 15
EMA_FAST = 9
EMA_SLOW = 21
EMA_TREND = 200
# ========================================================================

TD_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.environ.get("TELEGRAM_CHAT_ID", "")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "").strip()
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "").strip()
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "").replace(" ", "")
EMAIL_TO = os.environ.get("EMAIL_TO", "").strip() or GMAIL_ADDRESS
TEST_MODE = os.environ.get("TEST_MODE", "false").strip().lower() == "true"

STATE_FILE = "state.json"
PKT = dt.timezone(dt.timedelta(hours=5))  # Pakistan time


# ----------------------------- helpers ----------------------------------
def http_get_json(url, params):
    full = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(full, timeout=30) as r:
        return json.loads(r.read().decode())


def send_telegram(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": TG_CHAT, "text": text}).encode()
    with urllib.request.urlopen(url, data=data, timeout=30) as r:
        resp = json.loads(r.read().decode())
    if not resp.get("ok"):
        raise RuntimeError(f"Telegram error: {resp}")


def send_ntfy(title, text, urgent=True):
    payload = {
        "topic": NTFY_TOPIC,
        "title": title,
        "message": text,
        "priority": 5 if urgent else 3,  # 5 = urgent (zor se bajta hai)
    }
    req = urllib.request.Request(
        "https://ntfy.sh/",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def send_email(title, text):
    msg = MIMEText(text, "plain", "utf-8")
    msg["Subject"] = title
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = EMAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
        s.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        s.send_message(msg)


def channels():
    ch = []
    if NTFY_TOPIC:
        ch.append("ntfy")
    if GMAIL_ADDRESS and GMAIL_APP_PASSWORD:
        ch.append("email")
    if TG_TOKEN and TG_CHAT:
        ch.append("telegram")
    return ch


def notify(text, urgent=True):
    """Message ko har set kiye gaye channel par bhejta hai.
    Kam az kam ek channel kaamyab ho to OK, warna error."""
    title = text.split("\n", 1)[0]
    ok, errs = 0, []
    for ch in channels():
        try:
            if ch == "ntfy":
                send_ntfy(title, text, urgent)
            elif ch == "email":
                send_email(title, text)
            elif ch == "telegram":
                send_telegram(text)
            ok += 1
        except Exception as e:
            errs.append(f"{ch}: {e}")
            print("Notification error ->", ch, e)
    if ok == 0:
        raise RuntimeError("Koi notification nahi gaya: " + "; ".join(errs))


def ema(values, n):
    out = [None] * len(values)
    if len(values) < n:
        return out
    k = 2 / (n + 1)
    s = sum(values[:n]) / n
    out[n - 1] = s
    for i in range(n, len(values)):
        s = values[i] * k + s * (1 - k)
        out[i] = s
    return out


def fetch_closed_candles(symbol):
    """Sirf band (closed) candles return karta hai, purani se nayi tak."""
    d = http_get_json(
        "https://api.twelvedata.com/time_series",
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "outputsize": 500,
            "timezone": "UTC",
            "apikey": TD_KEY,
        },
    )
    if d.get("status") != "ok":
        raise RuntimeError(f"{symbol} data error: {d.get('message', d)}")

    now = dt.datetime.now(dt.timezone.utc)
    candles = []
    for v in reversed(d["values"]):  # API nayi se purani deta hai
        t = dt.datetime.strptime(v["datetime"], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=dt.timezone.utc
        )
        if t + dt.timedelta(minutes=CANDLE_MINUTES) <= now:
            candles.append((t, float(v["close"])))
    return candles


def find_signals(candles, check_last=2):
    """Aakhri `check_last` closed candles par crossover dhoondta hai."""
    closes = [c for _, c in candles]
    fast, slow, trend = ema(closes, EMA_FAST), ema(closes, EMA_SLOW), ema(closes, EMA_TREND)
    signals = []
    start = max(EMA_TREND, len(candles) - check_last)
    for i in range(start, len(candles)):
        if None in (fast[i - 1], slow[i - 1], trend[i]):
            continue
        crossed_up = fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]
        crossed_dn = fast[i - 1] >= slow[i - 1] and fast[i] < slow[i]
        if crossed_up and closes[i] > trend[i]:
            signals.append(("LONG", candles[i][0], closes[i]))
        elif crossed_dn and closes[i] < trend[i]:
            signals.append(("SHORT", candles[i][0], closes[i]))
    return signals, fast, slow, trend


def build_message(symbol, side, t, entry, cfg):
    d = cfg["digits"]
    if side == "LONG":
        sl = entry * (1 - cfg["sl_pct"] / 100)
        tp = entry * (1 + cfg["tp_pct"] / 100)
        head = f"🟢 LONG — {symbol} (15m)"
    else:
        sl = entry * (1 + cfg["sl_pct"] / 100)
        tp = entry * (1 - cfg["tp_pct"] / 100)
        head = f"🔴 SHORT — {symbol} (15m)"
    when = t.astimezone(PKT).strftime("%d %b %I:%M %p PKT")
    return (
        f"{head}\n"
        f"Entry: {entry:.{d}f}\n"
        f"SL: {sl:.{d}f}  ({cfg['sl_pct']}%)\n"
        f"TP: {tp:.{d}f}  ({cfg['tp_pct']}%)\n"
        f"Candle: {when}\n"
        f"⚠️ Testing strategy hai, khud confirm kar ke trade karein."
    )


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


# ------------------------------- main ------------------------------------
def main():
    if not TD_KEY:
        print("TWELVEDATA_API_KEY secret missing hai")
        sys.exit(1)
    if not channels():
        print("Koi notification channel set nahi. NTFY_TOPIC ya "
              "GMAIL_ADDRESS + GMAIL_APP_PASSWORD secret daalo.")
        sys.exit(1)
    print("Notification channels:", ", ".join(channels()))

    state = load_state()
    errors = []
    status_lines = []

    for symbol, cfg in SYMBOLS.items():
        try:
            candles = fetch_closed_candles(symbol)
            if len(candles) <= EMA_TREND + 1:
                raise RuntimeError(f"{symbol}: candles kam hain ({len(candles)})")

            signals, fast, slow, trend = find_signals(candles)
            last_sent = state.get(symbol)

            for side, t, entry in signals:
                key = t.isoformat()
                if last_sent and key <= last_sent:
                    continue  # ye signal pehle bhej chuke
                notify(build_message(symbol, side, t, entry, cfg))
                print(f"Sent {side} {symbol} @ {entry} ({key})")
                state[symbol] = key
                last_sent = key

            t_last, c_last = candles[-1]
            direction = "upar ⬆️" if fast[-1] > slow[-1] else "neeche ⬇️"
            trend_txt = "upar" if c_last > trend[-1] else "neeche"
            status_lines.append(
                f"{symbol}: {c_last:.{cfg['digits']}f} | EMA9 EMA21 se {direction} | "
                f"price EMA200 se {trend_txt}"
            )
            if not signals:
                print(f"{symbol}: koi naya signal nahi")
        except Exception as e:  # ek symbol fail ho to baaki chalte rahein
            print("ERROR:", e)
            errors.append(str(e))

    save_state(state)

    if TEST_MODE:
        msg = "✅ Bot chal raha hai!\n\n" + "\n".join(status_lines or ["(data nahi mila)"])
        if errors:
            msg += "\n\n❌ Errors:\n" + "\n".join(errors)
        notify(msg, urgent=False)

    if errors:
        sys.exit(1)  # GitHub aapko failure ki email bhej dega


if __name__ == "__main__":
    main()
