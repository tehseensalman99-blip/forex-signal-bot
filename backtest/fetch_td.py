"""Twelve Data se 15m candles download karta hai (2024-01-01 se aaj tak)."""
import csv
import datetime as dt
import json
import os
import time
import urllib.parse
import urllib.request

KEY = os.environ["TWELVEDATA_API_KEY"]
SYMBOLS = ["XAU/USD", "EUR/USD"]
START = dt.date(2024, 1, 1)
CHUNK_DAYS = 45          # 45 din x 96 candles = 4320 (< 5000 limit)
PAUSE = 9                # free plan: 8 requests / minute
os.makedirs("data", exist_ok=True)


def get(params):
    url = "https://api.twelvedata.com/time_series?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.loads(r.read().decode())
        if d.get("status") == "ok":
            return d["values"]
        msg = str(d.get("message", d))
        if "No data" in msg:
            return []
        print("  retry:", msg[:120])
        time.sleep(65)
    raise RuntimeError(msg)


for sym in SYMBOLS:
    rows = {}
    a = START
    today = dt.datetime.now(dt.timezone.utc).date()
    while a <= today:
        b = min(a + dt.timedelta(days=CHUNK_DAYS), today + dt.timedelta(days=1))
        vals = get(dict(symbol=sym, interval="15min", start_date=f"{a} 00:00:00",
                        end_date=f"{b} 00:00:00", timezone="UTC", outputsize=5000,
                        order="ASC", apikey=KEY))
        for v in vals:
            rows[v["datetime"]] = (v["open"], v["high"], v["low"], v["close"])
        print(f"{sym} {a} -> {b}: {len(vals)} candles")
        a = b
        time.sleep(PAUSE)
    out = f"data/{sym.replace('/', '_')}_15m.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "open", "high", "low", "close"])
        for t in sorted(rows):
            w.writerow([t, *rows[t]])
    print(f"Saved {out}: {len(rows)} candles, {min(rows)} .. {max(rows)}")
