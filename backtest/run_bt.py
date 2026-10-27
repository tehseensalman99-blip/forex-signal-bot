"""15m data par backtest chalata hai aur results/ mein save karta hai."""
import json
import os

import pandas as pd

import backtest as bt

SPREAD = {"XAU/USD": 0.30, "EUR/USD": 0.00010}
VARIANTS = {
    "current (SL 0.8% / TP 1.6%)": {"XAU/USD": (0.8, 1.6), "EUR/USD": (0.8, 1.6)},
    "old (Gold 0.4/0.8, EUR 0.2/0.4)": {"XAU/USD": (0.4, 0.8), "EUR/USD": (0.2, 0.4)},
}
YEARS = [2024, 2025, 2026]
os.makedirs("results", exist_ok=True)

summary = {"generated_utc": pd.Timestamp.now("UTC").isoformat(), "variants": {}, "data": {}}
all_trades = []
for vname, conf in VARIANTS.items():
    summary["variants"][vname] = {}
    for sym, (sl, tp) in conf.items():
        path = f"data/{sym.replace('/', '_')}_15m.csv"
        bars = bt.load_m1([path])
        summary["data"][sym] = dict(candles=len(bars), first=str(bars.index[0]), last=str(bars.index[-1]))
        trades, open_t = bt.run(bars, bt.Cfg(sym, sl, tp, SPREAD[sym]))
        res = {"all": bt.stats(trades), "open_trade_at_end": bool(open_t)}
        if not trades.empty:
            for y in YEARS:
                res[str(y)] = bt.stats(trades[trades.entry_time.dt.year == y])
            trades = trades.assign(variant=vname)
            all_trades.append(trades)
        summary["variants"][vname][sym] = res
        print(vname, sym, json.dumps(res["all"]))

pd.concat(all_trades).to_csv("results/trades.csv", index=False)
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("done")
