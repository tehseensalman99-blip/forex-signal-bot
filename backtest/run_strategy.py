"""Ek strategy ko ek period par chalata hai.
Env: STRATEGY, START (YYYY-MM-DD), END (YYYY-MM-DD, exclusive)."""
import json
import os

import numpy as np
import pandas as pd

import backtest as bt
from strategies import STRATEGIES

SPREAD = {"XAU/USD": 0.30, "EUR/USD": 0.00010}
NAME = os.environ["STRATEGY"]
START = pd.Timestamp(os.environ["START"], tz="UTC")
END = pd.Timestamp(os.environ["END"], tz="UTC")
PARAMS = json.loads(os.environ.get("PARAMS") or "{}")
ACCOUNT, RISK = 10000, 1.0

func, desc = STRATEGIES[NAME]
os.makedirs("results", exist_ok=True)
out = {"strategy": NAME, "description": desc, "params": PARAMS,
       "start": str(START.date()), "end": str(END.date()),
       "generated_utc": pd.Timestamp.now("UTC").isoformat(), "pairs": {}}
all_tr = []
for sym, sp in SPREAD.items():
    bars = bt.load_m1([f"data/{sym.replace('/', '_')}_15m.csv"])
    sig = func(bars, PARAMS)
    trades, open_t = bt.simulate(bars, sig, sp, sym, START, END)
    st = bt.stats(trades, RISK)
    if st.get("trades"):
        st["net_usd_on_10k"] = round(ACCOUNT * st["return_pct"] / 100, 0)
        st["longs"] = int((trades.side == "LONG").sum())
        st["shorts"] = int((trades.side == "SHORT").sum())
    st["open_trade_at_end"] = bool(open_t)
    out["pairs"][sym] = st
    if not trades.empty:
        all_tr.append(trades)
    print(sym, json.dumps(st))

tr = pd.concat(all_tr).sort_values("exit_time") if all_tr else pd.DataFrame()
out["combined"] = bt.stats(tr, RISK) if not tr.empty else {"trades": 0}
if not tr.empty:
    out["combined"]["net_usd_on_10k"] = round(ACCOUNT * out["combined"]["return_pct"] / 100, 0)
tag = f"{NAME}_{START.date()}_{END.date()}"
tr.to_csv(f"results/{tag}_trades.csv", index=False)
with open(f"results/{tag}.json", "w") as f:
    json.dump(out, f, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
print("combined", json.dumps(out["combined"]))
