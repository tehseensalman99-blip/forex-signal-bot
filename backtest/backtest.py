"""
EMA 9/21 + EMA 200 backtest — bilkul wahi logic jo GitHub bot (bot.py) chalata hai.

- 1-minute data load karta hai, 15m candles banata hai
- Signal: 15m candle close par crossover + EMA200 filter (bot jaisa)
- Entry: signal candle ka close (+ spread for BUY)
- Exit: SL/TP — 1-minute data se check hota hai ke pehle kaun laga
  (ek hi minute mein dono lagein to SL maana jata hai = conservative)
- Ek waqt mein ek trade per pair; trade khula ho to naye signal ignore
"""
import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

EMA_FAST, EMA_SLOW, EMA_TREND = 9, 21, 200
TF = "15min"
PKT = "Asia/Karachi"


# ----------------------------------------------------------------- loading
def load_m1(paths):
    """Dukascopy CSV ya HistData ASCII (M1) — dono format samajhta hai.
    Return: UTC index wala DataFrame [open, high, low, close] (bid prices)."""
    frames = []
    for p in paths:
        with open(p, "r", errors="ignore") as f:
            head = f.readline()
        if ";" in head and head[:8].isdigit():
            # HistData: 20240101 170000;open;high;low;close;vol  (EST, no DST = UTC-5)
            df = pd.read_csv(p, sep=";", header=None,
                             names=["t", "open", "high", "low", "close", "vol"])
            ts = pd.to_datetime(df["t"], format="%Y%m%d %H%M%S") + pd.Timedelta(hours=5)
            df.index = ts.dt.tz_localize("UTC")
        else:
            df = pd.read_csv(p)
            df.columns = [c.strip().lower() for c in df.columns]
            tcol = next(c for c in df.columns if "time" in c or "date" in c or c == "t")
            raw = df[tcol].astype(str).str.replace(" GMT", "", regex=False)
            raw = raw.str.replace(r"([+-]\d{4})$", "", regex=True).str.strip()
            ts = None
            for fmt in ("%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S",
                        "%Y.%m.%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
                try:
                    ts = pd.to_datetime(raw, format=fmt)
                    break
                except (ValueError, TypeError):
                    continue
            if ts is None:
                if pd.api.types.is_numeric_dtype(df[tcol]):
                    ts = pd.to_datetime(df[tcol], unit="ms")
                else:
                    ts = pd.to_datetime(raw)
            df.index = pd.DatetimeIndex(ts).tz_localize("UTC") if ts.dt.tz is None else ts
        frames.append(df[["open", "high", "low", "close"]].astype(float))
    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated(keep="first")]
    # Dukascopy weekend par flat (volume 0) candles deta hai — unhein hata do
    flat = (m1["high"] == m1["low"]) & (m1["open"] == m1["close"])
    m1 = m1[~(flat & (m1.index.dayofweek >= 5))]
    return m1


def to_tf(m1):
    o = m1.resample(TF, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return o


def ema_bot(values, n):
    """bot.py jaisa EMA: pehli n values ka SMA seed, phir EMA."""
    out = np.full(len(values), np.nan)
    if len(values) < n:
        return out
    k = 2 / (n + 1)
    s = values[:n].mean()
    out[n - 1] = s
    for i in range(n, len(values)):
        s = values[i] * k + s * (1 - k)
        out[i] = s
    return out


# ---------------------------------------------------------------- backtest
@dataclass
class Cfg:
    symbol: str
    sl_pct: float
    tp_pct: float
    spread: float  # price units


def run(m1, cfg: Cfg, start=None, end=None):
    bars = to_tf(m1)
    c = bars["close"].to_numpy()
    fast, slow, trend = ema_bot(c, EMA_FAST), ema_bot(c, EMA_SLOW), ema_bot(c, EMA_TREND)

    m1_t = m1.index.tz_convert(None).to_numpy()  # naive UTC datetime64
    bar_len = pd.Series(m1.index[:1000]).diff().median()  # 1min ya 15min data
    m1_o, m1_h, m1_l = m1["open"].to_numpy(), m1["high"].to_numpy(), m1["low"].to_numpy()
    bar_close_time = bars.index + pd.Timedelta(TF)  # signal is known at candle close

    trades = []
    busy_until = None
    for i in range(1, len(bars)):
        if np.isnan(trend[i]) or np.isnan(slow[i - 1]):
            continue
        t_sig = bar_close_time[i]
        if start is not None and t_sig < start:
            continue
        if end is not None and t_sig >= end:
            break
        if busy_until is not None and t_sig < busy_until:
            continue
        up = fast[i - 1] <= slow[i - 1] and fast[i] > slow[i] and c[i] > trend[i]
        dn = fast[i - 1] >= slow[i - 1] and fast[i] < slow[i] and c[i] < trend[i]
        if not (up or dn):
            continue
        side = 1 if up else -1
        ref = c[i]  # bot isi price par SL/TP banata hai
        sl = ref * (1 - side * cfg.sl_pct / 100)
        tp = ref * (1 + side * cfg.tp_pct / 100)
        sp = cfg.spread
        entry = ref + sp if side == 1 else ref  # BUY ask par, SELL bid par

        j0 = np.searchsorted(m1_t, t_sig.tz_convert(None).to_datetime64(), side="left")
        if j0 >= len(m1_t):
            break
        # Exit trigger: BUY band hota hai bid par; SELL band hota hai ask (= bid + spread) par
        if side == 1:
            hit_sl = m1_l[j0:] <= sl
            hit_tp = m1_h[j0:] >= tp
        else:
            hit_sl = (m1_h[j0:] + sp) >= sl
            hit_tp = (m1_l[j0:] + sp) <= tp
        k_sl = np.argmax(hit_sl) if hit_sl.any() else None
        k_tp = np.argmax(hit_tp) if hit_tp.any() else None
        if k_sl is None and k_tp is None:
            trades.append(dict(open_trade=True, side=side, entry_time=t_sig))
            break  # data khatam, trade abhi khula hai
        if k_tp is None or (k_sl is not None and k_sl <= k_tp):
            k, result = k_sl, "SL"
            lvl = sl
        else:
            k, result = k_tp, "TP"
            lvl = tp
        j = j0 + k
        # Gap (jaise weekend) agar level ke paar khule to fill open par
        o_eff = m1_o[j] if side == 1 else m1_o[j] + sp
        if result == "SL":
            fill = min(lvl, o_eff) if side == 1 else max(lvl, o_eff)
        else:
            fill = max(lvl, o_eff) if side == 1 else min(lvl, o_eff)
        if j == j0:  # pehla minute — open par gap check bematlab, level hi lo
            fill = lvl
        pnl = (fill - entry) * side
        risk = ref * cfg.sl_pct / 100
        exit_time = pd.Timestamp(m1_t[j]).tz_localize("UTC") + bar_len
        trades.append(dict(
            symbol=cfg.symbol, side="LONG" if side == 1 else "SHORT",
            entry_time=t_sig, exit_time=exit_time, ref=ref, entry=entry, sl=sl, tp=tp,
            exit=fill, result=result, pnl_price=pnl, R=pnl / risk,
            pnl_pct=pnl / entry * 100,
        ))
        busy_until = exit_time
    closed = [t for t in trades if not t.get("open_trade")]
    return pd.DataFrame(closed), [t for t in trades if t.get("open_trade")]


def simulate(m1, sig, spread, symbol, start=None, end=None):
    """Generic: sig = DataFrame(index=15m bar start, side, sl, tp).
    Signal bar close par entry; exit SL/TP data (m1 ya 15m) se; ek waqt mein ek trade."""
    m1_t = m1.index.tz_convert(None).to_numpy()
    bar_len = pd.Series(m1.index[:1000]).diff().median()
    m1_o, m1_h, m1_l = m1["open"].to_numpy(), m1["high"].to_numpy(), m1["low"].to_numpy()
    closes = to_tf(m1)["close"].reindex(sig.index)
    s = sig[sig["side"] != 0].dropna()
    trades, open_trades = [], []
    busy_until = None
    for t_bar, row in s.iterrows():
        t_sig = t_bar + pd.Timedelta(TF)
        if start is not None and t_sig < start:
            continue
        if end is not None and t_sig >= end:
            break
        if busy_until is not None and t_sig < busy_until:
            continue
        side = int(row.side)
        ref, sl, tp = float(closes.loc[t_bar]), float(row.sl), float(row.tp)
        entry = ref + spread if side == 1 else ref
        j0 = np.searchsorted(m1_t, t_sig.tz_convert(None).to_datetime64(), side="left")
        if j0 >= len(m1_t):
            open_trades.append(dict(side=side, entry_time=t_sig))
            break
        if side == 1:
            hit_sl, hit_tp = m1_l[j0:] <= sl, m1_h[j0:] >= tp
        else:
            hit_sl, hit_tp = (m1_h[j0:] + spread) >= sl, (m1_l[j0:] + spread) <= tp
        k_sl = int(np.argmax(hit_sl)) if hit_sl.any() else None
        k_tp = int(np.argmax(hit_tp)) if hit_tp.any() else None
        if k_sl is None and k_tp is None:
            open_trades.append(dict(side=side, entry_time=t_sig))
            break
        if k_tp is None or (k_sl is not None and k_sl <= k_tp):
            k, result, lvl = k_sl, "SL", sl
        else:
            k, result, lvl = k_tp, "TP", tp
        j = j0 + k
        o_eff = m1_o[j] if side == 1 else m1_o[j] + spread
        if j == j0:
            fill = lvl
        elif result == "SL":
            fill = min(lvl, o_eff) if side == 1 else max(lvl, o_eff)
        else:
            fill = max(lvl, o_eff) if side == 1 else min(lvl, o_eff)
        pnl = (fill - entry) * side
        risk = abs(ref - sl)
        exit_time = pd.Timestamp(m1_t[j]).tz_localize("UTC") + bar_len
        trades.append(dict(symbol=symbol, side="LONG" if side == 1 else "SHORT",
                           entry_time=t_sig, exit_time=exit_time, ref=ref, entry=entry,
                           sl=sl, tp=tp, exit=fill, result=result, pnl_price=pnl,
                           R=pnl / risk, pnl_pct=pnl / entry * 100))
        busy_until = exit_time
    return pd.DataFrame(trades), open_trades


# ----------------------------------------------------------------- metrics
def stats(df, risk_pct=1.0):
    if df.empty:
        return dict(trades=0)
    R = df["R"].to_numpy()
    eq = np.cumprod(1 + R * risk_pct / 100)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = (eq / peak - 1).min() * 100
    wins = R[R > 0].sum()
    losses = -R[R < 0].sum()
    # lagataar haar
    streak = mx = 0
    for r in R:
        streak = streak + 1 if r < 0 else 0
        mx = max(mx, streak)
    dur = (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600
    return dict(
        trades=int(len(df)),
        wins=int((R > 0).sum()),
        losses=int((R <= 0).sum()),
        win_rate=round(float((R > 0).mean() * 100), 1),
        total_R=round(float(R.sum()), 2),
        avg_R=round(float(R.mean()), 3),
        profit_factor=round(float(wins / losses), 2) if losses > 0 else None,
        return_pct=round(float((eq[-1] - 1) * 100), 1),
        max_dd_pct=round(float(dd), 1),
        max_losing_streak=int(mx),
        avg_hours_in_trade=round(float(dur.mean()), 1),
    )


def session_of(ts_utc):
    h = ts_utc.hour
    if 0 <= h < 7:
        return "Asia (5am–12pm PKT)"
    if 7 <= h < 13:
        return "London (12pm–6pm PKT)"
    if 13 <= h < 21:
        return "New York (6pm–2am PKT)"
    return "Late/Close (2am–5am PKT)"


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    cfg = json.load(open(cfg_path))
    print(json.dumps(cfg, indent=2))
