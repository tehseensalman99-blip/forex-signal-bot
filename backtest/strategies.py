"""Strategies. Har strategy 15m bars leti hai aur har bar ke liye signal deti hai:
side (1 = BUY, -1 = SELL, 0 = kuch nahi), sl, tp (price levels, candle close se)."""
import numpy as np
import pandas as pd


# ------------------------------------------------------------- indicators
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def atr(b, n=14):
    pc = b["close"].shift()
    tr = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(), (b["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _out(b, long_c, short_c, sl_dist, tp_dist):
    side = np.where(long_c, 1, np.where(short_c, -1, 0))
    c = b["close"]
    return pd.DataFrame({
        "side": side,
        "sl": np.where(side == 1, c - sl_dist, c + sl_dist),
        "tp": np.where(side == 1, c + tp_dist, c - tp_dist),
    }, index=b.index)


# ------------------------------------------------------------- strategies
def s0_ema_cross(b, p):
    """Live bot wali: EMA 9/21 cross + EMA 200 filter, SL/TP % mein."""
    c = b["close"]
    f, s, t = ema(c, 9), ema(c, 21), ema(c, 200)
    lc = (f.shift() <= s.shift()) & (f > s) & (c > t)
    sc = (f.shift() >= s.shift()) & (f < s) & (c < t)
    return _out(b, lc, sc, c * p.get("sl_pct", 0.8) / 100, c * p.get("tp_pct", 1.6) / 100)


def s1_trend_pullback(b, p):
    """Trend Pullback: EMA50 & price EMA200 ke upar, RSI 40 ko neeche se cross kare,
    close EMA21 ke upar. SL 1.5 ATR, TP 3 ATR. Sirf London + New York session."""
    c = b["close"]
    e21, e50, e200 = ema(c, 21), ema(c, 50), ema(c, 200)
    r, a = rsi(c, 14), atr(b, 14)
    hour = (b.index + pd.Timedelta("15min")).hour  # signal candle close ka waqt (UTC)
    session = (hour >= 7) & (hour < 20)
    lo, hi = p.get("rsi_lo", 40), p.get("rsi_hi", 60)
    lc = (c > e200) & (e50 > e200) & (r.shift() < lo) & (r >= lo) & (c > e21) & session
    sc = (c < e200) & (e50 < e200) & (r.shift() > hi) & (r <= hi) & (c < e21) & session
    return _out(b, lc, sc, a * p.get("sl_atr", 1.5), a * p.get("tp_atr", 3.0))


STRATEGIES = {
    "s0_ema_cross": (s0_ema_cross, "EMA 9/21 cross + EMA200 filter, SL 0.8% / TP 1.6% (live bot)"),
    "s1_trend_pullback": (s1_trend_pullback,
                          "Trend Pullback: EMA50>EMA200 trend, RSI(14) 40/60 reclaim, close vs EMA21, "
                          "London+NY session, SL 1.5xATR / TP 3xATR"),
}
