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


def adx(b, n=14):
    h, l, c = b["high"], b["low"], b["close"]
    up, dn = h.diff(), -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(b, n)
    pdi = 100 * pd.Series(pdm, index=b.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    mdi = 100 * pd.Series(mdm, index=b.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def s2_london_breakout(b, p):
    """London Breakout: Asia session (00:00-07:00 UTC) ka high/low. 07:00-11:00 UTC mein
    pehli 15m candle jo range ke upar close ho = BUY, neeche = SELL (din mein ek hi).
    SL = range ka aadha (kam se kam 1 ATR), TP = 2 x SL."""
    c = b["close"]
    a = atr(b, 14)
    day = b.index.normalize()
    asia = b.index.hour < 7
    hi = b["high"].where(asia).groupby(day).transform("max")
    lo = b["low"].where(asia).groupby(day).transform("min")
    close_t = b.index + pd.Timedelta("15min")
    mins = close_t.hour * 60 + close_t.minute
    window = (mins > 7 * 60) & (mins <= 11 * 60) & (close_t.normalize() == day)
    brk_up = window & (c > hi)
    brk_dn = window & (c < lo)
    any_brk = (brk_up | brk_dn).astype(int)
    first = any_brk.groupby(day).cumsum().eq(1) & (any_brk == 1)
    lc, sc = brk_up & first, brk_dn & first
    sl = np.maximum((hi - lo) / 2, a * p.get("min_sl_atr", 1.0))
    return _out(b, lc, sc, sl, sl * p.get("rr", 2.0))


def s3_bb_reversion(b, p):
    """Bollinger mean reversion: ADX(14) < 20 (market range mein). Close neeche wale band ke
    bahar se wapas andar aaye = BUY, upar wale band se wapas andar = SELL. TP = beech wala band
    (SMA 20), SL = 1.5 ATR. TP 0.5 ATR se kam ho to skip. Sirf London + New York."""
    c = b["close"]
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    up_b, lo_b = mid + 2 * sd, mid - 2 * sd
    a, ax = atr(b, 14), adx(b, 14)
    hour = (b.index + pd.Timedelta("15min")).hour
    session = (hour >= 7) & (hour < 20)
    ranging = ax < p.get("adx_max", 20)
    tp_dist = (mid - c).abs()
    room = tp_dist >= 0.5 * a
    lc = (c.shift() < lo_b.shift()) & (c > lo_b) & (c < mid) & ranging & session & room
    sc = (c.shift() > up_b.shift()) & (c < up_b) & (c > mid) & ranging & session & room
    return _out(b, lc, sc, a * p.get("sl_atr", 1.5), tp_dist)


STRATEGIES = {
    "s2_london_breakout": (s2_london_breakout,
                           "London Breakout: Asia range 00-07 UTC, first 15m close outside it 07-11 UTC, "
                           "one trade a day, SL half range (min 1 ATR), TP 2x SL"),
    "s3_bb_reversion": (s3_bb_reversion,
                        "Bollinger reversion: ADX<20, close re-enters BB(20,2), TP middle band, SL 1.5 ATR, "
                        "London+NY session"),
    "s0_ema_cross": (s0_ema_cross, "EMA 9/21 cross + EMA200 filter, SL 0.8% / TP 1.6% (live bot)"),
    "s1_trend_pullback": (s1_trend_pullback,
                          "Trend Pullback: EMA50>EMA200 trend, RSI(14) 40/60 reclaim, close vs EMA21, "
                          "London+NY session, SL 1.5xATR / TP 3xATR"),
}
