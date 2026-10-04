"""
industrials_statarb.py - data loading, pair screening, signals, backtest and walk-forward
for mean-reversion pairs and trend following on the industrials universe in
industrials_map.py. No Streamlit code here, so a trading script can import it too.
"""
import time
from itertools import product

import numpy as np
import pandas as pd

from industrials_map import ALL_TICKERS, BROAD, SUBSECTORS, TICKER_TO_GROUPS

ANN = 252  # trading days


# ============================================================ data
def clean(close, dvol=None, min_rows=250):
    """Drop an unfinished bar for today, bad prints and names with too little history."""
    close = close.sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")]
    now = pd.Timestamp.now(tz="America/New_York")
    if len(close) and close.index[-1].date() == now.date() and now.hour < 17:
        close = close.iloc[:-1]                                  # market not closed yet
    close = close.where(close > 0).ffill(limit=3)
    keep = close.columns[close.notna().sum() >= min_rows]
    close = close[keep].dropna(how="all")
    if dvol is not None:
        dvol.index = pd.to_datetime(dvol.index).tz_localize(None).normalize()
        dvol = dvol[~dvol.index.duplicated(keep="last")].reindex(close.index)[keep]
    return close, dvol


def _stooq(ticker):
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    df = pd.read_csv(url, parse_dates=["Date"]).set_index("Date").sort_index()
    return df["Close"].rename(ticker), (df["Close"] * df["Volume"]).rename(ticker)


def fetch_yahoo(tickers, start, bench=BROAD):
    """Adjusted daily closes and dollar volume from Yahoo Finance, Stooq for anything missing.
    Returns (close, dollar_volume, failed, benchmark close)."""
    import yfinance as yf

    names = list(dict.fromkeys(list(tickers) + [bench]))
    closes, vols = [], []
    for i in range(0, len(names), 50):
        chunk = names[i:i + 50]
        df = yf.download(chunk, start=str(start), interval="1d", auto_adjust=True,
                         progress=False, threads=True)
        if df is None or df.empty:
            continue
        c, v = df["Close"], df["Volume"]
        if isinstance(c, pd.Series):
            c, v = c.to_frame(chunk[0]), v.to_frame(chunk[0])
        closes.append(c)
        vols.append(c * v)
        time.sleep(0.5)
    close = pd.concat(closes, axis=1).dropna(axis=1, how="all") if closes else pd.DataFrame()
    dvol = pd.concat(vols, axis=1)[close.columns] if closes else pd.DataFrame()
    for t in [n for n in names if n not in close.columns]:
        try:
            c, v = _stooq(t)
            c, v = c[c.index >= pd.Timestamp(start)], v[v.index >= pd.Timestamp(start)]
            close, dvol = pd.concat([close, c], axis=1), pd.concat([dvol, v], axis=1)
        except Exception:
            pass
    if close.empty:
        raise RuntimeError("Yahoo Finance returned no data (it rate-limits shared servers; "
                           "try again in a few minutes, or upload a CSV).")
    close, dvol = clean(close, dvol)
    bench_px = close[bench] if bench in close.columns else None
    close, dvol = close.drop(columns=bench, errors="ignore"), dvol.drop(columns=bench, errors="ignore")
    return close, dvol, [t for t in tickers if t not in close.columns], bench_px


def synthetic(tickers, n=2600, seed=1):
    """Made-up prices with sub-sector factors and some cointegrated pairs. Not market data."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=n)
    drift = pd.Series(rng.normal(0.0003, 0.002, n)).ewm(span=60).mean().values
    mkt = np.cumsum(drift + rng.normal(0, 0.009, n))
    sector = {g: np.cumsum(rng.normal(0, 0.006, n)) for g in SUBSECTORS}
    logs, vols, prev = {}, {}, {}
    for k, t in enumerate(tickers):
        g = TICKER_TO_GROUPS.get(t, ["Multis"])[0]
        base = rng.uniform(0.8, 1.2) * mkt + sector[g]
        if g in prev and k % 3 == 1:                           # cointegrated with a group peer
            theta = rng.uniform(0.03, 0.12)
            ou = np.zeros(n)
            for i in range(1, n):
                ou[i] = ou[i - 1] * (1 - theta) + rng.normal(0, 0.012)
            lp = rng.uniform(0.8, 1.2) * prev[g] + ou
        else:
            lp = base + np.cumsum(rng.normal(0, 0.012, n))
        prev[g] = lp
        lp = lp + rng.uniform(3, 6)
        if k % 9 == 8:
            lp[: int(rng.uniform(200, n - 600))] = np.nan       # later listing dates
        logs[t] = lp
        vols[t] = np.exp(rng.normal(np.log(8e7), 0.6, n))
    close = np.exp(pd.DataFrame(logs, index=idx))
    return close, pd.DataFrame(vols, index=idx).where(close.notna()), [], pd.Series(100 * np.exp(mkt), index=idx)


def eligible(px, dvol=None, min_dollar_vol=0.0):
    """Names with a full price history in this window and enough median daily dollar volume."""
    ok = px.notna().all()
    if dvol is not None and min_dollar_vol > 0:
        ok &= dvol.reindex(px.index).median() >= min_dollar_vol
    return list(ok[ok].index)


# ============================================================ pair screening
def candidate_pairs(names, groups=None):
    """Index pairs to test. With groups ({ticker: [sub-sectors]}), only pairs sharing a sub-sector."""
    iu, ju = np.triu_indices(len(names), 1)
    if groups:
        sets = [set(groups.get(t, [])) for t in names]
        keep = np.fromiter((bool(sets[i] & sets[j]) for i, j in zip(iu, ju)), bool, len(iu))
        iu, ju = iu[keep], ju[keep]
    return iu, ju


def screen_pairs(px, min_corr=0.5, adf_max=-3.34, hl_range=(2, 60), top_k=20, max_per_name=3, groups=None):
    """Rank candidate pairs in the window by an Engle-Granger style test on the log-price residual.

    adf_t is the Dickey-Fuller t-stat (no lags) of the residual; -3.34 is roughly the 5%
    critical value for two series. Testing thousands of pairs means many pass by chance,
    so the only fair test of this screen is trading its picks on later data (walk_forward_pairs).
    beta, mu and sigma describe the spread log(y) - beta*log(x) over this window; fixed-parameter
    trading freezes them.
    """
    cols = ["y", "x", "group", "beta", "adf_t", "half_life", "corr", "sigma", "mu"]
    px = px.dropna(axis=1)
    n = px.shape[1]
    if n < 2 or len(px) < 60:
        return pd.DataFrame(columns=cols)
    names = np.array(px.columns)
    L = np.log(px.values)
    corr = np.corrcoef(np.diff(L, axis=0).T)
    Lc = L - L.mean(0)
    cov = Lc.T @ Lc / len(L)
    var = np.diag(cov)
    iu, ju = candidate_pairs(list(names), groups)
    if not len(iu):
        return pd.DataFrame(columns=cols)
    yi, xi = np.concatenate([iu, ju]), np.concatenate([ju, iu])      # both regression directions
    beta = cov[yi, xi] / var[xi]
    mu = L.mean(0)[yi] - beta * L.mean(0)[xi]                        # mean of log(y) - beta*log(x)
    t, rho, sig = np.empty(len(yi)), np.empty(len(yi)), np.empty(len(yi))
    for a in range(0, len(yi), 400):                                  # chunked to bound memory
        sl = slice(a, a + 400)
        E = Lc[:, yi[sl]] - Lc[:, xi[sl]] * beta[sl]
        e0, de = E[:-1], np.diff(E, axis=0)
        sxx = (e0 ** 2).sum(0)
        rh = (e0 * de).sum(0) / sxx
        s2 = ((de - e0 * rh) ** 2).sum(0) / (len(de) - 1)
        rho[sl], t[sl], sig[sl] = rh, rh / np.sqrt(s2 / sxx), E.std(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.where(rho < 0, -np.log(2) / np.log1p(np.clip(rho, -0.999, -1e-12)), np.inf)
    df = pd.DataFrame({"y": names[yi], "x": names[xi], "beta": beta, "adf_t": t,
                       "half_life": hl, "corr": corr[yi, xi], "sigma": sig, "mu": mu})
    df = df[(df["corr"] >= min_corr) & (df["adf_t"] <= adf_max) & (df["beta"] > 0)
            & df["half_life"].between(*hl_range)].sort_values("adf_t")
    picked, used, seen = [], {}, set()
    for row in df.itertuples(index=False):
        key = frozenset((row.y, row.x))
        if key in seen or used.get(row.y, 0) >= max_per_name or used.get(row.x, 0) >= max_per_name:
            continue
        seen.add(key)
        used[row.y] = used.get(row.y, 0) + 1
        used[row.x] = used.get(row.x, 0) + 1
        common = sorted(set(TICKER_TO_GROUPS.get(row.y, [])) & set(TICKER_TO_GROUPS.get(row.x, [])))
        picked.append((row.y, row.x, common[0] if common else "cross-sector", row.beta, row.adf_t, row.half_life, row.corr, row.sigma, row.mu))
        if len(picked) >= top_k:
            break
    return pd.DataFrame(picked, columns=cols)


# ============================================================ signals
def pair_zscore(py, px_, lookback=60, beta=None, mu=None, sigma=None):
    """Z-score of the spread log(y) - beta*log(x), and the hedge ratio used.

    With beta, mu and sigma given they are held fixed (estimated on a formation window), so z
    only returns to zero if the spread itself converges. Otherwise all three come from a
    rolling regression over `lookback` bars and move every day.
    """
    ly, lx = np.log(py), np.log(px_)
    if beta is not None:
        return (ly - beta * lx - mu) / sigma, pd.Series(float(beta), index=py.index)
    beta = ly.rolling(lookback).cov(lx) / lx.rolling(lookback).var()
    alpha = ly.rolling(lookback).mean() - beta * lx.rolling(lookback).mean()
    resid = ly - (alpha + beta * lx)
    return resid / resid.rolling(lookback).std(), beta


def pair_legs(py, px_, lookback=60, entry=2.0, exit_=0.5, stop=4.0, max_hold=30, use_macd=False,
              beta=None, mu=None, sigma=None):
    """Z-score pairs trade. Returns dollar weights for (y, x) with gross exposure 1 when in a trade.

    Enter when |z| > entry (with use_macd, only once the MACD histogram of z has turned back
    toward zero). Exit when z is back inside exit_ (exit_=0 means the spread crossed its mean).
    stop is a z-score stop and max_hold a time stop in bars; pass None to switch either off.
    After either stop the pair waits until |z| is back inside entry before it can trade again.
    Pass beta, mu and sigma to trade a fixed spread instead of the rolling one (see pair_zscore).
    """
    z, b = pair_zscore(py, px_, lookback, beta, mu, sigma)
    macd = z.ewm(span=12).mean() - z.ewm(span=26).mean()
    hist = (macd - macd.ewm(span=9).mean()).values
    zv = z.values
    state, held, armed = 0.0, 0, True
    pos = np.zeros(len(zv))
    for i in range(len(zv)):
        zi = zv[i]
        if np.isnan(zi):
            state = 0.0
            continue
        if state == 0:
            if abs(zi) < entry:
                armed = True
            elif armed and (stop is None or abs(zi) < stop):
                side = -np.sign(zi)                         # z high -> short the spread
                if not use_macd or hist[i] * side > 0:
                    state, held = side, 0
        else:
            held += 1
            if stop is not None and abs(zi) > stop:
                state, armed = 0.0, False                   # z-score stop
            elif state * zi > -exit_:
                state = 0.0                                 # converged
            elif max_hold is not None and held >= max_hold:
                state, armed = 0.0, False                   # time stop
        pos[i] = state
    s = pd.Series(pos, index=py.index)
    b = b.clip(0.1, 5).fillna(0)
    gross = 1 + b
    return s / gross, -s * b / gross


def mr_weights(px, pairs, fixed=False, hold_half_lives=None, **kw):
    """Equal capital per pair. pairs is the screen_pairs table (or a list of (y, x) tuples).

    fixed=True trades each pair with the hedge ratio, mean and sigma frozen from the screen window.
    hold_half_lives sets each pair's time stop to that multiple of its own half-life.
    """
    if not isinstance(pairs, pd.DataFrame):
        pairs = pd.DataFrame(list(pairs), columns=["y", "x"])
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for p in pairs.itertuples(index=False):
        k = dict(kw)
        if fixed:
            k.update(beta=p.beta, mu=p.mu, sigma=p.sigma)
        if hold_half_lives:
            k["max_hold"] = int(max(2, round(hold_half_lives * p.half_life)))
        wy, wx = pair_legs(px[p.y], px[p.x], **k)
        w[p.y] += wy / len(pairs)
        w[p.x] += wx / len(pairs)
    return w


def trend_weights(px, speeds=((8, 32), (16, 64), (32, 128)), vol_lb=30, target_vol=0.20,
                  cap=0.25, long_only=False):
    """Blend of EMA crossovers, scaled by volatility, equal risk per name that has data."""
    r = px.pct_change(fill_method=None)
    dvol = r.ewm(span=vol_lb, min_periods=vol_lb).std()
    fc = 0
    for fast, slow in speeds:
        raw = (px.ewm(span=fast, min_periods=slow).mean()
               - px.ewm(span=slow, min_periods=slow).mean()) / (px * dvol)
        scaled = raw / raw.abs().expanding(min_periods=60).mean()        # uses past data only
        fc = fc + scaled.clip(-2, 2) / len(speeds)
    if long_only:
        fc = fc.clip(lower=0)
    live = fc.notna().sum(axis=1).clip(lower=1)
    w = (fc * target_vol / (dvol * np.sqrt(ANN))).div(live, axis=0)
    return w.clip(-cap, cap).fillna(0)


def vol_target(px, w, target=0.15, lb=60, max_lev=3.0):
    """Scale the whole book toward a target annualised vol, using only past realised vol."""
    pnl = (w.shift(1) * px.pct_change(fill_method=None)).sum(axis=1)
    realised = pnl.rolling(lb).std() * np.sqrt(ANN)
    lev = (target / realised.where(realised > 0)).shift(1).clip(upper=max_lev).fillna(0)
    return w.mul(lev, axis=0)


# ============================================================ backtest
def backtest(px, w, cost_bps=5, short_carry_bps=0, lag=1):
    """Daily results net of costs. Weights decided at close t earn returns from close t+lag-1 on.

    lag=1 assumes you trade at the same close the signal uses; lag=2 is the conservative check.
    cost_bps is charged on every unit of turnover; short_carry_bps is an annual cost on shorts
    (stock borrow).
    """
    r = px.pct_change(fill_method=None).fillna(0)
    held = w.reindex(index=px.index, columns=px.columns).fillna(0).shift(lag).fillna(0)
    turnover = held.diff().abs().sum(axis=1).fillna(0)
    carry = held.clip(upper=0).abs().sum(axis=1) * short_carry_bps / 1e4 / ANN
    ret = (held * r).sum(axis=1) - turnover * cost_bps / 1e4 - carry
    return pd.DataFrame({"ret": ret, "gross": held.abs().sum(axis=1), "turnover": turnover})


def stats(bt):
    r = bt["ret"].dropna()
    if len(r) < 2 or r.std() == 0:
        return {"Sharpe": 0.0, "Ann. return": 0.0, "Vol": 0.0, "Max DD": 0.0,
                "Avg gross": 0.0, "Turnover/yr": 0.0, "Days": len(r)}
    eq = (1 + r).cumprod()
    vol = r.std() * np.sqrt(ANN)
    return {
        "Sharpe": float(r.mean() * ANN / vol),
        "Ann. return": float(eq.iloc[-1] ** (ANN / len(r)) - 1),
        "Vol": float(vol),
        "Max DD": float((eq / eq.cummax() - 1).min()),
        "Avg gross": float(bt["gross"].mean()),
        "Turnover/yr": float(bt["turnover"].mean() * ANN),
        "Days": len(r),
    }


def sharpe(bt):
    return stats(bt)["Sharpe"]


# ============================================================ walk-forward
def _windows(n, train, test):
    return range(train, n, test)


def walk_forward_pairs(px, dvol=None, train=504, test=63, min_dollar_vol=0.0, screen_kw=None,
                       base_kw=None, grid=None, bt_kw=None):
    """Each window: screen pairs and (optionally) pick parameters on the training data only,
    then trade those pairs on the following test window. Returns (stitched weights, log).
    With fixed=True in base_kw, each pair's hedge ratio, mean and sigma come from the training
    window and stay frozen through the test window."""
    screen_kw, base_kw, bt_kw = screen_kw or {}, base_kw or {}, bt_kw or {}
    combos = [dict(zip(grid, v)) for v in product(*grid.values())] if grid else [{}]
    W = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    log = []
    for start in _windows(len(px), train, test):
        tr = px.iloc[start - train:start]
        names_ok = eligible(tr, dvol, min_dollar_vol)
        picks = screen_pairs(tr[names_ok], **screen_kw)
        row = {"test_start": px.index[start].date(), "names": len(names_ok), "pairs": len(picks)}
        if len(picks):
            best = max(combos, key=lambda p: sharpe(backtest(tr, mr_weights(tr, picks, **{**base_kw, **p}), **bt_kw)))
            seg = px.iloc[start - train:start + test]              # training part is warm-up only
            wseg = mr_weights(seg, picks, **{**base_kw, **best}).iloc[train:]
            W.loc[wseg.index] = wseg
            row.update(best)
            row["picked"] = ", ".join(f"{y}/{x}" for y, x in zip(picks["y"], picks["x"]))
        log.append(row)
    return W, pd.DataFrame(log)


def walk_forward_trend(px, dvol=None, train=504, test=63, min_dollar_vol=0.0, speed_sets=None,
                       base_kw=None, bt_kw=None):
    """Each window: pick the EMA speed set with the best training Sharpe, trade it on the test window."""
    base_kw, bt_kw = base_kw or {}, bt_kw or {}
    speed_sets = speed_sets or [((8, 32), (16, 64), (32, 128))]
    W = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    log = []
    for start in _windows(len(px), train, test):
        tr = px.iloc[start - train:start]
        names_ok = eligible(tr, dvol, min_dollar_vol)
        row = {"test_start": px.index[start].date(), "names": len(names_ok)}
        if names_ok:
            best = max(speed_sets, key=lambda s: sharpe(backtest(tr[names_ok], trend_weights(tr[names_ok], speeds=s, **base_kw), **bt_kw)))
            seg = px.iloc[start - train:start + test][names_ok]
            wseg = trend_weights(seg, speeds=best, **base_kw).iloc[train:]
            W.loc[wseg.index, names_ok] = wseg
            row["speeds"] = str(best)
        log.append(row)
    return W, pd.DataFrame(log)
