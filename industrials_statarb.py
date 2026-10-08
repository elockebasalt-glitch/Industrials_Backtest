"""
industrials_statarb.py - data loading, pair selection, signals, backtest and walk-forward for
relative-value strategies on the industrials universe in industrials_map.py.

Nothing here assumes a spread is stationary. Pairs are chosen by distance and a variance-ratio
check, traded around an adaptive mean with a time stop, and a second sleeve trades each name's
recent move against its sub-sector basket. No Streamlit code, so a trading script can import it.
"""
import io
import time
from itertools import product

import numpy as np
import pandas as pd

from industrials_map import ALL_TICKERS, BROAD, SUBSECTORS, TICKER_TO_GROUPS

VERSION = 3   # the app checks this so an out-of-date copy of this file is reported clearly
ANN = 252     # trading days


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


# ============================================================ optional uploads
def parse_valuation(csv_bytes, index):
    """Uploaded valuation multiples (date index, one column per ticker, e.g. forward P/E).
    Returns positive multiples aligned to the price dates; gaps of up to 10 days are filled."""
    raw = pd.read_csv(io.BytesIO(csv_bytes), index_col=0, parse_dates=True)
    raw = raw.apply(pd.to_numeric, errors="coerce").sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = raw[~raw.index.duplicated(keep="last")]
    raw.columns = [str(c).strip().upper().replace("/", "-") for c in raw.columns]
    return raw.where(raw > 0).reindex(raw.index.union(index)).ffill(limit=10).reindex(index)


def parse_earnings(csv_bytes, index, columns):
    """Uploaded earnings dates (two columns: ticker, date). Returns a True/False table marking
    the report day and the next trading day, since most reports land outside market hours."""
    raw = pd.read_csv(io.BytesIO(csv_bytes))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    tcol = next((c for c in raw.columns if c in ("ticker", "symbol")), raw.columns[0])
    dcol = next((c for c in raw.columns if "date" in c), raw.columns[1])
    out = pd.DataFrame(False, index=index, columns=columns)
    dates = pd.to_datetime(raw[dcol], errors="coerce").dt.normalize()
    for t, d in zip(raw[tcol].astype(str).str.strip().str.upper().str.replace("/", "-"), dates):
        if t in out.columns and pd.notna(d):
            i = index.searchsorted(d)
            out.iloc[i:i + 2, out.columns.get_loc(t)] = True
    return out


# ============================================================ events
def peer_residual_returns(px, groups=None):
    """Each name's daily log return minus the equal-weighted return of its sub-sector peers,
    itself excluded. Names in several sub-sectors use the average of their peer baskets."""
    groups = groups or SUBSECTORS
    r = np.log(px).diff()
    num = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    den = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    for members in groups.values():
        m = [t for t in members if t in px.columns]
        if len(m) < 2:
            continue
        sub = r[m]
        valid = sub.notna().values
        others = valid.sum(1)[:, None] - valid
        peer = (sub.fillna(0).values.sum(1)[:, None] - sub.fillna(0).values) / np.maximum(others, 1)
        ok = others > 0
        num[m] += np.where(ok, peer, 0.0)
        den[m] += ok
    market = r.mean(axis=1)
    peer = (num / den.where(den > 0)).apply(lambda c: c.fillna(market))
    return r - peer


def event_mask(px, jump_sigmas=4.0, vol_lb=60, groups=None, extra=None):
    """True on days a name moved against its peers by more than jump_sigmas of its own recent
    volatility: a price-only stand-in for earnings, guidance and deal news. extra adds known
    dates (see parse_earnings). Uses only data up to each day."""
    resid = peer_residual_returns(px, groups)
    vol = resid.rolling(vol_lb, min_periods=20).std().shift(1)
    ev = resid.abs() > jump_sigmas * vol
    if extra is not None:
        ev = ev | extra.reindex(index=px.index, columns=px.columns).fillna(False).astype(bool)
    return ev


def recent_events(ev, days=5):
    """True if the name had an event today or in the previous days-1 trading days."""
    return ev.astype(float).rolling(days, min_periods=1).max() > 0


# ============================================================ pair selection
def candidate_pairs(names, groups=None):
    """Index pairs to consider. With groups ({ticker: [sub-sectors]}), only pairs sharing a sub-sector."""
    iu, ju = np.triu_indices(len(names), 1)
    if groups:
        sets = [set(groups.get(t, [])) for t in names]
        keep = np.fromiter((bool(sets[i] & sets[j]) for i, j in zip(iu, ju)), bool, len(iu))
        iu, ju = iu[keep], ju[keep]
    return iu, ju


def _variance_ratio(S, q):
    """Variance of q-day spread changes over q times the variance of 1-day changes, per column.
    1 means a random walk; below 1 means moves tend to partly reverse within q days."""
    return (S[q:] - S[:-q]).var(0) / (q * np.diff(S, axis=0).var(0))


def screen_pairs(px, min_corr=0.4, dist_pct=20.0, vr_max=1.0, vr_q=10, top_k=20, max_per_name=3,
                 groups=None, persist=True):
    """Distance-method pair selection with a variance-ratio check. No stationarity test.

    For each candidate pair the spread is log(y) - log(x), a 1:1 dollar-neutral position.
    distance is the root-mean-square gap between the two normalised price paths over the window.
    A pair passes if its return correlation is at least min_corr, its distance is among the
    closest dist_pct percent of candidates, and its variance ratio at vr_q days is at most vr_max.
    persist=True also requires the variance ratio to pass on each half of the window separately,
    which removes many pairs that pass once by luck. Survivors are ranked by distance.
    mu and sigma are the spread's mean and standard deviation over the window (used by the
    formation-window mode).
    """
    cols = ["y", "x", "group", "distance", "var_ratio", "vr_1st_half", "vr_2nd_half", "corr", "sigma", "mu"]
    px = px.dropna(axis=1)
    n, T = px.shape[1], len(px)
    if n < 2 or T < 8 * vr_q:
        return pd.DataFrame(columns=cols)
    names = np.array(px.columns)
    L = np.log(px.values)
    corr = np.corrcoef(np.diff(L, axis=0).T)
    iu, ju = candidate_pairs(list(names), groups)
    if not len(iu):
        return pd.DataFrame(columns=cols)
    h = T // 2
    dist, vr, vr1, vr2, sig, mu = (np.empty(len(iu)) for _ in range(6))
    for a in range(0, len(iu), 400):                                  # chunked to bound memory
        sl = slice(a, a + 400)
        S = L[:, iu[sl]] - L[:, ju[sl]]
        dist[sl] = np.sqrt(((S - S[0]) ** 2).mean(0))
        vr[sl], vr1[sl], vr2[sl] = _variance_ratio(S, vr_q), _variance_ratio(S[:h], vr_q), _variance_ratio(S[h:], vr_q)
        sig[sl], mu[sl] = S.std(0), S.mean(0)
    df = pd.DataFrame({"y": names[iu], "x": names[ju], "distance": dist, "var_ratio": vr, "vr_1st_half": vr1,
                       "vr_2nd_half": vr2, "corr": corr[iu, ju], "sigma": sig, "mu": mu})
    keep = (df["corr"] >= min_corr) & (df["distance"] <= np.percentile(dist, dist_pct)) & (df["var_ratio"] <= vr_max)
    if persist:
        keep &= (df["vr_1st_half"] <= vr_max) & (df["vr_2nd_half"] <= vr_max)
    df = df[keep].sort_values("distance")
    picked, used = [], {}
    for row in df.itertuples(index=False):
        if used.get(row.y, 0) >= max_per_name or used.get(row.x, 0) >= max_per_name:
            continue
        used[row.y] = used.get(row.y, 0) + 1
        used[row.x] = used.get(row.x, 0) + 1
        common = sorted(set(TICKER_TO_GROUPS.get(row.y, [])) & set(TICKER_TO_GROUPS.get(row.x, [])))
        picked.append((row.y, row.x, common[0] if common else "cross-sector", row.distance, row.var_ratio,
                       row.vr_1st_half, row.vr_2nd_half, row.corr, row.sigma, row.mu))
        if len(picked) >= top_k:
            break
    return pd.DataFrame(picked, columns=cols)


# ============================================================ pair signals
MODES = ("rolling", "ewma", "kalman", "formation")


def _kalman(y, x, delta=1e-6, ve=1e-3):
    """Dynamic regression y = beta*x + alpha with beta and alpha drifting as random walks.
    Returns (one-step forecast errors, beta). Each value uses data up to that day only."""
    n = len(y)
    err, beta = np.full(n, np.nan), np.full(n, np.nan)
    vw = delta / (1 - delta)
    b = a = 0.0
    p11, p12, p22 = 1.0, 0.0, 1.0
    started = False
    for t in range(n):
        yt, xt = y[t], x[t]
        if np.isnan(yt) or np.isnan(xt):
            continue
        if not started:
            b, a, started = 1.0, yt - xt, True
        r11, r12, r22 = p11 + vw, p12, p22 + vw
        e = yt - (b * xt + a)
        f1, f2 = r11 * xt + r12, r12 * xt + r22               # R @ F with F = [x, 1]
        q = xt * f1 + f2 + ve
        k1, k2 = f1 / q, f2 / q
        b, a = b + k1 * e, a + k2 * e
        p11, p12, p22 = r11 - k1 * f1, r12 - k1 * f2, r22 - k2 * f2
        err[t], beta[t] = e, b
    return err, beta


def pair_zscore(sy, sx, mode="rolling", lookback=30, mu=None, sigma=None, delta=1e-6):
    """Z-score of the spread between two series, and the hedge ratio it implies.

    rolling    log(y/x) against its rolling mean and standard deviation over lookback days
    ewma       the same with exponentially weighted mean and standard deviation
    kalman     forecast error of a drifting regression of log(y) on log(x), scaled by its
               rolling standard deviation; the hedge ratio adapts
    formation  log(y/x) against a fixed mean and sigma from the selection window: the classic
               distance method, where z returns to zero only if the spread itself converges
    """
    ly, lx = np.log(sy), np.log(sx)
    one = pd.Series(1.0, index=sy.index)
    if mode == "kalman":
        e, b = _kalman(ly.values, lx.values, delta)
        e = pd.Series(e, index=sy.index)
        return e / e.rolling(lookback).std(), pd.Series(b, index=sy.index)
    s = ly - lx
    if mode == "formation":
        return (s - mu) / sigma, one
    if mode == "ewma":
        return (s - s.ewm(span=lookback, min_periods=lookback).mean()) / s.ewm(span=lookback, min_periods=lookback).std(), one
    return (s - s.rolling(lookback).mean()) / s.rolling(lookback).std(), one


def _run_states(zv, entry=2.0, exit_=0.5, stop=None, max_hold=15, block=None):
    """The trading rules, bar by bar. Returns (positions, bars held at the end, armed at the end).
    Position +1 is long the spread (long y, short x); -1 is short the spread.
    block marks bars where no new trade may open (for example just after an earnings jump)."""
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
            elif armed and (stop is None or abs(zi) < stop) and not (block is not None and block[i]):
                state, held = -np.sign(zi), 0               # z high -> short the spread
        else:
            held += 1
            if stop is not None and abs(zi) > stop:
                state, armed = 0.0, False                   # z-score stop
            elif state * zi > -exit_:
                state = 0.0                                 # back near the mean
            elif max_hold is not None and held >= max_hold:
                state, armed = 0.0, False                   # time stop
        pos[i] = state
    return pos, held, armed


def _pair_inputs(px, p, mode, sig, blocked):
    """Signal series, formation stats and entry block for one screened pair."""
    use_val = False
    if sig is not None and mode != "formation" and p.y in sig.columns and p.x in sig.columns:
        use_val = sig[[p.y, p.x]].reindex(px.index).notna().all(axis=1).mean() > 0.8   # coverage in this window
    sy, sx = (sig[p.y], sig[p.x]) if use_val else (px[p.y], px[p.x])
    block = None
    if blocked is not None:
        block = (blocked[p.y] | blocked[p.x]).reindex(px.index).fillna(False).values
    return sy.reindex(px.index), sx.reindex(px.index), use_val, block


def mr_weights(px, pairs, mode="rolling", lookback=30, entry=2.0, exit_=0.5, stop=None, max_hold=15,
               slots=None, sig=None, blocked=None):
    """Dollar weights for the pairs book. pairs is the screen_pairs table.

    slots is the number of pairs the book is sized for: each pair gets 1/slots of capital even
    when fewer pass the screen. sig is an optional table of valuation multiples; where both legs
    have one, the z-score is computed on the multiples and the trade is still done in the stocks.
    blocked is a True/False table of days on which a name may not open a new trade.
    """
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    n = max(len(pairs), slots or 0)
    for p in pairs.itertuples(index=False):
        sy, sx, use_val, block = _pair_inputs(px, p, mode, sig, blocked)
        z, b = pair_zscore(sy, sx, mode, lookback, p.mu, p.sigma)
        pos, _, _ = _run_states(z.values, entry, exit_, stop, max_hold, block)
        s = pd.Series(pos, index=px.index)
        b = pd.Series(1.0, index=px.index) if use_val else b.clip(0.1, 5).fillna(1.0)
        w[p.y] += s / (1 + b) / n
        w[p.x] += -s * b / (1 + b) / n
    return w


def current_signals(px, pairs, mode="rolling", lookback=30, entry=2.0, exit_=0.5, stop=None, max_hold=15,
                    slots=None, sig=None, blocked=None):
    """Where each screened pair stands after the last close in px, using the backtest's own rules.

    Status is "New entry" (opened at the last close), "Open" (opened earlier, still held),
    "Waiting" (z is past the entry level but the rules block a trade) or "No signal".
    """
    rows = []
    for p in pairs.itertuples(index=False):
        sy, sx, use_val, block = _pair_inputs(px, p, mode, sig, blocked)
        z, b = pair_zscore(sy, sx, mode, lookback, p.mu, p.sigma)
        pos, held, armed = _run_states(z.values, entry, exit_, stop, max_hold, block)
        z_now, side = float(z.iloc[-1]), pos[-1]
        hedge = 1.0 if use_val or not np.isfinite(b.iloc[-1]) else float(np.clip(b.iloc[-1], 0.1, 5))
        wy, wx = 1 / (1 + hedge), hedge / (1 + hedge)
        row = {"pair": f"{p.y}/{p.x}", "group": p.group, "z": z_now, "status": "No signal", "trade": "",
               "long": "", "short": "", "long_wt": np.nan, "short_wt": np.nan, "entered": None,
               "days_held": np.nan, "days_to_time_stop": np.nan, "hedge_ratio": hedge,
               "signal": "valuation" if use_val else "price"}
        if side != 0:
            lng, sht = (p.y, p.x) if side > 0 else (p.x, p.y)
            row.update(status="New entry" if held == 0 else "Open", trade=f"Long {lng} / Short {sht}",
                       long=lng, short=sht, long_wt=wy if side > 0 else wx, short_wt=wx if side > 0 else wy,
                       entered=px.index[len(px) - 1 - held].date(), days_held=held,
                       days_to_time_stop=(max_hold - held) if max_hold is not None else np.nan)
        elif np.isfinite(z_now) and abs(z_now) >= entry:
            why = ("past the stop level" if stop is not None and abs(z_now) >= stop else
                   "recent jump or earnings on a leg" if block is not None and block[-1] else
                   "stopped out, re-arms inside entry")
            row.update(status=f"Waiting ({why})")
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        order = out["status"].map(lambda v: 0 if v == "New entry" else 1 if v == "Open" else 2 if v.startswith("Waiting") else 3)
        out = out.assign(_o=order, _a=out["z"].abs()).sort_values(["_o", "_a"], ascending=[True, False]).drop(columns=["_o", "_a"])
    return out.reset_index(drop=True)


# ============================================================ basket reversal
def basket_scores(px, groups=None, lookback=10, vol_lb=60, cap=3.0, blocked=None):
    """Per sub-sector: each name's move against its peer basket over the last lookback days, in
    units of its usual lookback-day residual move. Returns {group: table of scores}; positive
    means the name has outrun its peers. Names with a recent event are left out (NaN)."""
    groups = groups or SUBSECTORS
    r = np.log(px).diff()
    out = {}
    for g, members in groups.items():
        m = [t for t in members if t in px.columns]
        if len(m) < 3:
            continue
        sub = r[m]
        valid = sub.notna().values
        others = valid.sum(1)[:, None] - valid
        peer = (sub.fillna(0).values.sum(1)[:, None] - sub.fillna(0).values) / np.maximum(others, 1)
        resid = sub - np.where(others > 0, peer, np.nan)
        score = (resid.rolling(lookback).sum() / (resid.rolling(vol_lb).std() * np.sqrt(lookback))).clip(-cap, cap)
        if blocked is not None:
            score = score.where(~blocked.reindex(index=px.index, columns=m).fillna(False).astype(bool))
        out[g] = score
    return out


def basket_reversal_weights(px, groups=None, lookback=10, hold=5, vol_lb=60, cap=3.0, blocked=None):
    """Trade returns, not levels: within each sub-sector go long the names that lagged their peer
    basket over the last lookback days and short the ones that led, dollar-neutral per sub-sector.
    Positions are averaged over the last hold days, so each day's signal is held for hold days.
    Gross exposure is at most 1. Needs no assumption that any spread is stationary."""
    w = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    total = 0
    for g, score in basket_scores(px, groups, lookback, vol_lb, cap, blocked).items():
        s = -score.sub(score.mean(axis=1), axis=0)
        gross = s.abs().sum(axis=1)
        w[s.columns] += s.div(gross.where(gross > 0), axis=0).fillna(0) * s.shape[1]
        total += s.shape[1]
    return (w / max(total, 1)).rolling(hold, min_periods=1).mean()


def vol_target(px, w, target=0.08, lb=60, max_lev=3.0):
    """Scale the whole book toward a target annualised vol, using only past realised vol."""
    pnl = (w.shift(1) * px.pct_change(fill_method=None)).sum(axis=1)
    realised = pnl.rolling(lb).std() * np.sqrt(ANN)
    lev = (target / realised.where(realised > 0)).shift(1).clip(upper=max_lev).fillna(0)
    return w.mul(lev, axis=0)


# ============================================================ backtest
def backtest(px, w, cost_bps=5, short_carry_bps=0, lag=1):
    """Daily results net of costs. Weights decided at close t earn returns from close t+lag-1 on.

    lag=1 assumes you trade at the same close the signal uses; lag=2 is the conservative check.
    cost_bps is charged on every unit of turnover; short_carry_bps is an annual borrow cost on shorts.
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
                       base_kw=None, grid=None, bt_kw=None, sig=None, blocked=None):
    """Each window: select pairs and (optionally) pick parameters on the training data only, then
    trade those pairs on the following test window. Returns (stitched weights, log)."""
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
            run = lambda data, p: mr_weights(data, picks, sig=sig, blocked=blocked, **{**base_kw, **p})
            best = max(combos, key=lambda p: sharpe(backtest(tr, run(tr, p), **bt_kw))) if len(combos) > 1 else combos[0]
            seg = px.iloc[start - train:start + test]              # training part is warm-up only
            wseg = run(seg, best).iloc[train:]
            W.loc[wseg.index] = wseg
            row.update(best)
            row["picked"] = ", ".join(f"{y}/{x}" for y, x in zip(picks["y"], picks["x"]))
        log.append(row)
    return W, pd.DataFrame(log)


def walk_forward_basket(px, dvol=None, train=504, test=63, min_dollar_vol=0.0, groups=None,
                        lookbacks=(10,), base_kw=None, bt_kw=None, blocked=None):
    """Each window: pick the lookback with the best training Sharpe, trade it on the test window.
    The basket book has no pair selection, so this is the only choice made from the data."""
    base_kw, bt_kw = base_kw or {}, bt_kw or {}
    W = pd.DataFrame(0.0, index=px.index, columns=px.columns)
    log = []
    for start in _windows(len(px), train, test):
        tr = px.iloc[start - train:start]
        names_ok = eligible(tr, dvol, min_dollar_vol)
        row = {"test_start": px.index[start].date(), "names": len(names_ok)}
        if len(names_ok) > 2:
            run = lambda data, lb: basket_reversal_weights(data[names_ok], groups, lookback=lb, blocked=blocked, **base_kw)
            best = max(lookbacks, key=lambda lb: sharpe(backtest(tr[names_ok], run(tr, lb), **bt_kw))) if len(lookbacks) > 1 else lookbacks[0]
            seg = px.iloc[start - train:start + test]
            wseg = run(seg, best).iloc[train:]
            W.loc[wseg.index, names_ok] = wseg
            row["lookback"] = best
        log.append(row)
    return W, pd.DataFrame(log)
