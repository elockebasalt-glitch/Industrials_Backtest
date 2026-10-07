"""
Industrials stat-arb research app: sub-sector pair screening, in-sample backtest
and walk-forward test. Needs industrials_statarb.py and industrials_map.py beside it.

    streamlit run industrials_statarb_app.py
"""
import datetime as dt

import pandas as pd
import streamlit as st

import industrials_statarb as sa

st.set_page_config(page_title="Industrials stat-arb research", layout="wide")
st.title("Industrials stat-arb research")
st.caption("Mean-reversion pairs within sub-sectors on daily adjusted closes, with optional trend following. "
           "Research tool only: it places no orders, and backtest results are not a forecast.")

GROUPS = list(sa.SUBSECTORS)
SPEED_SETS = {
    "Fast (8/32)": ((8, 32),),
    "Medium (16/64)": ((16, 64),),
    "Slow (32/128)": ((32, 128),),
    "Blend of all three": ((8, 32), (16, 64), (32, 128)),
}

# ------------------------------------------------------------ sidebar
with st.sidebar.form("settings"):
    st.header("Data")
    source = st.selectbox("Price source", ["Yahoo Finance", "Upload CSV", "Synthetic demo"])
    groups_sel = st.multiselect("Sub-sectors", GROUPS, default=GROUPS)
    start = st.date_input("History start", dt.date(2014, 1, 1), min_value=dt.date(2000, 1, 1))
    upload = st.file_uploader("CSV (date index, one close column per ticker)", type="csv")
    min_dv = st.number_input("Min median daily volume, $m", 0.0, 5000.0, 10.0, 5.0)
    holdout = st.number_input("Holdout: hide the last N trading days", 0, 504, 0, 21)

    st.header("Pair screen")
    same_group = st.checkbox("Only pair names that share a sub-sector", True)
    top_k = st.slider("Max pairs to trade", 1, 60, 20)
    max_per_name = st.slider("Max pairs per name", 1, 10, 3)
    min_corr = st.slider("Min return correlation", 0.0, 0.95, 0.4, 0.05)
    adf_max = st.slider("Max residual ADF t-stat (-3.04 = 90%, -3.34 = 95%)", -6.0, -1.5, -3.04, 0.02)
    hl_range = st.slider("Half-life range, days", 1, 120, (1, 90))
    persist = st.checkbox("Persistence filter (pair must hold up on both halves of the window)", True,
                          help="The spread must be stationary on the first half of the screening window, and "
                               "with the first-half hedge ratio and mean frozen it must stay stationary and "
                               "centred on the second half. Removes most pairs that pass by luck.")

    st.header("Pair rules")
    spread_mode = st.radio("Spread parameters", ["Fixed from the training window", "Rolling, re-estimated daily"],
                           help="Fixed freezes each pair's hedge ratio, mean and sigma from the window it was "
                                "screened on, so z returns to zero only if the spread itself converges.")
    lookback = st.slider("Z-score lookback, days (rolling only)", 20, 252, 60, 5)
    entry = st.slider("Entry |z|", 1.0, 3.5, 2.0, 0.1)
    exit_ = st.slider("Exit |z| (0 = spread crosses its mean)", 0.0, 1.5, 0.5, 0.1)
    use_zstop = st.checkbox("Z-score stop", False)
    stop = st.slider("Stop |z|", 2.5, 8.0, 4.0, 0.25)
    use_tstop = st.checkbox("Time stop", True)
    tstop_unit = st.radio("Time stop measured in", ["Days", "Multiples of each pair's half-life"], index=1)
    max_hold = st.slider("Time stop, days", 5, 252, 60, 5)
    hold_hl = st.slider("Time stop, half-lives", 1.0, 8.0, 4.0, 0.5)
    use_macd = st.checkbox("Require MACD turn on the z-score before entry", False)

    st.header("Trend rules")
    use_trend = st.checkbox("Include trend following", False)
    speed_name = st.selectbox("EMA speeds", list(SPEED_SETS), index=3)
    long_only = st.checkbox("Long or flat only (no shorts)", False)
    trend_vol = st.slider("Trend target vol", 0.05, 0.60, 0.20, 0.05)

    st.header("Portfolio and costs")
    mix = st.slider("Weight on pairs (rest on trend)", 0.0, 1.0, 0.5, 0.05)
    use_vt = st.checkbox("Add a vol-targeted book", True)
    book_vol = st.slider("Book target vol", 0.02, 0.40, 0.08, 0.01)
    cost_bps = st.slider("Cost per trade, bps of notional", 0, 50, 5)
    carry_bps = st.slider("Annual borrow cost on shorts, bps", 0, 1000, 50, 25)
    lag = st.radio("Execution lag, days", [1, 2], horizontal=True)

    st.header("Walk-forward")
    train = st.slider("Training window, trading days", 250, 1260, 504, 2)
    test = st.slider("Test window, trading days", 21, 252, 63, 21)
    tune = st.checkbox("Also re-pick entry (and rolling lookback) each window", True)
    st.form_submit_button("Run", type="primary")

tickers = sorted({t for g in groups_sel for t in sa.SUBSECTORS[g]})
pair_groups = ({t: [g for g in sa.TICKER_TO_GROUPS.get(t, []) if g in groups_sel] for t in tickers}
               if same_group else None)


# ------------------------------------------------------------ cached work
@st.cache_data(ttl=6 * 3600, show_spinner="Downloading prices...")
def load(source, tickers, start, csv_bytes):
    if source == "Yahoo Finance":
        return sa.fetch_yahoo(list(tickers), start)
    if source == "Upload CSV":
        import io
        raw = pd.read_csv(io.BytesIO(csv_bytes), index_col=0, parse_dates=True)
        close, _ = sa.clean(raw.apply(pd.to_numeric, errors="coerce"))
        close = close[close.index >= pd.Timestamp(start)]
        bench = close[sa.BROAD] if sa.BROAD in close.columns else None
        return close.drop(columns=sa.BROAD, errors="ignore"), None, [], bench
    close, dvol, failed, bench = sa.synthetic(list(tickers))
    keep = close.index >= pd.Timestamp(start)
    return close[keep], dvol[keep], failed, bench[keep]


@st.cache_data(show_spinner="Running in-sample backtest...")
def run_in_sample(px, dvol, min_dv, screen_kw, pair_kw, trend_kw, bt_kw):
    names = sa.eligible(px.dropna(axis=1), dvol, min_dv)          # full-history names for pairs
    picks = sa.screen_pairs(px[names], **screen_kw) if len(names) > 1 else pd.DataFrame()
    w_mr = sa.mr_weights(px, picks, **pair_kw) if len(picks) else pd.DataFrame(0.0, index=px.index, columns=px.columns)
    tnames = [c for c in px.columns if dvol is None or min_dv <= 0 or dvol[c].median() >= min_dv]
    w_tr = sa.trend_weights(px[tnames], **trend_kw).reindex(columns=px.columns).fillna(0)
    return picks, w_mr, w_tr, len(names)


@st.cache_data(show_spinner="Screening current pairs...")
def run_signals(px, dvol, min_dv, train, screen_kw, pair_kw):
    """Screen on the latest training window and report where each pair stands after the last close."""
    w = px.iloc[-train:]
    names = sa.eligible(w, dvol, min_dv)
    picks = sa.screen_pairs(w[names], **screen_kw) if len(names) > 1 else pd.DataFrame()
    return sa.current_signals(w, picks, **pair_kw) if len(picks) else pd.DataFrame()


@st.cache_data(show_spinner="Running walk-forward test (this is the slow one)...")
def run_walk_forward(px, dvol, min_dv, train, test, screen_kw, pair_kw, grid, trend_kw, speed_sets, bt_kw, use_trend):
    w_mr, log_mr = sa.walk_forward_pairs(px, dvol, train, test, min_dv, screen_kw, pair_kw, grid, bt_kw)
    if not use_trend:
        return w_mr, log_mr, None, None
    w_tr, log_tr = sa.walk_forward_trend(px, dvol, train, test, min_dv, speed_sets, trend_kw, bt_kw)
    return w_mr, log_mr, w_tr, log_tr


def make_books(px, w_mr, w_tr):
    """The books to report. With trend off this is pairs only, plus a vol-targeted copy if asked."""
    if use_trend:
        w = mix * w_mr + (1 - mix) * w_tr
        return {"Pairs": w_mr, "Trend": w_tr, "Combined": sa.vol_target(px, w, book_vol) if use_vt else w}
    books = {"Pairs": w_mr}
    if use_vt:
        books["Pairs (vol-targeted)"] = sa.vol_target(px, w_mr, book_vol)
    return books


def report(px, books, since=None):
    """Stats table plus equity and drawdown charts for a dict of {name: weights}."""
    rows, eq = {}, {}
    bret = None if bench is None else bench.reindex(px.index).pct_change(fill_method=None)
    for name, w in books.items():
        bt = sa.backtest(px, w, **bt_kw)
        if since is not None:
            bt = bt[bt.index >= since]
        rows[name] = sa.stats(bt)
        if bret is not None:
            rows[name][f"Corr to {sa.BROAD}"] = float(bt["ret"].corr(bret.reindex(bt.index))) if bt["ret"].std() > 0 else 0.0
        eq[name] = (1 + bt["ret"]).cumprod()
    table = pd.DataFrame(rows).T
    st.dataframe(table.style.format({"Sharpe": "{:.2f}", "Ann. return": "{:.1%}", "Vol": "{:.1%}",
                                     "Max DD": "{:.1%}", "Avg gross": "{:.2f}", "Turnover/yr": "{:.1f}",
                                     "Days": "{:.0f}", f"Corr to {sa.BROAD}": "{:.2f}"}))
    eq = pd.DataFrame(eq)
    st.markdown("**Growth of 1, net of costs**")
    st.line_chart(eq)
    st.markdown("**Drawdown**")
    st.line_chart(eq / eq.cummax() - 1)
    return table, eq


# ------------------------------------------------------------ load data
if source == "Upload CSV" and upload is None:
    st.info("Upload a CSV in the sidebar, then press Run.")
    st.stop()
if len(tickers) < 2 and source != "Upload CSV":
    st.error("Pick at least one sub-sector.")
    st.stop()
try:
    px_all, dvol_all, failed, bench = load(source, tuple(tickers), start, upload.getvalue() if upload else None)
except Exception as e:                                             # network, rate limit, bad file
    st.error(f"Could not load prices from {source}: {e}")
    st.stop()
if px_all.shape[1] < 2 or len(px_all) < train + test:
    st.error(f"Not enough data: {px_all.shape[1]} names and {len(px_all)} days loaded; "
             f"the walk-forward needs at least {train + test} days.")
    st.stop()

px = px_all.iloc[:-holdout] if holdout else px_all
dvol = None if dvol_all is None else dvol_all.reindex(px.index)
min_dv_usd = min_dv * 1e6 if dvol is not None else 0.0

screen_kw = dict(min_corr=min_corr, adf_max=adf_max, hl_range=tuple(hl_range), top_k=top_k,
                 max_per_name=max_per_name, groups=pair_groups, persist=persist)
fixed = spread_mode.startswith("Fixed")
by_half_life = use_tstop and tstop_unit != "Days"
pair_kw = dict(lookback=lookback, entry=entry, exit_=exit_, use_macd=use_macd, fixed=fixed, slots=top_k,
               stop=stop if use_zstop else None,
               max_hold=max_hold if use_tstop and not by_half_life else None,
               hold_half_lives=hold_hl if by_half_life else None)
trend_kw = dict(target_vol=trend_vol, long_only=long_only)
bt_kw = dict(cost_bps=cost_bps, short_carry_bps=carry_bps, lag=lag)
grid = None
if tune:
    grid = {"entry": [1.5, 2.0, 2.5]} if fixed else {"lookback": [30, 60, 90], "entry": [1.5, 2.0, 2.5]}

if source == "Synthetic demo":
    st.warning("Synthetic demo data: made-up prices with cointegration built in. Results mean nothing "
               "about real markets; this source only shows that the app works.")

tab_sum, tab_data, tab_screen, tab_is, tab_wf = st.tabs(
    ["Summary", "Data", "Pair screen", "Backtest (in-sample)", "Walk-forward (out-of-sample)"])

# ------------------------------------------------------------ data tab
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Names loaded", px.shape[1])
    c2.metric("First day", str(px.index[0].date()))
    c3.metric("Last day used", str(px.index[-1].date()))
    if holdout:
        st.info(f"The last {holdout} trading days ({px_all.index[-holdout].date()} onward) are held out and "
                "not used anywhere. Set the holdout to 0 once your rules are final to see how they did.")
    if failed:
        st.warning("No usable data (not found, or under 250 days of history) for: " + ", ".join(failed))
    cover = pd.DataFrame({"Sub-sectors": [" / ".join(sa.TICKER_TO_GROUPS.get(t, [])) for t in px.columns],
                          "First day": px.apply(lambda s: s.first_valid_index().date()),
                          "Days": px.notna().sum(), "Last price": px.ffill().iloc[-1]}, index=px.columns)
    if dvol is not None:
        cover["Median daily volume, $m"] = (dvol.median() / 1e6).round(1)
    st.dataframe(cover)
    st.caption("The ticker list is today's coverage universe. Names that were acquired, delisted or went "
               "bankrupt are missing, which flatters any backtest that reaches back several years.")

# ------------------------------------------------------------ in-sample
picks, w_mr, w_tr, n_full = run_in_sample(px, dvol, min_dv_usd, screen_kw, pair_kw,
                                          dict(trend_kw, speeds=SPEED_SETS[speed_name]), bt_kw)

with tab_screen:
    scope = "that share a sub-sector" if same_group else "across all sub-sectors"
    st.markdown(f"Screened pairs {scope} among the **{n_full}** names with a full history and enough volume; "
                f"**{len(picks)}** selected. Ranked by the residual's ADF t-stat (more negative is stronger).")
    if len(picks):
        st.dataframe(picks.drop(columns="mu").style.format({"beta": "{:.2f}", "adf_t": "{:.2f}", "half_life": "{:.1f}",
                                                            "corr": "{:.2f}", "sigma": "{:.1%}", "t_1st_half": "{:.2f}",
                                                            "t_2nd_half": "{:.2f}", "shift": "{:+.2f}"}))
        st.caption("sigma is one standard deviation of the spread over the screening window: the rough size of a "
                   "1-point move in z when parameters are fixed. t_1st_half and t_2nd_half are the persistence "
                   "checks on each half of the window, and shift is how far the spread's mean moved between "
                   f"halves, in sigmas. The filter needs t_1st_half at or below {sa.PERSIST_T_FIRST}, t_2nd_half at or "
                   f"below {sa.PERSIST_T_SECOND} and shift within ±{sa.PERSIST_SHIFT}.")
        labels = [f"{y}/{x}" for y, x in zip(picks["y"], picks["x"])]
        chosen = st.selectbox("Inspect a pair", labels)
        y, x = chosen.split("/")
        row = picks[(picks["y"] == y) & (picks["x"] == x)].iloc[0]
        z, _ = (sa.pair_zscore(px[y], px[x], beta=row["beta"], mu=row["mu"], sigma=row["sigma"]) if fixed
                else sa.pair_zscore(px[y], px[x], lookback))
        st.markdown(f"**{'Fixed-parameter' if fixed else 'Rolling'} z-score of {chosen}**")
        st.line_chart(pd.DataFrame({"z": z, "entry": entry, "-entry": -entry}).dropna())
    else:
        st.info("No pairs passed. Loosen the screen (higher ADF t-stat, lower correlation), switch off the "
                "persistence filter, or add sub-sectors.")
    st.caption("This screen uses the whole sample, so these pairs were chosen with hindsight. "
               "The walk-forward tab re-runs the screen using only past data.")

with tab_is:
    st.warning("In-sample: pairs were selected and traded on the same data, and the settings are whatever "
               "you chose after looking at results. Treat these numbers as an upper bound."
               + (" With fixed parameters each spread's mean and sigma are also taken from the whole sample, "
                  "so every trade here knows where the spread ends up. Only the walk-forward tab is a fair test."
                  if fixed else ""))
    is_table, _ = report(px, make_books(px, w_mr, w_tr))

# ------------------------------------------------------------ walk-forward
wf_mr, log_mr, wf_tr, log_tr = run_walk_forward(
    px, dvol, min_dv_usd, train, test, screen_kw, pair_kw, grid, trend_kw,
    list(SPEED_SETS.values()), bt_kw, use_trend)
wf_books = make_books(px, wf_mr, wf_tr)

with tab_wf:
    oos_start = px.index[train]
    st.markdown(f"Every **{test}** trading days the app re-screens pairs and re-picks parameters on the previous "
                f"**{train}** days, then trades them on the next {test}. "
                + ("Each pair's hedge ratio, mean and sigma are frozen from its training window. " if fixed else "")
                + f"Out-of-sample period starts "
                f"**{oos_start.date()}**; costs are charged when the pair set changes.")
    wf_table, wf_eq = report(px, wf_books, since=oos_start)
    cmp = pd.DataFrame({"In-sample Sharpe": is_table["Sharpe"], "Walk-forward Sharpe": wf_table["Sharpe"]})
    st.markdown("**In-sample vs walk-forward**")
    st.dataframe(cmp.style.format("{:.2f}"))
    st.caption("A large drop from the first column to the second is the size of the overfitting.")

    st.markdown("**Current target weights** (what the walk-forward book holds after the last close)")
    last = pd.DataFrame({k: w.iloc[-1] for k, w in wf_books.items()})
    last = last[(last.abs() > 1e-4).any(axis=1)].sort_values(last.columns[-1])
    if len(last):
        st.dataframe(last.style.format("{:+.2%}"))
    else:
        st.write("Flat.")
    st.caption("To forward-test, record these weights each day and compare the paper P&L with this page later.")

    with st.expander("Pairs chosen in each window"):
        st.dataframe(log_mr)
    if use_trend:
        with st.expander("Trend speeds chosen in each window"):
            st.dataframe(log_tr)
    out = pd.DataFrame({k: sa.backtest(px, w, **bt_kw)["ret"] for k, w in wf_books.items()})
    st.download_button("Download daily out-of-sample returns (CSV)",
                       out[out.index >= oos_start].to_csv().encode(), "walk_forward_returns.csv", "text/csv")
    st.download_button("Download current target weights (CSV)", last.to_csv().encode(),
                       "target_weights.csv", "text/csv")

# ------------------------------------------------------------ summary
sig = run_signals(px, dvol, min_dv_usd, train, screen_kw, pair_kw)

with tab_sum:
    asof = px.index[-1].date()
    st.markdown(f"Pairs screened on the last **{train}** trading days through **{asof}**"
                + (" with the persistence filter on" if persist else "")
                + f". Signals use the sidebar rules: enter at |z| ≥ {entry:g}, exit at |z| ≤ {exit_:g}"
                + (f", z-stop at {stop:g}" if use_zstop else ", no z-stop")
                + (f", time stop {hold_hl:g} half-lives." if by_half_life else
                   f", time stop {max_hold} days." if use_tstop else ", no time stop."))
    if not len(sig):
        st.info("No pairs passed the screen on the latest window, so there are no signals. Loosen the screen "
                "or switch off the persistence filter to see more candidates.")
    else:
        active = sig[sig["status"].isin(["New entry", "Open"])]
        c1, c2, c3 = st.columns(3)
        c1.metric("Pairs screened", len(sig))
        c2.metric("New entries at the last close", int((sig["status"] == "New entry").sum()))
        c3.metric("Open trades", int((sig["status"] == "Open").sum()))

        def show(df):
            view = pd.DataFrame({
                "Pair": df["pair"], "Sub-sector": df["group"], "Status": df["status"], "Trade": df["trade"],
                "Long leg": [f"{t} {w:.0%}" if t else "" for t, w in zip(df["long"], df["long_wt"])],
                "Short leg": [f"{t} {w:.0%}" if t else "" for t, w in zip(df["short"], df["short_wt"])],
                "z now": df["z"], "Entered": df["entered"], "Days held": df["days_held"],
                "Days to time stop": df["days_to_time_stop"], "Half-life": df["half_life"],
                "Hedge ratio": df["hedge_ratio"]})
            st.dataframe(view.style.format({"z now": "{:+.2f}", "Days held": "{:.0f}", "Days to time stop": "{:.0f}",
                                            "Half-life": "{:.1f}", "Hedge ratio": "{:.2f}"}, na_rep=""),
                         hide_index=True)

        st.markdown("**Pairs with a live signal**")
        if len(active):
            show(active)
        else:
            st.write("None. No screened pair is past its entry level and tradable right now.")
        st.caption("How to read a trade: the Long and Short legs show each ticker and its share of the money in "
                   "that pair, set by the hedge ratio. A pair written A/B with z below the negative entry level is "
                   "Long A / Short B; above the positive entry level it is Short A / Long B. "
                   f"Each pair is sized at 1/{top_k} of the book. \"New entry\" opened at the last close; "
                   "\"Open\" was opened earlier and has not yet hit an exit.")
        st.caption("Check earnings dates on both legs before acting: a spread that moved on one company's "
                   "results is the kind least likely to revert. This app does not look up earnings dates.")
        with st.expander(f"All {len(sig)} screened pairs"):
            show(sig)
        last_screen = log_mr["test_start"].iloc[-1] if len(log_mr) else None
        st.caption("This tab re-screens as of the last close and uses the sidebar entry level. The walk-forward "
                   f"tab's current weights come from its last scheduled re-screen ({last_screen}) and its "
                   "per-window entry level, so the two can differ.")
