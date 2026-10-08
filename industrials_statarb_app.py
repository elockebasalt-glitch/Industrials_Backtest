"""
Industrials relative-value research app. No stationarity test anywhere: pairs are chosen by
distance and a variance-ratio check and traded around an adaptive mean with a time stop, and a
second sleeve trades each name's recent move against its sub-sector basket.
Needs industrials_statarb.py (version 3) and industrials_map.py beside it.

    streamlit run industrials_statarb_app.py
"""
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

import industrials_statarb as sa

st.set_page_config(page_title="Industrials relative-value research", layout="wide")
st.title("Industrials relative-value research")
st.caption("Distance pairs with an adaptive mean, plus sub-sector basket reversal, on daily adjusted closes. "
           "Research tool only: it places no orders, and backtest results are not a forecast.")

NEEDS = 3
if getattr(sa, "VERSION", 0) != NEEDS:
    st.error(f"industrials_statarb.py in this repo is version {getattr(sa, 'VERSION', 'older')}, but this app needs "
             f"version {NEEDS}. Upload the matching industrials_statarb.py, then use Manage app > Reboot app.")
    st.stop()

GROUPS = list(sa.SUBSECTORS)
MODE_LABELS = {
    "Rolling mean (1:1)": "rolling",
    "Exponentially weighted mean (1:1)": "ewma",
    "Kalman filter (adaptive hedge ratio)": "kalman",
    "Formation-window mean (classic distance method)": "formation",
}

# ------------------------------------------------------------ sidebar
with st.sidebar.form("settings"):
    st.header("Data")
    source = st.selectbox("Price source", ["Yahoo Finance", "Upload CSV", "Synthetic demo"])
    groups_sel = st.multiselect("Sub-sectors", GROUPS, default=GROUPS)
    start = st.date_input("History start", dt.date(2014, 1, 1), min_value=dt.date(2000, 1, 1))
    upload = st.file_uploader("Prices CSV (date index, one close column per ticker)", type="csv")
    min_dv = st.number_input("Min median daily volume, $m", 0.0, 5000.0, 10.0, 5.0)
    holdout = st.number_input("Holdout: hide the last N trading days", 0, 504, 0, 21)

    st.header("Pair selection")
    same_group = st.checkbox("Only pair names that share a sub-sector", True)
    top_k = st.slider("Max pairs to trade", 1, 60, 20)
    max_per_name = st.slider("Max pairs per name", 1, 10, 3)
    min_corr = st.slider("Min return correlation", 0.0, 0.95, 0.4, 0.05)
    dist_pct = st.slider("Keep the closest N% of candidate pairs by distance", 1, 100, 20)
    vr_max = st.slider("Max variance ratio (1 = random walk, lower = moves partly reverse)", 0.3, 1.5, 1.0, 0.05)
    vr_q = st.slider("Variance-ratio horizon, days", 2, 30, 10)
    persist = st.checkbox("Persistence filter (variance ratio must pass on both halves of the window)", True)

    st.header("Pair rules")
    mode_label = st.radio("Spread model", list(MODE_LABELS),
                          help="The first three let the mean move, so they do not need a stationary spread. "
                               "The last freezes the mean and sigma from the selection window.")
    lookback = st.slider("Lookback, days (mean and sigma; not used by the formation model)", 10, 126, 30, 5)
    entry = st.slider("Entry |z|", 1.0, 3.5, 2.0, 0.1)
    exit_ = st.slider("Exit |z| (0 = spread crosses its mean)", 0.0, 1.5, 0.5, 0.1)
    use_tstop = st.checkbox("Time stop", True)
    max_hold = st.slider("Time stop, days", 3, 126, 15, 1)
    use_zstop = st.checkbox("Z-score stop", False)
    stop = st.slider("Stop |z|", 2.5, 8.0, 4.0, 0.25)
    signal_src = st.radio("Signal measured on", ["Prices", "Uploaded valuation multiples, where available"])
    val_upload = st.file_uploader("Valuation CSV (date index, one column per ticker, e.g. forward P/E)", type="csv")

    st.header("Event filter")
    use_events = st.checkbox("No new trades in a name just after a jump or earnings date", True)
    jump_sigmas = st.slider("Jump size, in sigmas of the name's move against its peers", 2.0, 8.0, 4.0, 0.5)
    wait_days = st.slider("Days to wait after an event", 1, 20, 5)
    earn_upload = st.file_uploader("Earnings dates CSV (columns: ticker, date) - optional", type="csv")

    st.header("Basket reversal")
    use_basket = st.checkbox("Include basket reversal", True)
    bk_lookback = st.slider("Move measured over, days", 2, 30, 10)
    bk_hold = st.slider("Holding period, days", 1, 20, 5)

    st.header("Portfolio and costs")
    mix = st.slider("Weight on pairs (rest on basket reversal)", 0.0, 1.0, 0.5, 0.05)
    use_vt = st.checkbox("Scale the final book to a vol target", True)
    book_vol = st.slider("Book target vol", 0.02, 0.40, 0.08, 0.01)
    cost_bps = st.slider("Cost per trade, bps of notional", 0, 50, 5)
    carry_bps = st.slider("Annual borrow cost on shorts, bps", 0, 1000, 50, 25)
    lag = st.radio("Execution lag, days", [1, 2], horizontal=True)

    st.header("Walk-forward")
    train = st.slider("Training window, trading days", 250, 1260, 504, 2)
    test = st.slider("Test window, trading days", 21, 252, 63, 21)
    tune = st.checkbox("Re-pick the pair entry level and basket lookback each window", True)
    st.form_submit_button("Run", type="primary")

tickers = sorted({t for g in groups_sel for t in sa.SUBSECTORS[g]})
sel_groups = {g: sa.SUBSECTORS[g] for g in groups_sel}
pair_groups = ({t: [g for g in sa.TICKER_TO_GROUPS.get(t, []) if g in groups_sel] for t in tickers}
               if same_group else None)
mode = MODE_LABELS[mode_label]


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


@st.cache_data(show_spinner="Flagging jumps and earnings dates...")
def build_blocked(px, groups, jump_sigmas, wait_days, earn_bytes):
    extra = sa.parse_earnings(earn_bytes, px.index, px.columns) if earn_bytes else None
    ev = sa.event_mask(px, jump_sigmas, groups=groups, extra=extra)
    return ev, sa.recent_events(ev, wait_days)


def liquid(px, dvol, min_dv):
    return [c for c in px.columns if dvol is None or min_dv <= 0 or dvol[c].median() >= min_dv]


@st.cache_data(show_spinner="Running in-sample backtest...")
def run_in_sample(px, dvol, min_dv, screen_kw, pair_kw, sig, blocked, groups, basket_kw, use_basket):
    names = sa.eligible(px.dropna(axis=1), dvol, min_dv)           # full-history names for pairs
    picks = sa.screen_pairs(px[names], **screen_kw) if len(names) > 1 else pd.DataFrame()
    w_mr = (sa.mr_weights(px, picks, sig=sig, blocked=blocked, **pair_kw) if len(picks)
            else pd.DataFrame(0.0, index=px.index, columns=px.columns))
    w_bk = None
    if use_basket:
        w_bk = (sa.basket_reversal_weights(px[liquid(px, dvol, min_dv)], groups, blocked=blocked, **basket_kw)
                .reindex(columns=px.columns).fillna(0))
    return picks, w_mr, w_bk, len(names)


@st.cache_data(show_spinner="Running walk-forward test (this is the slow one)...")
def run_walk_forward(px, dvol, min_dv, train, test, screen_kw, pair_kw, grid, sig, blocked, groups,
                     lookbacks, basket_kw, bt_kw, use_basket):
    w_mr, log_mr = sa.walk_forward_pairs(px, dvol, train, test, min_dv, screen_kw, pair_kw, grid, bt_kw, sig, blocked)
    if not use_basket:
        return w_mr, log_mr, None, None
    w_bk, log_bk = sa.walk_forward_basket(px, dvol, train, test, min_dv, groups, lookbacks, basket_kw, bt_kw, blocked)
    return w_mr, log_mr, w_bk, log_bk


@st.cache_data(show_spinner="Screening current pairs...")
def run_signals(px, dvol, min_dv, train, screen_kw, pair_kw, sig, blocked):
    """Select pairs on the latest training window and report where each stands after the last close."""
    w = px.iloc[-train:]
    names = sa.eligible(w, dvol, min_dv)
    picks = sa.screen_pairs(w[names], **screen_kw) if len(names) > 1 else pd.DataFrame()
    return sa.current_signals(w, picks, sig=sig, blocked=blocked, **pair_kw) if len(picks) else pd.DataFrame()


def make_books(px, w_mr, w_bk):
    """The books to report: each sleeve on its own, then the final book (vol-targeted if asked)."""
    scale = (lambda w: sa.vol_target(px, w, book_vol)) if use_vt else (lambda w: w)
    if use_basket:
        return {"Pairs": w_mr, "Basket reversal": w_bk, "Combined": scale(mix * w_mr + (1 - mix) * w_bk)}
    books = {"Pairs": w_mr}
    if use_vt:
        books["Pairs (vol-targeted)"] = scale(w_mr)
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
    st.info("Upload a prices CSV in the sidebar, then press Run.")
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

sig, notes = None, []
if signal_src != "Prices":
    if val_upload is None:
        notes.append("Valuation signals were selected but no valuation CSV is uploaded, so prices are used.")
    elif mode == "formation":
        notes.append("Valuation signals are not used with the formation-window model, so prices are used.")
    else:
        try:
            sig = sa.parse_valuation(val_upload.getvalue(), px.index)
            sig = sig[[c for c in sig.columns if c in px.columns]]
        except Exception as e:
            notes.append(f"Could not read the valuation CSV ({e}); prices are used.")
events, blocked = None, None
if use_events:
    try:
        events, blocked = build_blocked(px, sel_groups, jump_sigmas, wait_days,
                                        earn_upload.getvalue() if earn_upload else None)
    except Exception as e:
        notes.append(f"Could not read the earnings CSV ({e}); only price jumps are used.")
        events, blocked = build_blocked(px, sel_groups, jump_sigmas, wait_days, None)

screen_kw = dict(min_corr=min_corr, dist_pct=float(dist_pct), vr_max=vr_max, vr_q=vr_q, top_k=top_k,
                 max_per_name=max_per_name, groups=pair_groups, persist=persist)
pair_kw = dict(mode=mode, lookback=lookback, entry=entry, exit_=exit_, slots=top_k,
               stop=stop if use_zstop else None, max_hold=max_hold if use_tstop else None)
basket_kw = dict(hold=bk_hold)
bt_kw = dict(cost_bps=cost_bps, short_carry_bps=carry_bps, lag=lag)
grid = {"entry": [1.5, 2.0, 2.5]} if tune else None
lookbacks = (5, 10, 20) if tune else (bk_lookback,)

if source == "Synthetic demo":
    st.warning("Synthetic demo data: made-up prices with pair relationships built in. Results mean nothing "
               "about real markets; this source only shows that the app works.")
for n_ in notes:
    st.warning(n_)

tab_sum, tab_data, tab_screen, tab_is, tab_wf = st.tabs(
    ["Summary", "Data", "Pair selection", "Backtest (in-sample)", "Walk-forward (out-of-sample)"])

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
    if events is not None:
        cover["Event days per year"] = (events.sum() / (len(px) / sa.ANN)).round(1)
    if sig is not None:
        cover["Valuation data, % of days"] = (sig.reindex(columns=px.columns).notna().mean() * 100).round(0)
    st.dataframe(cover)
    if sig is not None:
        st.caption(f"Valuation multiples loaded for {sig.shape[1]} of {px.shape[1]} names. A pair uses them only "
                   "when both legs have data on at least 80% of days in the window.")
    st.caption("The ticker list is today's coverage universe. Names that were acquired, delisted or went "
               "bankrupt are missing, which flatters any backtest that reaches back several years.")

# ------------------------------------------------------------ in-sample
picks, w_mr, w_bk, n_full = run_in_sample(px, dvol, min_dv_usd, screen_kw, pair_kw, sig, blocked, sel_groups,
                                          dict(basket_kw, lookback=bk_lookback), use_basket)

with tab_screen:
    scope = "that share a sub-sector" if same_group else "across all sub-sectors"
    st.markdown(f"Candidate pairs {scope} among the **{n_full}** names with a full history and enough volume; "
                f"**{len(picks)}** selected, closest first. No stationarity test is used.")
    if len(picks):
        st.dataframe(picks.drop(columns="mu").style.format(
            {"distance": "{:.1%}", "var_ratio": "{:.2f}", "vr_1st_half": "{:.2f}", "vr_2nd_half": "{:.2f}",
             "corr": "{:.2f}", "sigma": "{:.1%}"}))
        st.caption("distance is the typical gap between the two normalised price paths over the window. "
                   f"var_ratio compares {vr_q}-day spread moves with 1-day moves: 1 is a random walk, and below 1 "
                   "means moves tend to partly reverse. The persistence filter needs it to pass on each half. "
                   "sigma is one standard deviation of the log price ratio.")
        labels = [f"{y}/{x}" for y, x in zip(picks["y"], picks["x"])]
        chosen = st.selectbox("Inspect a pair", labels)
        y, x = chosen.split("/")
        row = picks[(picks["y"] == y) & (picks["x"] == x)].iloc[0]
        z, _ = sa.pair_zscore(px[y], px[x], mode, lookback, row["mu"], row["sigma"])
        st.markdown(f"**z-score of {chosen}, {mode_label.lower()}**")
        st.line_chart(pd.DataFrame({"z": z, "entry": entry, "-entry": -entry}).dropna())
    else:
        st.info("No pairs passed. Raise the distance percentage or the variance-ratio limit, lower the "
                "correlation floor, switch off the persistence filter, or add sub-sectors.")
    st.caption("This selection uses the whole sample, so these pairs were chosen with hindsight. "
               "The walk-forward tab re-runs it using only past data.")

with tab_is:
    st.warning("In-sample: pairs were selected and traded on the same data, and the settings are whatever "
               "you chose after looking at results. Treat these numbers as an upper bound."
               + (" The formation-window model also takes each spread's mean from the whole sample."
                  if mode == "formation" else ""))
    is_table, _ = report(px, make_books(px, w_mr, w_bk))

# ------------------------------------------------------------ walk-forward
wf_mr, log_mr, wf_bk, log_bk = run_walk_forward(
    px, dvol, min_dv_usd, train, test, screen_kw, pair_kw, grid, sig, blocked, sel_groups,
    lookbacks, basket_kw, bt_kw, use_basket)
wf_books = make_books(px, wf_mr, wf_bk)

with tab_wf:
    oos_start = px.index[train]
    st.markdown(f"Every **{test}** trading days the app re-selects pairs"
                + (" and re-picks parameters" if tune else "")
                + f" on the previous **{train}** days, then trades the next {test}. Out-of-sample period starts "
                f"**{oos_start.date()}**; costs are charged when the pair set changes.")
    wf_table, wf_eq = report(px, wf_books, since=oos_start)
    cmp = pd.DataFrame({"In-sample Sharpe": is_table["Sharpe"], "Walk-forward Sharpe": wf_table["Sharpe"]})
    st.markdown("**In-sample vs walk-forward**")
    st.dataframe(cmp.style.format("{:.2f}"))
    st.caption("A large drop from the first column to the second is the size of the overfitting. The basket "
               "book selects nothing, so its two numbers should be close.")

    st.markdown("**Current target weights** (what the walk-forward book holds after the last close)")
    last = pd.DataFrame({k: w.iloc[-1] for k, w in wf_books.items()})
    last = last[(last.abs() > 1e-4).any(axis=1)].sort_values(last.columns[-1])
    if len(last):
        st.dataframe(last.style.format("{:+.2%}"))
    else:
        st.write("Flat.")
    with st.expander("Pairs chosen in each window"):
        st.dataframe(log_mr)
    if use_basket:
        with st.expander("Basket lookback chosen in each window"):
            st.dataframe(log_bk)
    out = pd.DataFrame({k: sa.backtest(px, w, **bt_kw)["ret"] for k, w in wf_books.items()})
    st.download_button("Download daily out-of-sample returns (CSV)",
                       out[out.index >= oos_start].to_csv().encode(), "walk_forward_returns.csv", "text/csv")
    st.download_button("Download current target weights (CSV)", last.to_csv().encode(),
                       "target_weights.csv", "text/csv")

# ------------------------------------------------------------ summary
signals = run_signals(px, dvol, min_dv_usd, train, screen_kw, pair_kw, sig, blocked)

with tab_sum:
    asof = px.index[-1].date()
    st.subheader("Pairs")
    st.markdown(f"Pairs selected on the last **{train}** trading days through **{asof}**. Rules: {mode_label.lower()}"
                + ("" if mode == "formation" else f" over {lookback} days")
                + f", enter at |z| ≥ {entry:g}, exit at |z| ≤ {exit_:g}"
                + (f", time stop {max_hold} days" if use_tstop else ", no time stop")
                + (f", z-stop at {stop:g}" if use_zstop else "")
                + (f", no entries for {wait_days} days after an event." if use_events else "."))
    if not len(signals):
        st.info("No pairs passed the selection on the latest window, so there are no pair signals.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Pairs selected", len(signals))
        c2.metric("New entries at the last close", int((signals["status"] == "New entry").sum()))
        c3.metric("Open trades", int((signals["status"] == "Open").sum()))

        def show(df):
            view = pd.DataFrame({
                "Pair": df["pair"], "Sub-sector": df["group"], "Status": df["status"], "Trade": df["trade"],
                "Long leg": [f"{t} {w:.0%}" if t else "" for t, w in zip(df["long"], df["long_wt"])],
                "Short leg": [f"{t} {w:.0%}" if t else "" for t, w in zip(df["short"], df["short_wt"])],
                "z now": df["z"], "Entered": df["entered"], "Days held": df["days_held"],
                "Days to time stop": df["days_to_time_stop"], "Hedge ratio": df["hedge_ratio"],
                "Signal": df["signal"]})
            st.dataframe(view.style.format({"z now": "{:+.2f}", "Days held": "{:.0f}", "Days to time stop": "{:.0f}",
                                            "Hedge ratio": "{:.2f}"}, na_rep=""), hide_index=True)

        active = signals[signals["status"].isin(["New entry", "Open"])]
        st.markdown("**Pairs with a live signal**")
        if len(active):
            show(active)
        else:
            st.write("None. No selected pair is past its entry level and tradable right now.")
        st.caption("The Long and Short legs show each ticker and its share of the money in that pair. A pair written "
                   "A/B with z below the negative entry level is Long A / Short B; above the positive entry level it "
                   f"is Short A / Long B. Each pair is sized at 1/{top_k} of the pairs book. \"New entry\" opened at "
                   "the last close; \"Open\" was opened earlier and has not yet hit an exit.")
        with st.expander(f"All {len(signals)} selected pairs"):
            show(signals)

    if use_basket:
        st.subheader("Basket reversal")
        names_bk = liquid(px, dvol, min_dv_usd)
        w_now = sa.basket_reversal_weights(px[names_bk].iloc[-(bk_lookback + 130):], sel_groups, lookback=bk_lookback,
                                           hold=bk_hold, blocked=blocked).iloc[-1]
        move = sa.peer_residual_returns(px[names_bk].iloc[-(bk_lookback + 5):], sel_groups).iloc[-bk_lookback:].sum()
        pos = pd.DataFrame({"Weight": w_now, f"{bk_lookback}-day move vs peers": np.expm1(move)})
        pos = pos[pos["Weight"].abs() > 1e-5]
        pos.insert(0, "Sub-sectors", [" / ".join(g for g in sa.TICKER_TO_GROUPS.get(t, []) if g in groups_sel) for t in pos.index])
        fmt = {"Weight": "{:+.2%}", f"{bk_lookback}-day move vs peers": "{:+.1%}"}
        cl, cs = st.columns(2)
        cl.markdown("**Largest longs** (lagged their peers)")
        cl.dataframe(pos.sort_values("Weight", ascending=False).head(12).style.format(fmt))
        cs.markdown("**Largest shorts** (led their peers)")
        cs.dataframe(pos.sort_values("Weight").head(12).style.format(fmt))
        st.caption(f"Each name is traded against the other names in its sub-sector: long the ones that lagged over the "
                   f"last {bk_lookback} days, short the ones that led, dollar-neutral inside every sub-sector, each "
                   f"day's signal held {bk_hold} days. Weights are shares of the basket book "
                   f"({len(pos)} positions, gross {pos['Weight'].abs().sum():.0%}). Names with a recent event are left out.")

    st.caption("Check earnings dates on both sides before acting: a move driven by one company's results is the kind "
               "least likely to reverse. The event filter catches large jumps, and exact dates only if you upload them.")
    last_screen = log_mr["test_start"].iloc[-1] if len(log_mr) else None
    st.caption("This tab re-selects pairs as of the last close and uses the sidebar settings. The walk-forward tab's "
               f"current weights come from its last scheduled re-selection ({last_screen}) and its per-window "
               "settings, so the two can differ.")
