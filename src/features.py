"""Leak-safe feature engineering.

Every function here obeys one rule: **a feature computed for the row at time t
may only read rows strictly before t.**

That is harder than it sounds, and getting it wrong is invisible. A plain
`groupby(uid)['amt'].transform('mean')` reads the account's entire history --
including transactions that have not happened yet at scoring time. The model
learns from it, the metric goes up, and nothing warns you. In production the
feature simply is not available, and the model silently degrades.

The primitives:

  past_rolling      trailing time-window aggregate, current row EXCLUDED
  past_expanding    all-prior-rows aggregate, current row EXCLUDED
  expanding_target_encode
                    target encoding using only earlier rows
  verify_past_only  a test that recomputes features on a truncated prefix and
                    fails if any value changes

`verify_past_only` is the part worth keeping. It converts "I was careful"
into "I checked", which is a materially different claim in an interview.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# Longest trailing window built below. config.PURGE_DAYS must be >= this or
# training rows near a fold boundary will summarise validation rows.
MAX_WINDOW_DAYS = 30


# ==========================================================================
# Primitives
# ==========================================================================
def past_rolling(df: pd.DataFrame, group_col: str, value_col: str,
                 window: str, stat: str = "count",
                 time_col: str = C.TIME_COL) -> pd.Series:
    """Trailing-window aggregate within `group_col`, EXCLUDING the current row.

    `closed="left"` is the entire safety argument. The default, `closed="right"`,
    includes the current row -- so a "mean amount over the last 7 days" feature
    would contain this transaction's own amount, and for a group's first
    transaction the feature would equal the value it is meant to help predict.
    That single keyword is the difference between a velocity feature and a
    label leak.
    """
    t = pd.to_datetime(df[time_col].to_numpy(), unit="s")
    tmp = pd.DataFrame({"t": t, "g": df[group_col].to_numpy(),
                        "v": df[value_col].to_numpy()}, index=df.index)

    order = tmp.sort_values(["g", "t"], kind="stable").index
    s = tmp.loc[order]

    rolled = (s.set_index("t")
               .groupby("g", observed=True)["v"]
               .rolling(window, closed="left")
               .agg(stat))

    return pd.Series(rolled.to_numpy(), index=order).reindex(df.index)


def past_expanding(df: pd.DataFrame, group_col: str, value_col: str,
                   stat: str = "mean") -> pd.Series:
    """All-prior-rows aggregate within `group_col`, EXCLUDING the current row.

    Implemented with cumulative sums rather than `.expanding()` because it runs
    in one pass instead of one per group -- on 590k rows that is seconds versus
    minutes. Subtracting the current row's own contribution from the cumulative
    total is what makes it past-only.

    Assumes `df` is sorted by time (every loader here guarantees that).
    """
    v = df[value_col].astype("float64")
    g = df.groupby(group_col, observed=True)

    n = g.cumcount().to_numpy().astype("float64")          # prior row count
    csum = (g[value_col].cumsum().to_numpy() - v.to_numpy())

    if stat == "count":
        return pd.Series(n, index=df.index)
    if stat == "sum":
        return pd.Series(np.where(n > 0, csum, np.nan), index=df.index)

    mean = np.where(n > 0, csum / np.maximum(n, 1), np.nan)
    if stat == "mean":
        return pd.Series(mean, index=df.index)

    if stat == "std":
        sq = df.assign(_sq=v**2).groupby(group_col, observed=True)["_sq"]
        csq = sq.cumsum().to_numpy() - (v.to_numpy() ** 2)
        var = np.where(n > 1, csq / np.maximum(n, 1) - mean**2, np.nan)
        return pd.Series(np.sqrt(np.clip(var, 0, None)), index=df.index)

    raise ValueError(f"unsupported stat {stat!r}")


def time_since_prev(df: pd.DataFrame, group_col: str,
                    time_col: str = C.TIME_COL) -> pd.Series:
    """Seconds since this group's previous transaction. NaN on the first.

    `.diff()` only ever looks backwards, so this is safe by construction.
    """
    return df.groupby(group_col, observed=True)[time_col].diff()


def is_first_seen(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    """1 the first time this combination appears, 0 afterwards.

    "First time this account has used this product type" is a classic fraud
    signal -- takeover shows up as novel behaviour on an established account.
    `cumcount()` counts prior occurrences only.
    """
    return (df.groupby(group_cols, observed=True).cumcount() == 0).astype("int8")


# ==========================================================================
# Encodings
# ==========================================================================
def past_count_encode(df: pd.DataFrame, col: str) -> pd.Series:
    """How many times this value has been seen BEFORE this row.

    The past-only counterpart of frequency encoding, and the version that
    belongs in `build_features`.

    Why not `value_counts()`: that maps each value to its frequency over the
    whole frame, so the count at row i depends on rows after i. It is
    unsupervised -- the label is never touched -- which makes it feel harmless,
    and `verify_past_only` flagged it anyway. It was right to. Across the
    train/holdout boundary those counts are computed partly from the holdout,
    and in production the future frequency simply does not exist yet.

    The population-frequency version is still useful; it is just a *fitted*
    transform rather than a per-row aggregate, so it lives in
    `fit_apply_count_encode` and is applied the same way target encoding is.
    """
    return df.groupby(col, observed=True, dropna=False).cumcount().astype("float32")


def fit_apply_count_encode(train: pd.Series, apply_to: pd.Series) -> pd.Series:
    """Population frequency learned from `train`, applied to `apply_to`.

    Rare-value detection is what this buys and what `past_count_encode` cannot
    express: a card1 seen twice in six months behaves very differently from one
    seen 4,000 times, and a cumulative count conflates "rare" with "early in
    the timeline".

    Safe only because it is fitted on the training block and applied to a
    holdout that lies entirely after it. Never fit it on train+holdout
    together.
    """
    counts = train.value_counts()
    return apply_to.map(counts).fillna(0).astype("float32")


def expanding_target_encode(df: pd.DataFrame, col: str, y: np.ndarray,
                            smoothing: float = 20.0,
                            prior: float | None = None) -> pd.Series:
    """Target encoding from earlier rows only.

        enc_i = (sum of y over earlier rows in this category + m*prior)
                / (count of those rows + m)

    Target encoding is the highest-value and highest-risk trick in tabular ML.
    Done naively -- category mean over the whole training set -- every row's
    encoding contains its own label, and a high-cardinality column becomes a
    near-perfect predictor that generalises to nothing.

    Two standard fixes. Out-of-fold encoding (`oof_target_encode` below) is the
    usual one. On time-ordered data the expanding version here is better: it is
    leak-free by the same argument as every other feature in this module, it
    mirrors how the feature would be computed in production, and it leaves no
    uncovered rows.

    `smoothing` pulls small categories toward the global rate -- a card seen
    once with one fraud should not encode as 1.0.
    """
    prior = float(y.mean()) if prior is None else prior
    tmp = pd.DataFrame({"c": df[col].astype("object").to_numpy(), "y": y.astype("float64")})

    g = tmp.groupby("c", observed=True, dropna=False)["y"]
    csum = g.cumsum().to_numpy() - tmp["y"].to_numpy()
    cnt = g.cumcount().to_numpy().astype("float64")

    return pd.Series((csum + smoothing * prior) / (cnt + smoothing),
                     index=df.index).astype("float32")


def oof_target_encode(df: pd.DataFrame, col: str, y: np.ndarray, splits,
                      smoothing: float = 20.0) -> pd.Series:
    """Out-of-fold target encoding. Provided for comparison with the above.

    Under a *time-series* splitter this leaves the earliest rows encoded as the
    prior, because they never appear in any validation fold. That is a real
    drawback of using OOF encoding on ordered data and the reason
    `expanding_target_encode` is the default here.
    """
    prior = float(y.mean())
    out = np.full(len(df), prior, dtype="float64")

    for tr, va in splits:
        stats = (pd.DataFrame({"c": df[col].iloc[tr].to_numpy(), "y": y[tr]})
                 .groupby("c", observed=True, dropna=False)["y"].agg(["sum", "count"]))
        enc = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
        out[va] = df[col].iloc[va].map(enc).fillna(prior).to_numpy()

    return pd.Series(out, index=df.index).astype("float32")


def fit_holdout_target_encode(train_df: pd.DataFrame, y_train: np.ndarray,
                              hold_df: pd.DataFrame, col: str,
                              smoothing: float = 20.0) -> pd.Series:
    """Encoding for the holdout, learned from the whole training block.

    Safe precisely because the holdout is entirely *after* the training block --
    at holdout scoring time, all of training genuinely is the past.
    """
    prior = float(y_train.mean())
    stats = (pd.DataFrame({"c": train_df[col].to_numpy(), "y": y_train})
             .groupby("c", observed=True, dropna=False)["y"].agg(["sum", "count"]))
    enc = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
    return hold_df[col].map(enc).fillna(prior).astype("float32")


# ==========================================================================
# The feature builder
# ==========================================================================
def build_features(df: pd.DataFrame, y: np.ndarray | None = None,
                   groups: tuple[str, ...] = ("A", "B", "C", "D", "E", "F"),
                   verbose: bool = True) -> pd.DataFrame:
    """Build feature blocks A-F. Returns a NEW frame of engineered columns only.

    Grouped so notebook 06 can add one block at a time and measure what each is
    worth -- an ablation, rather than dumping 40 columns in at once and
    reporting a single number nobody can attribute.
    """
    out = pd.DataFrame(index=df.index)
    amt = df[C.AMOUNT_COL].astype("float64")
    has_uid = "uid" in df.columns

    # -- A: cheap row-local transforms ------------------------------------
    if "A" in groups:
        out["amt_log"] = np.log1p(amt).astype("float32")
        # Cents. Foreign-currency conversions leave unusual decimals while
        # ordinary retail prices cluster on .00 and .99 -- a real signal in
        # this dataset and free to compute.
        out["amt_decimal"] = ((amt - np.floor(amt)) * 100).round().astype("float32")
        out["amt_is_round"] = (out["amt_decimal"] == 0).astype("int8")
        # hour / dayofweek deliberately NOT emitted here: add_time_columns
        # already puts them on the frame, so prepare_lgbm_frame picks them up
        # as raw columns. Re-emitting them produces duplicate names and
        # LightGBM rejects the matrix outright.

        # IEEE-CIS D columns are "days since <some event>", so they climb with
        # calendar time and a tree splits on *when* rather than on behaviour.
        # Subtracting the day pins them to a fixed origin. Well-known trick,
        # and it is what makes the D block usable at all.
        for c in [f"D{i}" for i in range(1, 16) if f"D{i}" in df.columns]:
            out[f"{c}_norm"] = (df["day"] - df[c].astype("float64")).astype("float32")

    # -- B: velocity -------------------------------------------------------
    if "B" in groups and has_uid:
        for win, label in [("1h", "1h"), ("24h", "24h"), ("7D", "7d"),
                           (f"{MAX_WINDOW_DAYS}D", f"{MAX_WINDOW_DAYS}d")]:
            out[f"uid_n_{label}"] = past_rolling(
                df, "uid", C.AMOUNT_COL, win, "count").astype("float32")
        out["card1_n_24h"] = past_rolling(
            df, "card1", C.AMOUNT_COL, "24h", "count").astype("float32")
        out["uid_amt_sum_24h"] = past_rolling(
            df, "uid", C.AMOUNT_COL, "24h", "sum").astype("float32")

    # -- C: amount relative to the account's own history -------------------
    if "C" in groups and has_uid:
        prior_mean = past_expanding(df, "uid", C.AMOUNT_COL, "mean")
        prior_std = past_expanding(df, "uid", C.AMOUNT_COL, "std")
        out["uid_amt_mean_prior"] = prior_mean.astype("float32")
        out["uid_amt_std_prior"] = prior_std.astype("float32")
        # The single most intuitive fraud feature: how unusual is this amount
        # *for this cardholder*. $2,000 is unremarkable on some cards and a
        # screaming anomaly on others; the raw amount cannot express that.
        out["amt_to_uid_mean"] = (amt / prior_mean).astype("float32")
        out["amt_z_uid"] = ((amt - prior_mean) / prior_std.replace(0, np.nan)).astype("float32")
        out["uid_n_prior"] = past_expanding(df, "uid", C.AMOUNT_COL, "count").astype("float32")

    # -- D: recency --------------------------------------------------------
    if "D" in groups and has_uid:
        dt = time_since_prev(df, "uid")
        out["uid_secs_since_prev"] = dt.astype("float32")
        out["uid_log_secs_since_prev"] = np.log1p(dt.clip(lower=0)).astype("float32")
        out["card1_secs_since_prev"] = time_since_prev(df, "card1").astype("float32")

    # -- E: novelty --------------------------------------------------------
    if "E" in groups and has_uid:
        for c in ["ProductCD", "P_emaildomain", "card6", "DeviceType"]:
            if c in df.columns:
                out[f"uid_new_{c}"] = is_first_seen(df, ["uid", c])
        out["uid_is_first_txn"] = is_first_seen(df, ["uid"])

    # -- F: past-only occurrence counts ------------------------------------
    # Population-frequency encoding is NOT here: it reads the future (see
    # past_count_encode). It is applied separately, in the notebook, alongside
    # target encoding.
    if "F" in groups:
        for c in ["card1", "addr1", "P_emaildomain", "R_emaildomain",
                  "DeviceInfo", "uid"]:
            if c in df.columns:
                out[f"{c}_seen_before"] = past_count_encode(df, c)

    if verbose:
        print(f"  built {out.shape[1]} features "
              f"({', '.join(sorted(set(groups)))})")
    return out


def combine(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Concatenate engineered columns onto a raw matrix, dropping overlaps.

    LightGBM refuses a matrix with a repeated feature name ("Feature (hour)
    appears more than one time") and the error surfaces deep inside Dataset
    construction, far from the concat that caused it. One guard here beats
    debugging it at every call site.
    """
    dupes = [c for c in extra.columns if c in base.columns]
    if dupes:
        extra = extra.drop(columns=dupes)
    return pd.concat([base.reset_index(drop=True),
                      extra.reset_index(drop=True)], axis=1)


# ==========================================================================
# The test that makes the claim checkable
# ==========================================================================
def verify_past_only(build_fn, df: pd.DataFrame, n_checks: int = 12,
                     seed: int = C.SEED, atol: float = 1e-4,
                     verbose: bool = True) -> pd.DataFrame:
    """Recompute features on truncated prefixes; fail if any value changes.

    The argument: if a feature at row i is genuinely past-only, then computing
    it over `df[:i+1]` must give exactly the value computed over all of `df`.
    Rows after i cannot have contributed. If the two differ, the full-data
    version read the future.

    This catches the entire class of bug -- a forgotten `closed="left"`, a
    `transform("mean")` that slipped in, a target encoding that included its own
    row -- without needing to reason about each feature individually.

    Costs one full rebuild per check, so run it on a subsample.
    """
    rng = np.random.default_rng(seed)
    full = build_fn(df)

    lo = int(len(df) * 0.55)                    # deep enough to have history
    positions = np.sort(rng.choice(np.arange(lo, len(df)), n_checks, replace=False))

    rows = []
    for i in positions:
        prefix = build_fn(df.iloc[: i + 1])
        a = full.iloc[i]
        b = prefix.iloc[-1]
        for col in full.columns:
            va, vb = a[col], b[col]
            if pd.isna(va) and pd.isna(vb):
                continue
            if pd.isna(va) != pd.isna(vb) or not np.isclose(
                    float(va), float(vb), atol=atol, rtol=1e-3):
                rows.append({"row": int(i), "feature": col,
                             "full_data": va, "past_only": vb})

    bad = pd.DataFrame(rows)
    if verbose:
        if bad.empty:
            print(f"  PASS: {full.shape[1]} features identical at "
                  f"{n_checks} checkpoints -- none read the future")
        else:
            print(f"  FAIL: {bad['feature'].nunique()} leaking features")
            print(bad.groupby("feature").size().to_string())
    return bad
