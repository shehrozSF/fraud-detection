"""Loading, memory reduction, and the pseudo-account id.

The IEEE-CIS files are ~1.9 GB as float64. `reduce_mem_usage` gets that to
roughly 550 MB, which is the difference between this project fitting in RAM
and not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------
def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to the smallest dtype that holds their range.

    Floats go to float32, never float16: float16 carries ~3 decimal digits,
    which silently corrupts TransactionAmt.
    """
    start = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue

        c_min, c_max = s.min(), s.max()
        if pd.isna(c_min):                      # all-NaN column
            continue

        if pd.api.types.is_integer_dtype(s):
            for dt in (np.int8, np.int16, np.int32, np.int64):
                if c_min >= np.iinfo(dt).min and c_max <= np.iinfo(dt).max:
                    df[col] = s.astype(dt)
                    break
        else:
            if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                df[col] = s.astype(np.float32)

    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        pct = 100 * (start - end) / max(start, 1e-9)
        print(f"  memory {start:,.0f} MB -> {end:,.0f} MB  ({pct:.0f}% saved)")
    return df


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
_DOWNLOAD_HINT = """
Could not find the IEEE-CIS files in {d}

Download them with the Kaggle CLI (accept the competition rules in the
browser once, first):

    pip install kaggle
    kaggle competitions download -c ieee-fraud-detection -p data/raw
    cd data/raw && unzip ieee-fraud-detection.zip

Or run any notebook with SMOKE = True to use the synthetic generator.
"""


def load_raw(nrows=None, verbose: bool = True) -> pd.DataFrame:
    """Load train_transaction left-joined to train_identity on TransactionID.

    The identity table covers only ~24% of transactions. We left-join, so the
    *absence* of an identity record is itself information: do not drop those
    rows, and do not fill the resulting NaNs with 0 (see notebook 01).
    """
    tx_path = C.RAW_DIR / "train_transaction.csv"
    id_path = C.RAW_DIR / "train_identity.csv"

    if not tx_path.exists():
        raise FileNotFoundError(_DOWNLOAD_HINT.format(d=C.RAW_DIR))

    if verbose:
        print("loading train_transaction ...")
    tx = pd.read_csv(tx_path, nrows=nrows)
    tx = reduce_mem_usage(tx, verbose)

    if id_path.exists():
        if verbose:
            print("loading train_identity ...")
        idf = reduce_mem_usage(pd.read_csv(id_path), verbose=False)
        tx = tx.merge(idf, on=C.ID_COL, how="left")

    tx = tx.sort_values(C.TIME_COL).reset_index(drop=True)
    if verbose:
        print(f"  shape {tx.shape},  fraud rate {tx[C.TARGET].mean():.4%}")
    return tx


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive readable time from TransactionDT.

    `day` is the model-facing quantity (float days from the data's own
    origin). `datetime` exists only so plots have readable axes.
    """
    dt = df[C.TIME_COL].astype("float64")
    stamp = pd.to_datetime(C.DT_ORIGIN) + pd.to_timedelta(dt, unit="s")

    # Built as one frame and concatenated once. Assigning four columns one at a
    # time to a 434-column frame triggers a copy per insert and pandas warns
    # about fragmentation -- harmless but slow, and it buries real warnings.
    new = pd.DataFrame({
        "day": dt / C.SECONDS_PER_DAY,
        "datetime": stamp,
        "hour": stamp.dt.hour.astype("int8"),
        "dayofweek": stamp.dt.dayofweek.astype("int8"),
    }, index=df.index)
    return pd.concat([df, new], axis=1)


def build_uid(df: pd.DataFrame) -> pd.Series:
    """A pseudo-account id: which rows plausibly belong to the same card.

    D1 is documented as "days since the card began transacting", so
    `day - D1` is that card's *registration day* and is constant for a given
    card. Pairing it with card1 + addr1 gives a stable per-account key.

    It is a heuristic, not a ground-truth id -- it collides for cards sharing
    a card1/addr1 that registered the same day. Two uses:
      1. the grouping variable that exposes identity leakage (notebook 02);
      2. later, the key every velocity feature aggregates over (day 9-15).
    """
    day = df[C.TIME_COL].astype("float64") / C.SECONDS_PER_DAY
    reg_day = np.floor(day - df["D1"].astype("float64"))

    card = df["card1"].astype("float64").fillna(-1).astype("int64").astype(str)
    addr = df["addr1"].astype("float64").fillna(-1).astype("int64").astype(str)
    reg = pd.Series(reg_day, index=df.index).fillna(-999).astype("int64").astype(str)

    return card.str.cat([addr, reg], sep="_").rename("uid")


# --------------------------------------------------------------------------
# Synthetic fallback -- run the pipeline before the download finishes
# --------------------------------------------------------------------------
def _solve_intercept(base_logit: np.ndarray, target_rate: float,
                     lo: float = -14.0, hi: float = 8.0) -> float:
    """Bisect for the intercept that makes the mean event probability hit target.

    Cheaper than hand-tuning a constant every time another term is added, and
    it keeps prevalence pinned while the other coefficients move.
    """
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(-(mid + base_logit)))
        if p.mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_synthetic(n_rows: int = 60_000, n_cards: int = 20_000,
                   target_rate: float = 0.035, seed: int = C.SEED) -> pd.DataFrame:
    """A small stand-in carrying the two structures notebook 02 needs:

      * account identity is predictive (some cards are compromised), so a
        random split can leak that identity across the fold boundary;
      * the fraud pattern drifts, so a future block is genuinely harder.

    NOT a substitute for the real data for any modelling conclusion.
    """
    rng = np.random.default_rng(seed)

    # Account activity follows a long tail: a few heavily used cards and many
    # that appear once or twice. Drawing card1 uniformly instead gives every
    # account ~n_rows/n_cards transactions, which makes almost every
    # validation row's account already seen under *any* split -- and the
    # leakage contrast in notebook 02 collapses to nothing.
    w = 1.0 / np.arange(1, n_cards + 1) ** 0.8
    card1 = 1000 + rng.choice(n_cards, n_rows, p=w / w.sum())
    addr1 = (card1 % 300 + 100).astype(float)
    day = np.sort(rng.uniform(0, 182, n_rows))

    # D1 is whole days since the card first transacted, as in the real data.
    # It has to be integer-valued: build_uid recovers the registration day as
    # floor(day - D1), and with a continuous D1 that lands on reg_day or
    # reg_day-1 depending on the fractional part, shattering one account into
    # several uids.
    reg_day = -(card1 % 90).astype(float)            # deterministic per card
    d1 = np.floor(day) - reg_day

    amt = np.exp(rng.normal(3.9, 1.05, n_rows))
    log_amt = np.log1p(amt)
    z_amt = (log_amt - log_amt.mean()) / log_amt.std()
    product = rng.choice(list("WCHRS"), n_rows, p=[.74, .12, .06, .05, .03])

    # Per-card compromise propensity -> identity carries signal. Stable over
    # time, so it stays learnable under an honest split and leaks under a
    # shuffled one.
    card_risk = rng.beta(0.5, 12.0, n_cards)[card1 - 1000]

    # Drift: the amount effect decays across the timeline rather than
    # inverting. Because z_amt is standardised, changing its coefficient
    # shifts *which* transactions are fraudulent without moving overall
    # prevalence -- which is what tactic drift actually looks like. An
    # inversion would make a model trained on the past worse than random on
    # the future, which is a pathology, not drift.
    amt_coef = 0.60 - 0.35 * (day / day.max())

    base = (7.0 * card_risk + amt_coef * z_amt
            + 0.45 * (product == "C") + rng.normal(0, 0.6, n_rows))
    logit = _solve_intercept(base, target_rate) + base
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit)))

    df = pd.DataFrame({
        C.ID_COL: np.arange(n_rows),
        C.TARGET: y.astype("int8"),
        C.TIME_COL: (day * C.SECONDS_PER_DAY).astype("int64"),
        C.AMOUNT_COL: amt.astype("float32"),
        "ProductCD": product,
        "card1": card1.astype("int32"),
        "card4": rng.choice(["visa", "mastercard", "amex", "discover"], n_rows),
        "card6": rng.choice(["debit", "credit"], n_rows, p=[.75, .25]),
        "addr1": addr1.astype("float32"),
        "dist1": np.where(rng.random(n_rows) < .6, np.nan,
                          rng.exponential(25, n_rows)).astype("float32"),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com", None],
                                    n_rows),
        "D1": d1.astype("float32"),
        "C1": rng.poisson(2, n_rows).astype("float32"),
        "C13": rng.poisson(11, n_rows).astype("float32"),
        "V1": rng.normal(size=n_rows).astype("float32"),
        "V2": rng.normal(size=n_rows).astype("float32"),
    })
    return df.sort_values(C.TIME_COL).reset_index(drop=True)


def load(smoke: bool = False, nrows=None) -> pd.DataFrame:
    """Single entry point used by every notebook."""
    df = make_synthetic() if smoke else load_raw(nrows=nrows)
    df = add_time_columns(df)
    df = pd.concat([df, build_uid(df)], axis=1)
    return df.copy()          # single defragmenting copy before anything else
