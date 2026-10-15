"""Splitters.

This module is the whole point of days 1-3. Everything downstream is only as
trustworthy as the split it was measured under.

Three regimes, deliberately kept side by side so notebook 02 can contrast them:

  random_kfold          the wrong answer -- shuffles the timeline and lets the
                        same account appear on both sides of the boundary
  group_kfold_uid       removes account leakage but still trains on the future
  PurgedTimeSeriesSplit the right answer -- train on the past, validate on the
                        future, with a gap wide enough that no rolling feature
                        straddles the boundary
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from . import config as C


# --------------------------------------------------------------------------
# Final holdout
# --------------------------------------------------------------------------
def temporal_holdout(df: pd.DataFrame, val_fraction: float = C.VAL_FRACTION,
                     time_col: str = C.TIME_COL):
    """Split once at a point in time: everything before -> train, after -> val.

    This mirrors how the model would actually be deployed (fit on history,
    score tomorrow) and how the competition's own public/private test set was
    constructed. Touch the returned validation block as little as possible --
    it is the only estimate you have that is not contaminated by your own
    tuning decisions.
    """
    t = df[time_col].to_numpy()
    cut = np.quantile(t, 1.0 - val_fraction)
    train_idx = np.flatnonzero(t <= cut)
    val_idx = np.flatnonzero(t > cut)
    return train_idx, val_idx, cut


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------
class PurgedTimeSeriesSplit:
    """Expanding-window CV with a purge gap.

    Fold k validates on block k+1 at its FULL width, and trains on everything
    up to `gap_days` BEFORE that block starts:

        train:  [t_min, val_start - gap)
        val:    [val_start, val_end)

    Why the gap: a velocity feature such as "transactions by this card in the
    trailing 30 days" computed for a row at time t reads rows in [t-30d, t]. A
    training row sitting just before the boundary therefore summarises rows
    that fall inside validation. That is a real leak, invisible in the metric,
    and it inflates the score. Purging `gap_days` >= the longest rolling
    window removes it.

    The gap is taken out of the TRAINING tail, not the validation block. An
    earlier version of this class subtracted it from the validation side
    instead, which silently produced 0.3-day validation slivers at
    gap_days=30 -- and zero folds once the frame got shorter. Purging is
    meant to discard *training* observations whose feature windows overlap
    validation (Lopez de Prado, "Advances in Financial Machine Learning",
    ch. 7); it should never cost validation width.

    The real cost is that a large gap eats early folds entirely: with a
    145-day span and a 30-day gap, the first blocks have no usable training
    history left. Those folds are skipped and the count is reported, rather
    than being silently dropped.

    Expanding rather than sliding because fraud models in production are
    refit on all available history.
    """

    def __init__(self, n_splits: int = C.N_SPLITS, gap_days: float = C.PURGE_DAYS,
                 time_col: str = C.TIME_COL, min_train_days: float = 21.0,
                 warn: bool = True):
        self.n_splits = n_splits
        self.gap_days = gap_days
        self.time_col = time_col
        self.min_train_days = min_train_days
        self.warn = warn

    def _folds(self, X: pd.DataFrame):
        t = X[self.time_col].to_numpy().astype("float64")
        gap = self.gap_days * C.SECONDS_PER_DAY
        min_train = self.min_train_days * C.SECONDS_PER_DAY

        t_min, t_max = t.min(), t.max()
        edges = np.linspace(t_min, t_max, self.n_splits + 2)

        out = []
        for k in range(self.n_splits):
            val_start, val_end = edges[k + 1], edges[k + 2]
            train_end = val_start - gap

            if train_end - t_min < min_train:
                continue                      # no usable history left after purging

            train_idx = np.flatnonzero(t < train_end)
            val_idx = np.flatnonzero((t >= val_start) & (t < val_end))
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            out.append((train_idx, val_idx))

        if not out:
            span = (t_max - t_min) / C.SECONDS_PER_DAY
            raise ValueError(
                f"PurgedTimeSeriesSplit produced no usable folds: span {span:.0f}d, "
                f"n_splits={self.n_splits}, gap={self.gap_days}d, "
                f"min_train={self.min_train_days}d. The purge gap is too large "
                f"for this span -- lower n_splits, shorten the longest feature "
                f"window, or use a longer training block.")

        if self.warn and len(out) < self.n_splits:
            print(f"    [purged split] {len(out)}/{self.n_splits} folds usable "
                  f"(gap {self.gap_days}d consumed the earliest blocks)")
        return out

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self._folds(X)) if X is not None else self.n_splits

    def split(self, X: pd.DataFrame, y=None, groups=None):
        yield from self._folds(X)


def random_kfold(df: pd.DataFrame, n_splits: int = C.N_SPLITS, seed: int = C.SEED):
    """The split you must NOT use here. Kept so notebook 02 can measure it."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(df))


def group_kfold_uid(df: pd.DataFrame, n_splits: int = C.N_SPLITS,
                    group_col: str = "uid"):
    """Random folds, but an account never spans the boundary.

    Isolates *identity* leakage from *temporal* leakage: whatever gap remains
    between this and random_kfold is attributable to the model memorising
    accounts rather than learning fraud behaviour.
    """
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, groups=df[group_col]))


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------
def describe_split(df: pd.DataFrame, train_idx, val_idx,
                   group_col: str = "uid") -> dict:
    """Summarise one fold: size, fraud rate, date span, and overlap.

    `pct_val_rows_seen_uid` is the leakage thermometer. Under a temporal split
    it is naturally nonzero (real accounts recur over time, and that is
    legitimate). What matters is that under random_kfold it is near-total,
    meaning the model is scored mostly on accounts it has already been shown.
    """
    tr, va = df.iloc[train_idx], df.iloc[val_idx]
    tr_g, va_g = set(tr[group_col]), set(va[group_col])
    shared = tr_g & va_g

    return {
        "n_train": len(tr),
        "n_val": len(va),
        "fraud_train": tr[C.TARGET].mean(),
        "fraud_val": va[C.TARGET].mean(),
        "train_day_range": (round(tr["day"].min(), 1), round(tr["day"].max(), 1)),
        "val_day_range": (round(va["day"].min(), 1), round(va["day"].max(), 1)),
        "val_starts_after_train": va["day"].min() >= tr["day"].max(),
        "pct_val_uids_seen": 100 * len(shared) / max(len(va_g), 1),
        "pct_val_rows_seen_uid": 100 * va[group_col].isin(tr_g).mean(),
    }


def split_report(df: pd.DataFrame, splits, name: str) -> pd.DataFrame:
    """describe_split over every fold of a scheme, as a tidy frame."""
    rows = []
    for i, (tr, va) in enumerate(splits):
        d = describe_split(df, tr, va)
        d["scheme"], d["fold"] = name, i
        rows.append(d)
    cols = ["scheme", "fold", "n_train", "n_val", "fraud_train", "fraud_val",
            "train_day_range", "val_day_range", "val_starts_after_train",
            "pct_val_uids_seen", "pct_val_rows_seen_uid"]
    return pd.DataFrame(rows)[cols]
