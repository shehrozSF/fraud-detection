"""Class-imbalance strategies, implemented rather than imported.

Three families, and they are not interchangeable:

  reweighting   change the *loss*. The data is untouched, so nothing
                implausible can be invented. `scale_pos_weight`.
  undersampling change the *data* by discarding majority rows. Cheap and
                fast; throws away real information.
  oversampling  change the *data* by adding minority rows. Duplication adds
                no information; SMOTE interpolates, which adds rows that may
                not correspond to any transaction that could occur.

Written out longhand instead of calling imbalanced-learn because the failure
mode of SMOTE on this dataset is visible in the arithmetic, and reading it is
more convincing than being told.

Every function here takes and returns a *training fold only*. Resampling a
validation fold changes the base rate you are measuring against and makes
every metric meaningless -- see `resample_the_wrong_way` for what that costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config as C


# --------------------------------------------------------------------------
# 1. Reweighting -- no data change
# --------------------------------------------------------------------------
def scale_pos_weight(y) -> float:
    """n_negative / n_positive, the standard LightGBM/XGBoost setting.

    Multiplies the gradient contribution of every positive by this factor, so
    the rare class stops being ignorable. Exactly analogous to a `pos_weight`
    in BCEWithLogitsLoss.

    Side effect worth knowing before you use the output: predicted
    probabilities are no longer calibrated to the true prior. The model is now
    estimating probabilities under a reweighted world in which fraud is 50% of
    traffic. `correct_prior` undoes it.
    """
    y = np.asarray(y)
    n_pos = max(int((y == 1).sum()), 1)
    return float((y == 0).sum() / n_pos)


# --------------------------------------------------------------------------
# 2. Undersampling
# --------------------------------------------------------------------------
def undersample(X: pd.DataFrame, y: np.ndarray, target_ratio: float = 0.1,
                seed: int = C.SEED):
    """Drop random majority rows until positives are `target_ratio` of the fold.

    target_ratio=0.1 keeps ten negatives per positive. Going all the way to
    0.5 (fully balanced) discards ~97% of the data here, which costs far more
    in variance than it gains in balance.

    Returns (X_res, y_res, beta) where beta is the fraction of negatives kept
    -- `correct_prior` needs it.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    n_neg_keep = int(len(pos_idx) * (1 - target_ratio) / target_ratio)
    n_neg_keep = min(n_neg_keep, len(neg_idx))
    keep_neg = rng.choice(neg_idx, size=n_neg_keep, replace=False)

    keep = np.sort(np.concatenate([pos_idx, keep_neg]))
    beta = n_neg_keep / max(len(neg_idx), 1)
    return X.iloc[keep], y[keep], beta


def oversample_naive(X: pd.DataFrame, y: np.ndarray, target_ratio: float = 0.1,
                     seed: int = C.SEED):
    """Duplicate random minority rows.

    Included as the control that isolates what SMOTE's *interpolation* buys.
    Duplication adds no information at all -- it only reweights, and less
    precisely than `scale_pos_weight` does. If SMOTE cannot beat this, its
    synthetic points are contributing nothing.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    n_pos_target = int(len(neg_idx) * target_ratio / (1 - target_ratio))
    n_extra = max(n_pos_target - len(pos_idx), 0)
    extra = rng.choice(pos_idx, size=n_extra, replace=True)

    idx = np.concatenate([np.arange(len(y)), extra])
    return X.iloc[idx], y[idx]


# --------------------------------------------------------------------------
# 3. SMOTE
# --------------------------------------------------------------------------
def smote(X: pd.DataFrame, y: np.ndarray, target_ratio: float = 0.1,
          k: int = 5, categorical: list[str] | None = None,
          seed: int = C.SEED, scale_for_knn: bool = True):
    """Synthetic Minority Over-sampling. Chawla et al., 2002.

    The algorithm, in full:
      1. take the minority rows;
      2. find each one's k nearest minority neighbours;
      3. to make a new point, pick a minority row i, pick one of its
         neighbours j, and step a random fraction of the way from i to j.

    Two decisions here that most tutorials get wrong, and both matter:

    `scale_for_knn` -- step 2 is a Euclidean distance, so an unscaled
    TransactionAmt (range ~0-32,000) dominates every other column and the
    "neighbours" are just transactions of similar size. Standardising first is
    not optional if the result is meant to mean anything.

    `categorical` -- interpolation is nonsense for a categorical. Halfway
    between card_type=1 (debit) and card_type=2 (credit) is 1.5, which is not
    a card type. This is the SMOTE-NC correction: categorical columns take the
    *modal* value among the neighbours instead of a blend. Vanilla SMOTE on
    label-encoded categoricals silently produces rows that could not exist,
    which is the single best explanation for why it underperforms here.

    Note what SMOTE cannot know: it interpolates in feature space with no
    notion of whether the result is a *plausible transaction*. A synthetic row
    can carry a card's 30-day average from one account and a merchant category
    from another.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    categorical = categorical or []

    num_cols = [c for c in X.columns if c not in categorical]
    pos_mask = y == 1
    X_pos = X.loc[pos_mask]

    n_neg = int((~pos_mask).sum())
    n_target = int(n_neg * target_ratio / (1 - target_ratio))
    n_new = max(n_target - len(X_pos), 0)
    if n_new == 0 or len(X_pos) <= k:
        return X, y

    # --- neighbour search on numeric columns only -------------------------
    A = X_pos[num_cols].to_numpy(dtype="float64")
    # SMOTE has no missing-value story; median-fill for the distance metric.
    med = np.nanmedian(A, axis=0)
    A = np.where(np.isnan(A), med, A)

    if scale_for_knn:
        mu, sd = A.mean(0), A.std(0)
        sd[sd == 0] = 1.0
        A_s = (A - mu) / sd
    else:
        A_s = A

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(A_s))).fit(A_s)
    _, nbr = nn.kneighbors(A_s)
    nbr = nbr[:, 1:]                                  # drop self

    # --- synthesise -------------------------------------------------------
    i = rng.integers(0, len(A), n_new)
    j = nbr[i, rng.integers(0, nbr.shape[1], n_new)]
    gap = rng.random((n_new, 1))
    new_num = A[i] + gap * (A[j] - A[i])

    new = pd.DataFrame(new_num, columns=num_cols)

    # --- categoricals: modal neighbour value, never a blend ---------------
    for c in categorical:
        vals = X_pos[c].to_numpy()
        cand = np.stack([vals[nbr[i, m]] for m in range(nbr.shape[1])], axis=1)
        modes = [pd.Series(row).mode() for row in cand]
        new[c] = [m.iloc[0] if len(m) else vals[i[r]] for r, m in enumerate(modes)]

    new = new[X.columns]
    for c in X.columns:                               # restore dtypes
        try:
            new[c] = new[c].astype(X[c].dtype)
        except (TypeError, ValueError):
            pass

    X_res = pd.concat([X, new], ignore_index=True)
    y_res = np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
    return X_res, y_res


# --------------------------------------------------------------------------
# 4. Undoing the damage to the probabilities
# --------------------------------------------------------------------------
def correct_prior(p, beta: float):
    """Map probabilities from a resampled world back to the real base rate.

        p_true = p / (p + (1 - p) / beta)

    where beta is the fraction of negatives retained. Undersampling to a 1:10
    ratio inflates every predicted probability by roughly 1/beta; without this
    correction, `expected_value_decision` multiplies dollars by a number that
    is an order of magnitude too large.

    Elkan (2001), "The Foundations of Cost-Sensitive Learning". Ranking is
    unchanged -- the transform is monotone -- so PR-AUC will not move at all
    and Brier will move a great deal. That contrast is the point.
    """
    p = np.asarray(p, dtype="float64")
    beta = max(float(beta), 1e-12)
    return p / (p + (1.0 - p) / beta)


# --------------------------------------------------------------------------
# 5. The mistake, quantified
# --------------------------------------------------------------------------
def resample_the_wrong_way(X: pd.DataFrame, y: np.ndarray, method="smote", **kw):
    """Resample the WHOLE dataset before splitting. Never do this.

    It is in this module because notebook 05 measures what it costs, in the
    same spirit as notebook 02 measuring the random split.

    Why it is catastrophic with SMOTE specifically: synthetic minority points
    are interpolated from real minority points. Split afterwards and a
    synthetic row built from validation-set neighbours lands in training. The
    model has effectively been shown the validation labels, and PR-AUC goes
    almost to 1.0 -- a result that has fooled a very large number of published
    fraud notebooks.
    """
    if method == "smote":
        return smote(X, y, **kw)
    if method == "oversample":
        return oversample_naive(X, y, **kw)
    raise ValueError(f"unknown method {method!r}")
