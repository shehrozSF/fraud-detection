"""Models. Day 1-3 only needs the baselines; day 4+ appends to this file.

Two baselines, and both matter:

  majority / amount-only  the floor. If a model cannot beat "rank by
                          transaction amount", it has learned nothing.
  logistic regression     the real baseline. Every later number is quoted as
                          a delta against it.

Coming from deep learning, the sklearn idiom to internalise here is the
Pipeline: preprocessing is *part of the estimator*, so it is fit on the
training fold only and applied to the validation fold. Imputing or scaling
before you split -- the reflex from a preprocessed image dataset -- leaks
validation statistics into training and is the second most common way these
projects quietly break.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

# Columns that are identifiers or time bookkeeping, never model inputs.
LEAK_COLS = [C.ID_COL, C.TARGET, C.TIME_COL, "datetime", "uid", "day"]


def split_feature_types(df: pd.DataFrame, max_cardinality: int = 25):
    """Partition columns into (numeric, low-card categorical, dropped).

    High-cardinality categoricals (card1 has ~13k levels) are *dropped* for
    the linear baseline rather than one-hot encoded. One-hot on card1 would
    hand logistic regression a free per-card memory and inflate the baseline
    for exactly the wrong reason -- the identity leakage notebook 02 is about
    to demonstrate. They come back on day 9-15 as count and out-of-fold
    target encodings, which is the correct way to use them.
    """
    feats = [c for c in df.columns if c not in LEAK_COLS]

    numeric, low_card, dropped = [], [], []
    for c in feats:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
        elif s.nunique(dropna=True) <= max_cardinality:
            low_card.append(c)
        else:
            dropped.append(c)
    return numeric, low_card, dropped


def make_logreg_pipeline(numeric, categorical, C_reg: float = 0.1,
                         class_weight="balanced") -> Pipeline:
    """Median-impute + scale numerics, mode-impute + one-hot categoricals.

    `class_weight="balanced"` reweights the loss by inverse class frequency.
    It is the linear-model analogue of a weighted cross-entropy and the
    cheapest of the three imbalance strategies benchmarked on day 4-8 -- it
    changes the loss, not the data, so it cannot invent implausible rows the
    way SMOTE can.

    A missing value here is rarely "unknown at random": no identity record
    means the transaction came through a channel that did not collect one.
    `add_indicator=True` keeps that signal instead of burying it under a
    median.
    """
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50,
                                 sparse_output=False)),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop", verbose_feature_names_out=False)

    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            C=C_reg,
            class_weight=class_weight,
            max_iter=2000,
            solver="lbfgs",
            random_state=C.SEED,
        )),
    ])


def amount_only_baseline(df: pd.DataFrame) -> np.ndarray:
    """The floor: rank purely by transaction amount.

    Worth computing once. It is a real, if weak, fraud signal, and quoting a
    model's lift over *this* rather than over 0.5 ROC-AUC is the difference
    between a claim a risk team believes and one they discount.
    """
    return df[C.AMOUNT_COL].to_numpy(dtype="float64")


# ==========================================================================
# LightGBM  (day 4-8)
# ==========================================================================
# Translating from deep learning, the mapping is:
#
#   n_estimators      ~ epochs. Set it high and let early stopping choose.
#   learning_rate     ~ learning rate, and the same trade: halve it, roughly
#                       double n_estimators. There is no schedule.
#   num_leaves        ~ width/capacity. LightGBM grows leaf-wise rather than
#                       level-wise, so a large num_leaves overfits far faster
#                       than the equivalent max_depth suggests.
#   min_child_samples ~ the main regulariser. Minimum rows in a leaf; raising
#                       it is the most effective single defence on noisy
#                       tabular data.
#   colsample_bytree  ~ dropout, loosely. Decorrelates trees.
#   reg_lambda        ~ weight decay, on the leaf output values.
#
# What has no analogue: there is no imputation step. LightGBM learns a default
# direction for missing values at every split, which is usually better than
# any median you could substitute. The pipeline above needs SimpleImputer;
# this does not.

LGBM_BASE = dict(
    objective="binary",
    n_estimators=3000,          # an upper bound; early stopping picks the real number
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    colsample_bytree=0.7,
    subsample=0.8,
    subsample_freq=1,           # without this, subsample is silently ignored
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=C.SEED,
    verbose=-1,
)


def prepare_lgbm_frame(df: pd.DataFrame, max_cardinality: int | None = None):
    """Feature matrix for LightGBM. Returns (X, categorical_columns).

    Unlike the linear baseline this keeps high-cardinality categoricals such
    as card1. Trees can split on them natively, and whether that *helps* is an
    empirical question the purged split is there to answer honestly -- it is
    also the mechanism by which a model memorises accounts, so notebook 04
    tests it both ways rather than assuming.
    """
    feats = [c for c in df.columns if c not in LEAK_COLS]
    X = df[feats].copy()

    cat_cols = [c for c in X.columns
                if not pd.api.types.is_numeric_dtype(X[c])]
    if max_cardinality is not None:
        drop = [c for c in cat_cols if X[c].nunique(dropna=True) > max_cardinality]
        X = X.drop(columns=drop)
        cat_cols = [c for c in cat_cols if c not in drop]

    for c in cat_cols:
        X[c] = X[c].astype("category")
    return X, cat_cols


def align_categories(X_train: pd.DataFrame, X_other: pd.DataFrame,
                     cat_cols: list[str]) -> pd.DataFrame:
    """Force `X_other` onto the category levels learned from `X_train`.

    The leak this closes is small but real. Calling .astype("category") on the
    full frame before splitting builds the level set from validation rows too,
    so the model's encoding has seen the future. Setting categories from the
    training fold and mapping unseen levels to NaN is the honest version --
    and it also mirrors deployment, where a card type that never appeared in
    training genuinely is unknown.
    """
    X_other = X_other.copy()
    for c in cat_cols:
        X_other[c] = (X_other[c].astype("object")
                      .astype(pd.CategoricalDtype(categories=X_train[c].cat.categories)))
    return X_other


def make_lgbm(**overrides):
    """LGBMClassifier from LGBM_BASE with overrides applied."""
    import lightgbm as lgb
    return lgb.LGBMClassifier(**{**LGBM_BASE, **overrides})


def fit_lgbm_early_stopping(X_tr: pd.DataFrame, y_tr: np.ndarray,
                            cat_cols: list[str], inner_frac: float = 0.15,
                            stopping_rounds: int = 100, **overrides):
    """Fit with early stopping on an inner split carved from the END of train.

    This is the part that is easy to get wrong. The obvious move is to early-
    stop on the fold's validation set -- but then the number of trees was
    chosen using the same rows you are about to report a score on, and the
    score is optimistic. It is a small leak, and it is everywhere.

    The fix is a nested split: hold out the last `inner_frac` of the training
    fold *by time*, stop on that, report on the untouched outer validation
    fold. Taking it from the end rather than at random keeps the inner split
    temporally ordered too.

    Returns (fitted_model, best_iteration).
    """
    import lightgbm as lgb

    n = len(X_tr)
    cut = int(n * (1 - inner_frac))
    X_in, y_in = X_tr.iloc[:cut], y_tr[:cut]
    X_es, y_es = X_tr.iloc[cut:], y_tr[cut:]

    # Degenerate inner split (no positives to stop on) -- fall back to a fixed
    # tree count rather than silently early-stopping on noise.
    if y_es.sum() < 5 or len(X_es) < 50:
        model = make_lgbm(n_estimators=500, **overrides)
        model.fit(X_tr, y_tr, categorical_feature=cat_cols or "auto")
        return model, 500

    X_es = align_categories(X_in, X_es, cat_cols)

    model = make_lgbm(**overrides)
    model.fit(
        X_in, y_in,
        # LightGBM 4.7 deprecated the eval_set=[(X, y)] list form.
        eval_X=X_es, eval_y=y_es,
        eval_metric="average_precision",
        categorical_feature=cat_cols or "auto",
        callbacks=[lgb.early_stopping(stopping_rounds, verbose=False),
                   lgb.log_evaluation(0)],
    )
    return model, int(model.best_iteration_ or LGBM_BASE["n_estimators"])
