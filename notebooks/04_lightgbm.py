# %% [markdown]
# # 04 - LightGBM
#
# The workhorse. Four questions, in order:
#
# 1. Does it beat logistic regression by more than fold-to-fold noise?
# 2. Should high-cardinality categoricals (card1, ~13k levels) be included?
# 3. What do the hyperparameters actually do, and which matter here?
# 4. Which features is it using — and can that importance be trusted?
#
# Question 1 has a bar attached: notebook 03 recorded the standard deviation
# of PR-AUC across folds. An improvement smaller than that is not an
# improvement, and saying so out loud is most of what separates a careful
# portfolio from a lucky one.
#
# ## Translating from deep learning
#
# | LightGBM | closest DL analogue |
# |---|---|
# | `n_estimators` | epochs — set high, let early stopping choose |
# | `learning_rate` | same trade: halve it, roughly double the trees. No schedule. |
# | `num_leaves` | width / capacity |
# | `min_child_samples` | the main regulariser, more like weight decay than dropout |
# | `colsample_bytree` | dropout, loosely — decorrelates trees |
# | `reg_lambda` | weight decay on leaf values |
#
# The one with no analogue: **there is no imputation step.** LightGBM learns a
# default direction for missing values at each split, which usually beats any
# median you would substitute. The notebook-03 pipeline needed `SimpleImputer`;
# this does not.

# %%
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config as C
from src import data as D
from src import evaluate as E
from src import models as M
from src import validation as V

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 200)

SMOKE = False          # data/raw is populated; set True to use the synthetic generator
N_SEARCH_TRIALS = 8         # drop to 8 on the real data for a first pass

# %%
df = D.load(smoke=SMOKE)
y = df[C.TARGET].to_numpy()

train_idx = np.load(C.INTERIM_DIR / "train_idx.npy")
holdout_idx = np.load(C.INTERIM_DIR / "holdout_idx.npy")

train_df = df.iloc[train_idx].reset_index(drop=True)
hold_df = df.iloc[holdout_idx].reset_index(drop=True)
y_train, y_hold = y[train_idx], y[holdout_idx]

print(f"train {len(train_df):,}  holdout {len(hold_df):,}")

# %% [markdown]
# ## Q2 first: do high-cardinality categoricals help?
#
# Answering this before tuning, because it changes the feature matrix
# everything else is tuned on.
#
# `card1` has thousands of levels. A tree can split on it directly, which is
# either useful generalisation ("this BIN range is risky") or pure
# memorisation ("card 3847 was fraudulent"). Notebook 02 established that the
# purged split is the instrument that can tell those apart — so just measure
# it.

# %%
X_all, cats_all = M.prepare_lgbm_frame(train_df)                     # keep everything
X_cap, cats_cap = M.prepare_lgbm_frame(train_df, max_cardinality=25)  # low-card only

print(f"all categoricals : {len(X_all.columns)} cols, {len(cats_all)} categorical")
print(f"capped at 25     : {len(X_cap.columns)} cols, {len(cats_cap)} categorical")
print(f"excluded         : {sorted(set(X_all.columns) - set(X_cap.columns))}")


def cv_score(X, cat_cols, label, frame=None, **overrides):
    """Purged CV. Returns per-fold metrics."""
    frame = train_df if frame is None else frame
    rows, t0 = [], time.time()

    for k, (tr, va) in enumerate(V.PurgedTimeSeriesSplit().split(frame)):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        X_va = M.align_categories(X_tr, X_va, cat_cols)

        model, n_trees = M.fit_lgbm_early_stopping(
            X_tr, y_train[tr], cat_cols, **overrides)
        p = model.predict_proba(X_va)[:, 1]

        rec, _ = E.recall_at_fpr(y_train[va], p)
        rows.append({"label": label, "fold": k,
                     "pr_auc": E.pr_auc(y_train[va], p),
                     "recall@1%fpr": rec, "n_trees": n_trees})
    out = pd.DataFrame(rows)
    print(f"  {label:24s} PR-AUC {out.pr_auc.mean():.4f} +/- {out.pr_auc.std():.4f}"
          f"   trees~{out.n_trees.mean():.0f}   ({time.time()-t0:.0f}s)")
    return out


if set(X_all.columns) == set(X_cap.columns):
    # No categorical exceeds the cap (always true under SMOTE=True, where the
    # synthetic frame has none). Nothing to compare -- skip the second fit
    # rather than paying for an identical run.
    print("\nno high-cardinality categoricals present; comparison is degenerate")
    cmp_cats = cv_score(X_all, cats_all, "all categoricals")
else:
    cmp_cats = pd.concat([
        cv_score(X_all, cats_all, "all categoricals"),
        cv_score(X_cap, cats_cap, "capped at 25 levels"),
    ], ignore_index=True)

# %% [markdown]
# Whichever wins, the *reason* is the interesting half. If keeping card1 helps
# under a purged split, the model is finding genuine issuer/BIN-level risk,
# not memorising — because the accounts it memorised in the training block
# largely are not the ones being scored in a later block. If it hurts, that is
# overfitting to identity and the cap stays.
#
# Record the decision and move on with the winner.

# %%
winner = cmp_cats.groupby("label")["pr_auc"].mean().idxmax()
print(f"using: {winner}")
X, cat_cols = (X_all, cats_all) if winner == "all categoricals" else (X_cap, cats_cap)

X_hold_full, _ = M.prepare_lgbm_frame(hold_df)
X_hold = M.align_categories(X, X_hold_full[X.columns], cat_cols)

# %% [markdown]
# ## Q1 - Does it beat the linear baseline?

# %%
base_cv = cv_score(X, cat_cols, "lgbm default")

prev = pd.read_csv(C.RESULTS_CSV)
lr_row = prev[prev.name == "logreg_balanced"].tail(1)
lr_pr = float(lr_row["pr_auc"].iloc[0]) if len(lr_row) else float("nan")

print(f"""
  logistic regression (holdout)  {lr_pr:.4f}
  lightgbm (CV mean)             {base_cv.pr_auc.mean():.4f}
  fold-to-fold sd                {base_cv.pr_auc.std():.4f}

An improvement smaller than that sd is noise, not a result.
""")

# %% [markdown]
# ## Q3 - Hyperparameter search
#
# Random search, not grid. With six parameters a grid spends most of its budget
# re-testing values of parameters that do not matter; random search covers each
# individual axis far better for the same number of fits (Bergstra & Bengio,
# 2012).
#
# `n_estimators` is deliberately *not* in the search space — early stopping
# picks it per fold, so searching it would be tuning a number that is already
# being chosen for you.

# %%
rng = np.random.default_rng(C.SEED)

SPACE = {
    "learning_rate":     lambda: float(10 ** rng.uniform(-2.0, -0.9)),   # .01 - .126
    "num_leaves":        lambda: int(rng.choice([15, 31, 63, 127, 255])),
    "min_child_samples": lambda: int(rng.choice([20, 50, 100, 200, 400])),
    "colsample_bytree":  lambda: float(rng.uniform(0.4, 0.95)),
    "subsample":         lambda: float(rng.uniform(0.6, 1.0)),
    "reg_lambda":        lambda: float(10 ** rng.uniform(-1, 2)),
}

trials = []
for t in range(N_SEARCH_TRIALS):
    params = {k: f() for k, f in SPACE.items()}
    r = cv_score(X, cat_cols, f"trial {t:02d}", **params)
    trials.append({**params, "pr_auc": r.pr_auc.mean(), "pr_auc_sd": r.pr_auc.std(),
                   "n_trees": r.n_trees.mean()})

search = pd.DataFrame(trials).sort_values("pr_auc", ascending=False)
print(search.head(10).to_string(index=False))

# %%
best_params = search.iloc[0].drop(["pr_auc", "pr_auc_sd", "n_trees"]).to_dict()
best_params["num_leaves"] = int(best_params["num_leaves"])
best_params["min_child_samples"] = int(best_params["min_child_samples"])

print("best:", {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in best_params.items()})
print(f"\nCV PR-AUC {search.iloc[0].pr_auc:.4f} vs default {base_cv.pr_auc.mean():.4f}")
print(f"gain {search.iloc[0].pr_auc - base_cv.pr_auc.mean():+.4f}, "
      f"fold sd {base_cv.pr_auc.std():.4f}")

# %% [markdown]
# Read that comparison honestly. Tuning a GBDT on tabular data typically buys
# a few percent — real, but an order of magnitude less than feature
# engineering will buy on day 9-15. If the gain here is inside one fold
# standard deviation, the correct conclusion is "defaults were fine", and
# reporting that is more credible than quoting a tuned number that is noise.

# %%
# Which parameters actually moved the score?
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, p in zip(axes.ravel(), SPACE):
    ax.scatter(search[p], search["pr_auc"], s=28, alpha=0.75)
    ax.set_xlabel(p)
    ax.set_ylabel("CV PR-AUC")
    if p in ("learning_rate", "reg_lambda"):
        ax.set_xscale("log")
fig.suptitle("Sensitivity: a flat cloud means the parameter does not matter here")
plt.tight_layout()
plt.savefig(C.FIG_DIR / "04_hparam_sensitivity.png", dpi=130)
plt.show()

# %% [markdown]
# ## Fit the final model and score the holdout

# %%
model, n_trees = M.fit_lgbm_early_stopping(X, y_train, cat_cols, **best_params)
p_lgb = model.predict_proba(X_hold)[:, 1]

n_days = hold_df["day"].max() - hold_df["day"].min()
m_lgb = E.evaluate_all(y_hold, p_lgb, amount=hold_df[C.AMOUNT_COL], n_days=n_days)

for k in ["pr_auc", "roc_auc", "recall_at_0.01fpr", "brier", "lift_over_base",
          "alerts_per_day", "min_total_cost"]:
    print(f"  {k:20s} {m_lgb[k]:,.4f}")

E.log_result("lgbm_tuned", "temporal_holdout", m_lgb,
             f"trees={n_trees}, {winner}, "
             + ",".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in best_params.items()))

# %% [markdown]
# ## Q4 - Feature importance, and why the default one misleads
#
# LightGBM's built-in importances are both biased:
#
# - **split count** favours high-cardinality features simply because they
#   offer more places to cut;
# - **gain** is measured on the *training* data, so a feature that overfits
#   looks important precisely because it overfits.
#
# Permutation importance on the *holdout* avoids both: shuffle one column,
# measure how much held-out PR-AUC drops. It answers the question you actually
# care about — does this feature help on data the model has not seen — at the
# cost of one scoring pass per feature.

# %%
gain = pd.Series(model.booster_.feature_importance("gain"),
                 index=X.columns).sort_values(ascending=False)


def permutation_importance(model, X_ref, y_ref, features, n_repeats=3, seed=C.SEED):
    rng = np.random.default_rng(seed)
    base = E.pr_auc(y_ref, model.predict_proba(X_ref)[:, 1])
    out = {}
    for f in features:
        dtype = X_ref[f].dtype          # .to_numpy() on a category column returns
        drops = []                      # object; assigning it back would drop the
        for _ in range(n_repeats):      # dtype and LightGBM rejects the schema.
            Xp = X_ref.copy()
            shuffled = rng.permutation(Xp[f].to_numpy())
            Xp[f] = pd.Series(shuffled, index=Xp.index).astype(dtype)
            drops.append(base - E.pr_auc(y_ref, model.predict_proba(Xp)[:, 1]))
        out[f] = float(np.mean(drops))
    return pd.Series(out).sort_values(ascending=False)


top = gain.head(20).index.tolist()
perm = permutation_importance(model, X_hold, y_hold, top)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
gain.head(20)[::-1].plot.barh(ax=axes[0], color="#4C72B0")
axes[0].set_title("Gain importance (train, biased)")
perm[::-1].plot.barh(ax=axes[1], color="#55A868")
axes[1].set_title("Permutation importance (holdout PR-AUC drop)")
axes[1].axvline(0, c="0.4", lw=1)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "04_importance.png", dpi=130)
plt.show()

print("\nFeatures ranked high by gain but ~0 by permutation are being used to")
print("fit training noise. Any with a NEGATIVE permutation score are actively")
print("hurting - the model does better when they are shuffled.")

# %%
fig, ax = plt.subplots(figsize=(6.5, 5))
E.plot_pr_curves({"lightgbm (tuned)": (y_hold, p_lgb)},
                 ax=ax, base_rate=float(y_hold.mean()))
plt.tight_layout()
plt.savefig(C.FIG_DIR / "04_lgbm_pr.png", dpi=140)
plt.show()

np.save(C.INTERIM_DIR / "p_lgbm_holdout.npy", p_lgb)
pd.Series(best_params).to_json(C.INTERIM_DIR / "lgbm_best_params.json")
print(f"saved predictions + params to {C.INTERIM_DIR}")

# %% [markdown]
# ## Carry into notebook 05
#
# - a tuned LightGBM and its CV standard deviation, which is the significance
#   bar for everything that follows;
# - a recorded, measured decision on high-cardinality categoricals;
# - the knowledge that `Brier` is bad and the model is over-confident — note
#   that no imbalance handling has been applied *yet*, so this is the
#   uncorrected starting point.
#
# Notebook 05 benchmarks the three imbalance strategies against exactly this
# configuration.
