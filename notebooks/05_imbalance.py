# %% [markdown]
# # 05 - Class imbalance: a benchmark, not a ritual
#
# "Use SMOTE for imbalanced data" is repeated so often it rarely gets tested.
# This notebook tests it, against the tuned LightGBM from notebook 04, under
# the purged split.
#
# Four arms:
#
# | arm | changes | cost |
# |---|---|---|
# | none | nothing | — |
# | `scale_pos_weight` | the **loss** | probabilities lose calibration |
# | undersampling | the **data**, by deleting negatives | discards real information |
# | SMOTE | the **data**, by inventing positives | invents rows that may not be possible |
#
# Plus one control — naive duplication — which isolates what SMOTE's
# *interpolation* buys over plain reweighting. If SMOTE cannot beat
# duplication, its synthetic points contribute nothing.
#
# **The rule that makes or breaks this notebook: resample the training fold
# only.** Never the validation fold, never before the split. Section 4
# measures what breaking it costs.
#
# ## Prediction, written before running
#
# SMOTE will underperform `scale_pos_weight`. Two reasons, and they compound:
# interpolating between transactions in a mixed categorical/numeric space
# produces rows that could not occur; and gradient boosting with a weighted
# loss already handles imbalance without touching the data. Resampling is a
# workaround for algorithms that cannot be reweighted — GBDTs can be.

# %%
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config as C
from src import data as D
from src import evaluate as E
from src import imbalance as IMB
from src import models as M
from src import validation as V

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 220)

SMOKE = False          # data/raw is populated; set True to use the synthetic generator
TARGET_RATIO = 0.10          # resample to ~1 positive per 10 rows, not to 50/50

# %%
df = D.load(smoke=SMOKE)
y = df[C.TARGET].to_numpy()

train_idx = np.load(C.INTERIM_DIR / "train_idx.npy")
holdout_idx = np.load(C.INTERIM_DIR / "holdout_idx.npy")
train_df = df.iloc[train_idx].reset_index(drop=True)
hold_df = df.iloc[holdout_idx].reset_index(drop=True)
y_train, y_hold = y[train_idx], y[holdout_idx]

best_params = pd.read_json(C.INTERIM_DIR / "lgbm_best_params.json",
                           typ="series").to_dict()
best_params["num_leaves"] = int(best_params["num_leaves"])
best_params["min_child_samples"] = int(best_params["min_child_samples"])

X, cat_cols = M.prepare_lgbm_frame(train_df, max_cardinality=25)
X_hold = M.align_categories(X, M.prepare_lgbm_frame(hold_df)[0][X.columns], cat_cols)

print(f"train {len(X):,} rows, fraud {y_train.mean():.3%}")
print(f"1 positive per {1/y_train.mean():.0f} rows")

# %% [markdown]
# ## Why not resample to 50/50
#
# The reflex is to balance completely. Look at what that costs first.

# %%
n_pos, n_neg = int(y_train.sum()), int((y_train == 0).sum())
for ratio in (0.5, 0.25, 0.10, 0.05):
    keep = int(n_pos * (1 - ratio) / ratio)
    print(f"  ratio {ratio:>5.0%}  keeps {min(keep, n_neg):>8,} of {n_neg:,} negatives"
          f"   ({100*min(keep,n_neg)/n_neg:5.1f}% of the data)")

print(f"\nFully balancing discards most of the majority class. The variance that")
print(f"adds usually costs more than the balance buys, which is why this")
print(f"notebook uses {TARGET_RATIO:.0%} rather than 50%.")

# %% [markdown]
# ## The benchmark
#
# One function, four arms. The resampling happens **inside** the fold loop,
# after `tr`/`va` are separated — that placement is the entire methodological
# content of this notebook.

# %%
def run_arm(name: str, sampler=None, use_weight: bool = False, correct=False):
    rows, t0 = [], time.time()

    for k, (tr, va) in enumerate(V.PurgedTimeSeriesSplit().split(train_df)):
        X_tr, y_tr = X.iloc[tr], y_train[tr]
        X_va = M.align_categories(X_tr, X.iloc[va], cat_cols)
        y_va = y_train[va]

        beta = 1.0
        if sampler is not None:
            out = sampler(X_tr, y_tr)
            X_tr, y_tr = out[0], out[1]
            beta = out[2] if len(out) > 2 else 1.0
            X_tr = X_tr.reset_index(drop=True)

        over = dict(best_params)
        if use_weight:
            over["scale_pos_weight"] = IMB.scale_pos_weight(y_tr)

        model, _ = M.fit_lgbm_early_stopping(X_tr, y_tr, cat_cols, **over)
        p = model.predict_proba(X_va)[:, 1]
        if correct and beta < 1.0:
            p = IMB.correct_prior(p, beta)

        rec, _ = E.recall_at_fpr(y_va, p)
        rows.append({"arm": name, "fold": k,
                     "pr_auc": E.pr_auc(y_va, p),
                     "recall@1%fpr": rec,
                     "brier": E.brier(y_va, np.clip(p, 0, 1)),
                     "n_train": len(y_tr),
                     "train_fraud_rate": float(y_tr.mean())})

    out = pd.DataFrame(rows)
    print(f"  {name:26s} PR-AUC {out.pr_auc.mean():.4f} +/- {out.pr_auc.std():.4f}"
          f"   Brier {out.brier.mean():.4f}   ({time.time()-t0:.0f}s)")
    return out


arms = [
    run_arm("none"),
    run_arm("scale_pos_weight", use_weight=True),
    run_arm("undersample 1:10",
            sampler=lambda Xt, yt: IMB.undersample(Xt, yt, TARGET_RATIO)),
    run_arm("undersample + prior fix",
            sampler=lambda Xt, yt: IMB.undersample(Xt, yt, TARGET_RATIO),
            correct=True),
    run_arm("oversample (duplicate)",
            sampler=lambda Xt, yt: IMB.oversample_naive(Xt, yt, TARGET_RATIO)),
    run_arm("SMOTE",
            sampler=lambda Xt, yt: IMB.smote(Xt, yt, TARGET_RATIO,
                                             categorical=cat_cols)),
]
res = pd.concat(arms, ignore_index=True)

# %%
summary = (res.groupby("arm")
           .agg(pr_auc=("pr_auc", "mean"), sd=("pr_auc", "std"),
                recall=("recall@1%fpr", "mean"), brier=("brier", "mean"),
                rows=("n_train", "mean"))
           .sort_values("pr_auc", ascending=False)
           .round(4))
print(summary.to_string())

baseline = summary.loc["none", "pr_auc"]
sd = summary.loc["none", "sd"]
print(f"""
Bar for significance: one fold sd = {sd:.4f}
Anything within +/-{sd:.4f} of {baseline:.4f} is indistinguishable from doing nothing.
""")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
order = summary.index.tolist()

sns.barplot(data=res, y="arm", x="pr_auc", order=order, hue="arm", hue_order=order,
            legend=False, errorbar="sd", ax=axes[0], palette="Blues_r")
axes[0].axvline(baseline, ls="--", c="0.35", lw=1.2)
axes[0].axvspan(baseline - sd, baseline + sd, color="0.85", zorder=0)
axes[0].set_title("PR-AUC (shaded band = +/-1 fold sd of doing nothing)")

sns.barplot(data=res, y="arm", x="brier", order=order, hue="arm", hue_order=order,
            legend=False, errorbar="sd", ax=axes[1], palette="Oranges_r")
axes[1].set_title("Brier - lower is better\n(ranking metrics cannot see this)")

plt.tight_layout()
plt.savefig(C.FIG_DIR / "05_imbalance.png", dpi=140)
plt.show()

# %% [markdown]
# ## What resampling does to the probabilities
#
# Train on data that is 10% fraud and the model estimates probabilities for a
# world that is 10% fraud. Reality is ~3.5%. Every prediction is inflated by
# roughly the ratio of the two priors.
#
# PR-AUC will not notice — the distortion is monotone, so the *ranking* is
# untouched. Brier notices immediately. This is the cleanest available
# demonstration that ranking quality and probability quality are different
# things.

# %%
X_tr_u, y_tr_u, beta = IMB.undersample(X, y_train, TARGET_RATIO)
print(f"kept {beta:.1%} of negatives; fold fraud rate "
      f"{y_train.mean():.3%} -> {y_tr_u.mean():.3%}")

mu, _ = M.fit_lgbm_early_stopping(X_tr_u.reset_index(drop=True), y_tr_u,
                                  cat_cols, **best_params)
p_raw = mu.predict_proba(X_hold)[:, 1]
p_fix = IMB.correct_prior(p_raw, beta)

print(f"""
                     mean p    Brier     PR-AUC
  true rate          {y_hold.mean():.4f}        -          -
  uncorrected        {p_raw.mean():.4f}   {E.brier(y_hold, np.clip(p_raw,0,1)):.4f}   {E.pr_auc(y_hold, p_raw):.4f}
  prior-corrected    {p_fix.mean():.4f}   {E.brier(y_hold, np.clip(p_fix,0,1)):.4f}   {E.pr_auc(y_hold, p_fix):.4f}

PR-AUC identical to the last digit - the correction is monotone. Brier moves a
lot. If you only ever report ranking metrics, this entire failure is invisible.
""")

# %%
fig, ax = plt.subplots(figsize=(6, 5.5))
for lab, p in [("uncorrected", p_raw), ("prior-corrected", p_fix)]:
    r = E.reliability_table(y_hold, p, n_bins=10)
    ax.plot(r["mean_predicted"], r["observed"], "o-", lw=2, label=lab)
lim = max(p_raw.mean() * 2, 0.1)
ax.plot([0, lim], [0, lim], "--", c="0.5", lw=1, label="perfect")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed fraud rate")
ax.set_title("Prior correction after undersampling")
ax.legend()
plt.tight_layout()
plt.savefig(C.FIG_DIR / "05_prior_correction.png", dpi=130)
plt.show()

# %% [markdown]
# ## Section 4 - The mistake, measured
#
# The most common error in published fraud notebooks: **resample the whole
# dataset, then split.**
#
# With SMOTE it is catastrophic in a specific way. Synthetic positives are
# interpolated from real positives. Split afterwards and a synthetic row built
# from a validation-set neighbour ends up in training — the model has been
# shown a blurred copy of the validation labels. PR-AUC goes near 1.0 and the
# notebook looks brilliant.
#
# Run it once so you recognise the smell.

# %%
X_bad, y_bad = IMB.resample_the_wrong_way(X, y_train, method="smote",
                                          target_ratio=TARGET_RATIO,
                                          categorical=cat_cols)
X_bad = X_bad.reset_index(drop=True)

# A RANDOM split of already-resampled data — the full mistake, both halves.
rs = np.random.default_rng(C.SEED).permutation(len(y_bad))
cut = int(0.8 * len(rs))
tr_b, va_b = rs[:cut], rs[cut:]

mb, _ = M.fit_lgbm_early_stopping(X_bad.iloc[tr_b].reset_index(drop=True),
                                  y_bad[tr_b], cat_cols, **best_params)
pb = mb.predict_proba(M.align_categories(X_bad.iloc[tr_b], X_bad.iloc[va_b],
                                         cat_cols))[:, 1]

honest = summary.loc["SMOTE", "pr_auc"]
bad = E.pr_auc(y_bad[va_b], pb)
print(f"""
  SMOTE, resampled inside the fold, purged split : {honest:.4f}   <- real
  SMOTE, resampled before a random split         : {bad:.4f}   <- fiction

  overstatement: {100*(bad-honest)/honest:.0f}%

A fraud PR-AUC above ~0.9 on this dataset is almost always this bug.
""")

# %% [markdown]
# ## Holdout: the winner vs. doing nothing

# %%
final = {}
for label, weight in [("none", False), ("scale_pos_weight", True)]:
    over = dict(best_params)
    if weight:
        over["scale_pos_weight"] = IMB.scale_pos_weight(y_train)
    mdl, _ = M.fit_lgbm_early_stopping(X, y_train, cat_cols, **over)
    final[label] = mdl.predict_proba(X_hold)[:, 1]

n_days = hold_df["day"].max() - hold_df["day"].min()
for label, p in final.items():
    m = E.evaluate_all(y_hold, p, amount=hold_df[C.AMOUNT_COL], n_days=n_days)
    print(f"{label:20s} PR-AUC {m['pr_auc']:.4f}  recall@1%FPR "
          f"{m['recall_at_0.01fpr']:.3f}  Brier {m['brier']:.4f}  "
          f"alerts/day {m['alerts_per_day']:,.0f}")
    E.log_result(f"lgbm_{label}", "temporal_holdout", m, "notebook 05 imbalance")

for arm in arms:
    a = arm.iloc[0]["arm"]
    E.log_result(f"cv_{a}", "purged_cv",
                 {"pr_auc": arm.pr_auc.mean(), "pr_auc_sd": arm.pr_auc.std(),
                  "recall_at_0.01fpr": arm["recall@1%fpr"].mean(),
                  "brier": arm.brier.mean()},
                 "notebook 05 benchmark")

fig, ax = plt.subplots(figsize=(6.5, 5))
E.plot_pr_curves({k: (y_hold, v) for k, v in final.items()},
                 ax=ax, base_rate=float(y_hold.mean()))
plt.tight_layout()
plt.savefig(C.FIG_DIR / "05_final_pr.png", dpi=140)
plt.show()

# %% [markdown]
# ## Writing up the result
#
# Fill the blanks from `summary` above — the shape of the finding matters more
# than which arm won:
#
# > Benchmarked four imbalance strategies against a tuned LightGBM under a
# > purged expanding-window split. `scale_pos_weight` and doing nothing were
# > within one fold standard deviation of each other; SMOTE was [X] and cost
# > [Y]x the training time. Resampling also degraded Brier by [Z] because it
# > shifts the prior, which ranking metrics cannot detect — corrected with
# > Elkan's prior adjustment.
#
# **If SMOTE won on your run, say so.** The value here is the benchmark and
# the methodology, not the conclusion. An honest "the popular technique
# underperformed and here is why" and an honest "it helped, and here is the
# measurement" are both far stronger than an unexamined `SMOTE().fit_resample()`.
#
# ## Days 9-15 next: feature engineering
#
# Expect it to move PR-AUC by an order of magnitude more than anything in
# notebooks 04 and 05 did. Velocity features over `uid`, computed strictly
# from past windows — which is when `PURGE_DAYS` in `config.py` starts
# earning its place, and when it has to be raised to match your longest
# rolling window.
