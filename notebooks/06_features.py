# %% [markdown]
# # 06 - Feature engineering
#
# The phase that should matter most. Notebooks 04 and 05 moved PR-AUC by
# fractions of a fold standard deviation; this one should move it by
# multiples.
#
# That asymmetry is the central fact about tabular ML, and it is the biggest
# adjustment coming from deep learning. In vision or NLP the network learns
# its own representation and your leverage is architecture, scale and
# optimisation. Here the model is a fixed, well-understood function and
# **you** are the representation learner. LightGBM cannot invent "this
# transaction is 8x this cardholder's 30-day average" — it can only split on
# columns you hand it.
#
# ## The discipline
#
# Every feature is computed from **strictly past rows**. A plain
# `groupby("uid")["amt"].transform("mean")` reads the account's whole history,
# including transactions that have not happened yet at scoring time. It will
# raise your score and it is not available in production.
#
# Section 2 doesn't assert this — it tests it.
#
# ## Plan
#
# 1. Build feature blocks A–F
# 2. **Verify** none of them reads the future
# 3. Ablate: add one block at a time, measure what each is worth
# 4. Show the purge gap now matters (it didn't much, at 7 days)
# 5. Final model, holdout, importance

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
from src import features as F
from src import models as M
from src import validation as V

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 220)

SMOKE = False          # data/raw is populated; set True to use the synthetic generator

# %%
df = D.load(smoke=SMOKE)
y = df[C.TARGET].to_numpy()

train_idx = np.load(C.INTERIM_DIR / "train_idx.npy")
holdout_idx = np.load(C.INTERIM_DIR / "holdout_idx.npy")

best_params = pd.read_json(C.INTERIM_DIR / "lgbm_best_params.json",
                           typ="series").to_dict()
best_params["num_leaves"] = int(best_params["num_leaves"])
best_params["min_child_samples"] = int(best_params["min_child_samples"])

print(f"purge gap is now {C.PURGE_DAYS} days "
      f"(longest window {F.MAX_WINDOW_DAYS}d)")

# %% [markdown]
# ## Section 1 - Build on the FULL frame, then split
#
# A point that trips people up, so it is worth being explicit.
#
# Building past-only features over the whole frame before splitting is
# **correct**, and it is not the same mistake as fitting a scaler before
# splitting. A scaler learns a statistic *from* the data (a mean over all rows,
# future included). A past-only feature at row *i* reads only rows < *i* by
# construction — whether rows after *i* happen to be in the same DataFrame is
# irrelevant to its value.
#
# It is also what production does: at scoring time the account's real history
# exists, and you want the feature computed over it.
#
# The one exception is **target** encoding, which does read labels. It uses the
# expanding (past-only) form here for the training block, and a
# training-block-only map for the holdout.

# %%
t0 = time.time()
feat = F.build_features(df)
print(f"  {time.time()-t0:.0f}s")

feat["uid_te"] = F.expanding_target_encode(df, "uid", y)
feat["card1_te"] = F.expanding_target_encode(df, "card1", y)
if "P_emaildomain" in df.columns:
    feat["email_te"] = F.expanding_target_encode(df, "P_emaildomain", y)

print(f"total engineered: {feat.shape[1]}")
print(feat.dtypes.value_counts().to_string())

# %%
# Sanity-check a few by hand before trusting any of them.
show = [c for c in ["uid_n_24h", "amt_to_uid_mean", "uid_secs_since_prev",
                    "uid_amt_mean_prior", "uid_te"] if c in feat.columns]
print(feat[show].describe().T.round(3).to_string())
print()
print("uid_n_24h == 0 on an account's first transaction, by construction.")
print(f"  share of rows with 0 prior 24h transactions: "
      f"{(feat['uid_n_24h'].fillna(0) == 0).mean():.1%}")

# %% [markdown]
# ## Section 2 - Prove the features cannot see the future
#
# The argument: if a feature at row *i* is genuinely past-only, recomputing it
# over `df[:i+1]` must give **exactly** the value computed over all of `df`.
# Rows after *i* could not have contributed. Any difference means the full-data
# version read forward.
#
# This catches the whole class of bug at once — a forgotten `closed="left"`, a
# `transform("mean")` that crept in, a target encoding including its own row —
# without reasoning about each feature separately.

# %%
sub = df.iloc[:40_000] if len(df) > 40_000 else df
bad = F.verify_past_only(lambda d: F.build_features(d, verbose=False),
                         sub, n_checks=10)
assert bad.empty, "leaking features -- fix before measuring anything"

# %% [markdown]
# **This test already earned its place.** The first version of block F used
# `value_counts()` frequency encoding — map each card1 to how often it appears.
# It never touches the label, so it feels obviously safe, and it failed this
# test immediately: a row's count depends on rows that come after it, and
# across the train/holdout boundary those counts are computed partly from the
# holdout.
#
# The fix was to split it into two honestly-different features:
#
# - `*_seen_before` — cumulative prior occurrences, past-only, lives in block F;
# - population frequency — a *fitted* transform, learned on train and applied
#   to the holdout, exactly like target encoding (applied below).
#
# Unsupervised does not mean leak-free. That distinction is worth having ready.

# %%
# And a deliberately broken feature, to show the test actually bites.
def leaky_build(d):
    out = F.build_features(d, groups=("A",), verbose=False)
    # The classic mistake: whole-history mean, future included.
    out["uid_amt_mean_LEAKY"] = d.groupby("uid")[C.AMOUNT_COL].transform("mean")
    return out

print("control - this SHOULD fail:")
caught = F.verify_past_only(leaky_build, sub, n_checks=6)
assert not caught.empty, "the leak detector failed to detect a known leak"
print(f"\ncaught it: {caught['feature'].unique().tolist()}")

# %% [markdown]
# Keep that control cell. A passing test proves nothing unless you have shown
# it can fail — and "I wrote a test that catches feature leakage, here is it
# catching one" is a much stronger claim than "I was careful."

# %% [markdown]
# ## Section 3 - Ablation: what is each block worth?

# %%
BLOCKS = {
    "A amount/time/D-norm": ("A",),
    "B velocity":           ("A", "B"),
    "C amount-vs-history":  ("A", "B", "C"),
    "D recency":            ("A", "B", "C", "D"),
    "E novelty":            ("A", "B", "C", "D", "E"),
    "F count encodings":    ("A", "B", "C", "D", "E", "F"),
}

train_df = df.iloc[train_idx].reset_index(drop=True)
hold_df = df.iloc[holdout_idx].reset_index(drop=True)
y_train, y_hold = y[train_idx], y[holdout_idx]

base_X, base_cats = M.prepare_lgbm_frame(train_df, max_cardinality=25)
feat_train = feat.iloc[train_idx].reset_index(drop=True)
feat_hold = feat.iloc[holdout_idx].reset_index(drop=True)


def cv_with(extra_cols, label):
    X = F.combine(base_X, feat_train[extra_cols]) if extra_cols else base_X
    rows, t0 = [], time.time()
    for k, (tr, va) in enumerate(V.PurgedTimeSeriesSplit().split(train_df)):
        X_tr = X.iloc[tr]
        X_va = M.align_categories(X_tr, X.iloc[va], base_cats)
        mdl, _ = M.fit_lgbm_early_stopping(X_tr, y_train[tr], base_cats, **best_params)
        p = mdl.predict_proba(X_va)[:, 1]
        rec, _ = E.recall_at_fpr(y_train[va], p)
        rows.append({"label": label, "fold": k,
                     "pr_auc": E.pr_auc(y_train[va], p), "recall@1%fpr": rec})
    out = pd.DataFrame(rows)
    print(f"  {label:24s} n_feat {X.shape[1]:>3}  PR-AUC "
          f"{out.pr_auc.mean():.4f} +/- {out.pr_auc.std():.4f}  ({time.time()-t0:.0f}s)")
    return out


abl = [cv_with([], "0 raw only")]
for label, groups in BLOCKS.items():
    cols = [c for c in F.build_features(df.iloc[:200], groups=groups,
                                        verbose=False).columns if c in feat.columns]
    abl.append(cv_with(cols, label))

te_cols = [c for c in ["uid_te", "card1_te", "email_te"] if c in feat.columns]
abl.append(cv_with(list(feat.columns), "G + target encoding"))

ablation = pd.concat(abl, ignore_index=True)

# %%
summ = (ablation.groupby("label", sort=False)
        .agg(pr_auc=("pr_auc", "mean"), sd=("pr_auc", "std"),
             recall=("recall@1%fpr", "mean"))
        .round(4))
summ["delta"] = summ["pr_auc"].diff().round(4)
summ["cum_lift"] = (summ["pr_auc"] / summ["pr_auc"].iloc[0]).round(2)
print(summ.to_string())

raw, best = summ["pr_auc"].iloc[0], summ["pr_auc"].max()
print(f"""
  raw features only   {raw:.4f}
  best block set      {best:.4f}
  lift                {best/raw:.2f}x

Compare against notebooks 04-05, where tuning and imbalance handling together
moved PR-AUC by less than one fold standard deviation.
""")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
order = summ.index.tolist()

sns.barplot(data=ablation, y="label", x="pr_auc", order=order, hue="label",
            hue_order=order, legend=False, errorbar="sd", ax=axes[0],
            palette="viridis")
axes[0].axvline(raw, ls="--", c="0.35", lw=1.2)
axes[0].set_title("Cumulative PR-AUC as blocks are added")

d = summ["delta"].fillna(0)
axes[1].barh(range(len(d)), d.to_numpy(),
             color=["#55A868" if v > 0 else "#C44E52" for v in d])
axes[1].set_yticks(range(len(d)))
axes[1].set_yticklabels(d.index, fontsize=8)
axes[1].axvline(0, c="0.3", lw=1)
axes[1].invert_yaxis()
axes[1].set_title("Marginal contribution of each block")

plt.tight_layout()
plt.savefig(C.FIG_DIR / "06_ablation.png", dpi=140)
plt.show()

# %% [markdown]
# Blocks with a **negative** marginal delta are worth as much as the positive
# ones in a writeup. Adding columns that do not help is not free: it costs fit
# time, it dilutes column sampling, and on a drifting problem it adds surface
# area for the model to overfit. Being able to say "novelty flags did not pay
# for themselves and I dropped them" reads as judgement rather than as
# throwing every feature at the wall.

# %% [markdown]
# ## Section 4 - Now the purge gap earns its keep
#
# In notebook 02 with raw features and 7 days, purging barely moved the number.
# The feature set is now dominated by trailing aggregates up to 30 days, which
# is exactly the regime where a training row near a fold boundary summarises
# validation rows.
#
# Same features, same model, three gap settings.

# %%
X_full = F.combine(base_X, feat_train)
gap_rows = []
for gap in (0, 7, F.MAX_WINDOW_DAYS):
    scores = []
    for tr, va in V.PurgedTimeSeriesSplit(gap_days=gap).split(train_df):
        X_tr = X_full.iloc[tr]
        X_va = M.align_categories(X_tr, X_full.iloc[va], base_cats)
        mdl, _ = M.fit_lgbm_early_stopping(X_tr, y_train[tr], base_cats, **best_params)
        scores.append(E.pr_auc(y_train[va], mdl.predict_proba(X_va)[:, 1]))
    gap_rows.append({"gap_days": gap, "pr_auc": float(np.mean(scores)),
                     "sd": float(np.std(scores))})
    print(f"  gap {gap:>2}d   PR-AUC {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

gaps = pd.DataFrame(gap_rows)
inflation = 100 * (gaps.pr_auc.iloc[0] - gaps.pr_auc.iloc[-1]) / gaps.pr_auc.iloc[-1]
print(f"""
Reporting with gap=0 would overstate PR-AUC by {inflation:.1f}%.

The mechanism, concretely: a training row one hour before the boundary carries
a "transactions in the trailing 30 days" value computed over a window that
extends past the boundary. The feature encodes validation rows. Skipping
{F.MAX_WINDOW_DAYS} days between train and validation removes the overlap.
""")

# %% [markdown]
# ## Section 5 - Final model and holdout

# %%
keep = list(feat.columns)
X_tr_final = F.combine(base_X, feat_train[keep])

hold_base, _ = M.prepare_lgbm_frame(hold_df)
X_ho_final = F.combine(hold_base[base_X.columns], feat_hold[keep])
X_ho_final = M.align_categories(X_tr_final, X_ho_final, base_cats)

# Target encoding for the holdout is learned from the whole training block --
# safe because the holdout is entirely after it.
for col, name in [("uid", "uid_te"), ("card1", "card1_te"),
                  ("P_emaildomain", "email_te")]:
    if name in X_ho_final.columns and col in df.columns:
        X_ho_final[name] = F.fit_holdout_target_encode(
            train_df, y_train, hold_df, col).to_numpy()

# Population-frequency encoding: the fitted transform that could not live in
# build_features. Fit on train, apply to both sides. This is the "rare value"
# signal that a cumulative prior count cannot express.
for col in ["card1", "addr1", "P_emaildomain", "uid"]:
    if col in df.columns:
        X_tr_final[f"{col}_freq"] = F.fit_apply_count_encode(
            train_df[col], train_df[col]).to_numpy()
        X_ho_final[f"{col}_freq"] = F.fit_apply_count_encode(
            train_df[col], hold_df[col]).to_numpy()
print(f"final matrix: {X_tr_final.shape[1]} columns")

model, n_trees = M.fit_lgbm_early_stopping(X_tr_final, y_train, base_cats,
                                           **best_params)
p_feat = model.predict_proba(X_ho_final)[:, 1]

n_days = hold_df["day"].max() - hold_df["day"].min()
m_feat = E.evaluate_all(y_hold, p_feat, amount=hold_df[C.AMOUNT_COL], n_days=n_days)

prev = pd.read_csv(C.RESULTS_CSV)
prior_best = prev[prev.name.isin(["lgbm_tuned", "lgbm_none"])]["pr_auc"].max()

print(f"engineered features, holdout   ({X_tr_final.shape[1]} columns, {n_trees} trees)")
for k in ["pr_auc", "roc_auc", "recall_at_0.01fpr", "brier", "lift_over_base",
          "alerts_per_day", "min_total_cost", "recall_at_best"]:
    print(f"  {k:20s} {m_feat[k]:,.4f}")
print(f"\n  previous best (day 4-8) PR-AUC {prior_best:.4f}"
      f"  ->  {m_feat['pr_auc']:.4f}   ({m_feat['pr_auc']/prior_best:.2f}x)")

E.log_result("lgbm_engineered", "temporal_holdout", m_feat,
             f"{X_tr_final.shape[1]} features, trees={n_trees}, purge={C.PURGE_DAYS}d")

# %%
gain = pd.Series(model.booster_.feature_importance("gain"),
                 index=X_tr_final.columns).sort_values(ascending=False)
eng = set(feat.columns)
top = gain.head(25)
colors = ["#55A868" if c in eng else "#4C72B0" for c in top.index]

fig, ax = plt.subplots(figsize=(8, 8))
top[::-1].plot.barh(ax=ax, color=colors[::-1])
ax.set_title("Top 25 by gain\ngreen = engineered, blue = raw")
plt.tight_layout()
plt.savefig(C.FIG_DIR / "06_importance.png", dpi=140)
plt.show()

share = gain[gain.index.isin(eng)].sum() / gain.sum()
print(f"engineered features carry {share:.1%} of total gain "
      f"({len(eng)} of {len(gain)} columns)")

# %%
fig, ax = plt.subplots(figsize=(6.5, 5))
curves = {"engineered": (y_hold, p_feat)}
prev_p = C.INTERIM_DIR / "p_lgbm_holdout.npy"
if prev_p.exists():
    curves["raw features (day 4-8)"] = (y_hold, np.load(prev_p))
E.plot_pr_curves(curves, ax=ax, base_rate=float(y_hold.mean()))
plt.tight_layout()
plt.savefig(C.FIG_DIR / "06_pr_comparison.png", dpi=140)
plt.show()

np.save(C.INTERIM_DIR / "p_engineered_holdout.npy", p_feat)
feat.to_parquet(C.INTERIM_DIR / "features.parquet")
pd.Series(keep).to_json(C.INTERIM_DIR / "feature_names.json")
print(f"saved features + predictions to {C.INTERIM_DIR}")

# %% [markdown]
# ## The CV bullet this phase earns
#
# > Engineered 40+ leak-safe velocity and behavioural features over 590k
# > transactions using strictly past-window aggregations under a temporal
# > validation scheme, lifting PR-AUC [X]x over raw inputs — an order of
# > magnitude more than hyperparameter tuning contributed. Verified leak-safety
# > with a prefix-recomputation test that fails if any feature value depends on
# > a future row.
#
# The second sentence is the one that will get asked about. Have
# `verify_past_only` open when you answer.
#
# ## Days 16-22 next: the deep learning arm
#
# Where your background stops being a gap and starts being the differentiator.
# Note what block C had to approximate with summary statistics — "how unusual
# is this amount for this cardholder" compressed into a mean and a z-score. A
# sequence model over the account's transaction history does not need the
# compression; it sees the trajectory. That is the one place a neural network
# has a structural advantage over a GBDT here, and it is the experiment worth
# running properly.
