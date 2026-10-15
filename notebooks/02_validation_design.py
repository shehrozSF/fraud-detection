# %% [markdown]
# # 02 - Validation design: measuring the lie
#
# One model, one feature set, one hyperparameter setting. The **only** thing
# that changes across this notebook is how the data is split. Whatever spread
# appears in the scores is pure methodology.
#
# Three schemes:
#
# | scheme | account can span folds | trains on future rows | verdict |
# |---|---|---|---|
# | `random_kfold` | yes | yes | wrong twice over |
# | `group_kfold_uid` | no | yes | isolates *identity* leakage |
# | `PurgedTimeSeriesSplit` | yes (legitimately) | no | what deployment looks like |
#
# Running all three separates the two leaks, which is the part that makes this
# a real diagnostic rather than a slogan.
#
# **On the model choice here:** this notebook uses LightGBM rather than the
# logistic-regression baseline from notebook 03, because leakage is a
# *memorisation* phenomenon. A regularised linear model on scaled features has
# limited capacity to memorise an account id; a tree ensemble will happily
# carve out individual cards, which is exactly what makes the effect visible.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from src import config as C
from src import data as D
from src import evaluate as E
from src import validation as V

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

SMOKE = False          # data/raw is populated; set True to use the synthetic generator

# %%
df = D.load(smoke=SMOKE)
print(df.shape, f"fraud {df[C.TARGET].mean():.3%}")

# %% [markdown]
# ## A deliberately naive feature set
#
# Raw numeric columns only, no engineering. Feature work starts on day 9; the
# point today is that the *split* dominates, and a weak feature set makes that
# easier to see rather than harder.

# %%
DROP = {C.ID_COL, C.TARGET, C.TIME_COL, "datetime", "uid", "day"}
FEATURES = [c for c in df.select_dtypes(include=[np.number]).columns if c not in DROP]
print(f"{len(FEATURES)} numeric features")

X = df[FEATURES]
y = df[C.TARGET].to_numpy()

# %% [markdown]
# ## Step 1 - Look at the folds before trusting any score
#
# `describe_split` reports, per fold, how much of the validation set the model
# has already met. Read `pct_val_rows_seen_uid` as a leakage thermometer.

# %%
schemes = {
    "random_kfold":   V.random_kfold(df),
    "group_kfold_uid": V.group_kfold_uid(df),
    "purged_time":    list(V.PurgedTimeSeriesSplit().split(df)),
}

report = pd.concat([V.split_report(df, s, name) for name, s in schemes.items()],
                   ignore_index=True)
report

# %%
summary = (report.groupby("scheme")
           .agg(mean_pct_val_rows_seen=("pct_val_rows_seen_uid", "mean"),
                mean_pct_val_uids_seen=("pct_val_uids_seen", "mean"),
                val_after_train=("val_starts_after_train", "all"),
                mean_n_train=("n_train", "mean"))
           .round(1))
print(summary.to_string())

print("""
Read that table before reading any metric.

  random_kfold        : the model has already seen most validation accounts,
                        AND has trained on rows from after the validation
                        period. Two independent leaks stacked.
  group_kfold_uid     : accounts are clean, but training still spans the
                        future. Any remaining inflation vs. purged_time is
                        temporal.
  purged_time         : validation always begins after training ends. Some
                        account overlap remains and that is CORRECT -- real
                        cardholders do transact across months, and a
                        deployed model genuinely has their history.
""")

# %% [markdown]
# ## Step 2 - Same model, three splits

# %%
import lightgbm as lgb

PARAMS = dict(
    objective="binary",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=C.SEED,
    verbose=-1,
)


def run_scheme(name, splits):
    """Fit/score every fold, returning per-fold metrics."""
    rows = []
    for k, (tr, va) in enumerate(splits):
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]

        rec, _ = E.recall_at_fpr(y[va], p)
        rows.append({
            "scheme": name, "fold": k,
            "pr_auc": E.pr_auc(y[va], p),
            "roc_auc": float(roc_auc_score(y[va], p)),
            "recall@1%fpr": rec,
            "n_val": len(va),
        })
        print(f"  {name:16s} fold {k}  PR-AUC {rows[-1]['pr_auc']:.4f}")
    return pd.DataFrame(rows)


results = pd.concat([run_scheme(n, s) for n, s in schemes.items()],
                    ignore_index=True)

# %%
agg = (results.groupby("scheme")
       .agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            recall_mean=("recall@1%fpr", "mean"))
       .round(4)
       .reindex(["random_kfold", "group_kfold_uid", "purged_time"]))
print(agg.to_string())

honest = agg.loc["purged_time", "pr_auc_mean"]
naive = agg.loc["random_kfold", "pr_auc_mean"]
grouped = agg.loc["group_kfold_uid", "pr_auc_mean"]

print(f"""
Decomposition of the inflation
------------------------------
  random k-fold  PR-AUC  {naive:.4f}     <- the number most portfolios report
  group k-fold   PR-AUC  {grouped:.4f}
  purged time    PR-AUC  {honest:.4f}     <- the only one that means anything

  total overstatement     {100*(naive-honest)/honest:6.1f}%
    from account identity {100*(naive-grouped)/honest:6.1f}%
    from training on the future {100*(grouped-honest)/honest:6.1f}%
""")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

order = ["random_kfold", "group_kfold_uid", "purged_time"]
PAL = ["#C44E52", "#DD8452", "#4C72B0"]
sns.barplot(data=results, x="scheme", y="pr_auc", order=order, hue="scheme",
            hue_order=order, legend=False, ax=axes[0], errorbar="sd", palette=PAL)
axes[0].axhline(honest, ls="--", c="0.4", lw=1)
axes[0].set_title("PR-AUC by validation scheme\n(identical model and features)")
axes[0].set_ylabel("PR-AUC")

sns.barplot(data=results, x="scheme", y="roc_auc", order=order, hue="scheme",
            hue_order=order, legend=False, ax=axes[1], errorbar="sd", palette=PAL)
axes[1].set_ylim(0.5, 1.0)
axes[1].set_title("ROC-AUC barely notices\n(why it is the wrong headline metric)")

for a in axes:
    a.tick_params(axis="x", rotation=12)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "02_split_comparison.png", dpi=140)
plt.show()

# %% [markdown]
# ## Step 3 - Does the honest score decay over time?
#
# Under `PurgedTimeSeriesSplit` each fold validates on a later slice than the
# last. Plotting PR-AUC against fold index shows whether the model degrades as
# it forecasts further from its training window. A downward slope is a drift
# signature and is the argument for a retraining cadence.

# %%
pt = results[results["scheme"] == "purged_time"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(pt["fold"], pt["pr_auc"], "o-", lw=2)
ax.set_xlabel("fold (later = further into the future)")
ax.set_ylabel("PR-AUC")
ax.set_title("Performance across successive future blocks")
plt.tight_layout()
plt.savefig(C.FIG_DIR / "02_temporal_decay.png", dpi=130)
plt.show()

# %% [markdown]
# ## Step 4 - Why the purge gap exists
#
# A concrete demonstration, since "purging" sounds like superstition until you
# watch it bite.
#
# Build one obviously-leaky feature: a rolling 7-day count per account. Fit
# with `gap_days=0` and again with `gap_days=7`, and compare. With no gap, a
# training row near the boundary summarises rows sitting inside validation.

# %%
tmp = df[[C.TIME_COL, "uid", C.TARGET]].copy()
tmp["one"] = 1.0
# Trailing 7-day transaction count per account (past-inclusive; day 9-15
# rebuilds this properly with the current row excluded).
tmp = tmp.sort_values(C.TIME_COL)
tmp["idx_time"] = pd.to_datetime(tmp[C.TIME_COL], unit="s")
tmp["velocity_7d"] = (tmp.set_index("idx_time")
                        .groupby("uid")["one"]
                        .rolling("7D").sum()
                        .reset_index(level=0, drop=True)
                        .to_numpy())

Xg = pd.concat([X.reset_index(drop=True),
                tmp["velocity_7d"].reset_index(drop=True)], axis=1)

for gap in (0, C.PURGE_DAYS):
    splits = list(V.PurgedTimeSeriesSplit(gap_days=gap).split(df))
    scores = []
    for tr, va in splits:
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xg.iloc[tr], y[tr])
        scores.append(E.pr_auc(y[va], m.predict_proba(Xg.iloc[va])[:, 1]))
    print(f"gap_days={gap:2d}   PR-AUC {np.mean(scores):.4f}  (folds: "
          f"{', '.join(f'{s:.4f}' for s in scores)})")

print("""
The gap=0 number is the optimistic one. The difference is small compared to
the split effect above -- and that is the honest finding to report. Purging
matters most once the feature set is dominated by long-window aggregates,
which is precisely what day 9-15 builds.
""")

# %% [markdown]
# ## Step 5 - Lock the final holdout and walk away from it
#
# From here on: tune against `PurgedTimeSeriesSplit` on the training block.
# The holdout gets scored **once**, at the end. Every extra peek at it is a
# hyperparameter fit by hand, and the estimate degrades accordingly.

# %%
train_idx, holdout_idx, cut = V.temporal_holdout(df)
print(f"cut at day {cut / C.SECONDS_PER_DAY:.1f}")
print(f"train   {len(train_idx):,} rows  fraud {y[train_idx].mean():.3%}")
print(f"holdout {len(holdout_idx):,} rows  fraud {y[holdout_idx].mean():.3%}")

np.save(C.INTERIM_DIR / "train_idx.npy", train_idx)
np.save(C.INTERIM_DIR / "holdout_idx.npy", holdout_idx)
print(f"\nsaved to {C.INTERIM_DIR}")

# %%
for scheme in order:
    r = results[results["scheme"] == scheme]
    E.log_result(
        name="lgbm_raw_numeric",
        scheme=scheme,
        metrics={"pr_auc": r["pr_auc"].mean(), "roc_auc": r["roc_auc"].mean(),
                 "recall_at_0.01fpr": r["recall@1%fpr"].mean(),
                 "base_rate": float(y.mean())},
        notes="notebook 02 split comparison, identical model/features",
    )
print(pd.read_csv(C.RESULTS_CSV).tail(3).to_string(index=False))

# %% [markdown]
# ## What to say in an interview
#
# > "The first thing I did was build the splitter, before any model. This data
# > is time-ordered with a drifting fraud rate, and accounts recur, so a
# > shuffled k-fold leaks in two directions at once. I measured both: random
# > k-fold overstated PR-AUC by roughly N%, and I could attribute X% of that to
# > account identity and Y% to training on future rows. Every number in the
# > project after that is quoted under an expanding-window split with a 7-day
# > purge gap, and the final holdout was scored once."
#
# That paragraph is worth more than any single model result in this repo.
