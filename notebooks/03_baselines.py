# %% [markdown]
# # 03 - Baselines
#
# Every number in the rest of the project is quoted as a delta against what is
# built here. Three rungs:
#
# 1. **amount-only** - rank by transaction size. The floor.
# 2. **logistic regression** - the real baseline, under the honest splitter.
# 3. **cost framing** - turn the score into a decision, in dollars.
#
# Step 3 is unusual this early. It is here on purpose: once you have seen the
# cost curve, you stop optimising PR-AUC in a vacuum, and every later modelling
# choice gets argued in terms of what it does to the queue.
#
# ## The sklearn idiom to internalise
#
# `Pipeline` makes preprocessing part of the estimator, so imputation and
# scaling are fit on the training fold and merely *applied* to validation.
# Coming from DL the reflex is to normalise the whole dataset once up front --
# fine when your split is random and iid, silently wrong here.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

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

# %%
df = D.load(smoke=SMOKE)
y = df[C.TARGET].to_numpy()

# Reuse the exact holdout locked in notebook 02 -- do not recompute it, and do
# not re-tune against it.
train_idx = np.load(C.INTERIM_DIR / "train_idx.npy")
holdout_idx = np.load(C.INTERIM_DIR / "holdout_idx.npy")

train_df = df.iloc[train_idx].reset_index(drop=True)
hold_df = df.iloc[holdout_idx].reset_index(drop=True)
y_train, y_hold = y[train_idx], y[holdout_idx]

print(f"train   {len(train_df):,}  days {train_df['day'].min():.0f}-{train_df['day'].max():.0f}  fraud {y_train.mean():.3%}")
print(f"holdout {len(hold_df):,}  days {hold_df['day'].min():.0f}-{hold_df['day'].max():.0f}  fraud {y_hold.mean():.3%}")

# %% [markdown]
# ## Rung 1 - Amount only
#
# No fitting. Just sort by transaction amount and read off the metrics.

# %%
p_amt = M.amount_only_baseline(hold_df)
m_amt = E.evaluate_all(y_hold, p_amt, amount=hold_df[C.AMOUNT_COL],
                       n_days=hold_df["day"].max() - hold_df["day"].min())

print(f"PR-AUC        {m_amt['pr_auc']:.4f}   (chance = {m_amt['base_rate']:.4f})")
print(f"ROC-AUC       {m_amt['roc_auc']:.4f}")
print(f"recall@1%FPR  {m_amt['recall_at_0.01fpr']:.3f}")
print()
print("If this beats chance, amount alone carries signal -- and any model that")
print("does not clear it decisively has learned nothing you didn't already have.")

E.log_result("amount_only", "temporal_holdout", m_amt, "no fitting, rank by amount")

# %% [markdown]
# ## Rung 2 - Logistic regression
#
# First under cross-validation on the *training block only*, then once on the
# holdout.

# %%
numeric, categorical, dropped = M.split_feature_types(train_df)
print(f"numeric      {len(numeric)}")
print(f"categorical  {len(categorical)}  -> {categorical}")
print(f"dropped      {len(dropped)}  (too high-cardinality for one-hot) -> {dropped}")

# %%
cv = V.PurgedTimeSeriesSplit()
fold_rows = []

for k, (tr, va) in enumerate(cv.split(train_df)):
    pipe = M.make_logreg_pipeline(numeric, categorical)
    pipe.fit(train_df.iloc[tr], y_train[tr])
    p = pipe.predict_proba(train_df.iloc[va])[:, 1]

    m = E.evaluate_all(y_train[va], p, amount=train_df.iloc[va][C.AMOUNT_COL])
    m["fold"] = k
    fold_rows.append(m)
    print(f"  fold {k}  PR-AUC {m['pr_auc']:.4f}   recall@1%FPR {m['recall_at_0.01fpr']:.3f}")

cv_res = pd.DataFrame(fold_rows)
print(f"\nCV PR-AUC  {cv_res['pr_auc'].mean():.4f} +/- {cv_res['pr_auc'].std():.4f}")

# %% [markdown]
# The fold-to-fold standard deviation is worth reading carefully. If a later
# model beats this baseline by less than one standard deviation, the
# improvement is noise. That is the bar every day-4-onward experiment has to
# clear before it goes in the README.

# %%
# Refit on the full training block, score the holdout once.
pipe = M.make_logreg_pipeline(numeric, categorical)
pipe.fit(train_df, y_train)
p_lr = pipe.predict_proba(hold_df)[:, 1]

n_days = hold_df["day"].max() - hold_df["day"].min()
m_lr = E.evaluate_all(y_hold, p_lr, amount=hold_df[C.AMOUNT_COL], n_days=n_days)

print("logistic regression, holdout")
for k in ["pr_auc", "roc_auc", "recall_at_0.01fpr", "brier", "lift_over_base"]:
    print(f"  {k:20s} {m_lr[k]:.4f}")

E.log_result("logreg_balanced", "temporal_holdout", m_lr,
             f"C=0.1, class_weight=balanced, {len(numeric)}num+{len(categorical)}cat")

# %%
fig, ax = plt.subplots(figsize=(6.5, 5))
E.plot_pr_curves(
    {"amount only": (y_hold, p_amt), "logistic regression": (y_hold, p_lr)},
    ax=ax, base_rate=float(y_hold.mean()),
)
plt.tight_layout()
plt.savefig(C.FIG_DIR / "03_baseline_pr.png", dpi=140)
plt.show()

# %% [markdown]
# ## Are the probabilities believable?
#
# Ranking metrics are invariant to any monotone squashing of the scores, so a
# model can rank perfectly and still be badly calibrated. That stops being
# acceptable the moment probabilities are multiplied by dollar amounts, which
# is the next cell.
#
# `class_weight="balanced"` guarantees miscalibration: it deliberately
# over-weights the positive class, so predicted probabilities come out far
# above the true fraud rate. Isotonic regression fixes it on day 23-26. Look
# now so the shape is familiar.

# %%
rel = E.reliability_table(y_hold, p_lr, n_bins=10)
print(rel.to_string(index=False))

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot([0, rel["mean_predicted"].max()], [0, rel["mean_predicted"].max()],
        "--", c="0.5", lw=1, label="perfect")
ax.plot(rel["mean_predicted"], rel["observed"], "o-", lw=2, label="logreg (balanced)")
ax.set_xlabel("mean predicted probability")
ax.set_ylabel("observed fraud rate")
ax.set_title(f"Reliability  (Brier = {m_lr['brier']:.4f})")
ax.legend()
plt.tight_layout()
plt.savefig(C.FIG_DIR / "03_reliability.png", dpi=130)
plt.show()

print("\nPoints below the diagonal = over-predicting fraud, as expected from")
print("class_weight='balanced'. Uncorrected, this inflates every expected-loss")
print("calculation below, so treat today's dollar figures as directional.")

# %% [markdown]
# ## Rung 3 - From a score to a decision
#
# Cost assumptions live in `src/config.py`:
# review $10/alert, chargeback fee $25, recovery 0%.

# %%
curve = E.cost_curve(y_hold, p_lr, hold_df[C.AMOUNT_COL])
best = curve.loc[curve["total_cost"].idxmin()]

no_model = E.expected_cost(y_hold, p_lr, hold_df[C.AMOUNT_COL], threshold=np.inf)
review_all = E.expected_cost(y_hold, p_lr, hold_df[C.AMOUNT_COL], threshold=-np.inf)

print(f"do nothing (flag none)   total cost ${no_model['total_cost']:>12,.0f}")
print(f"review everything        total cost ${review_all['total_cost']:>12,.0f}")
print(f"model at best threshold  total cost ${best['total_cost']:>12,.0f}")
print()
print(f"  threshold      {best['threshold']:.4f}")
print(f"  alerts         {best['n_alerts']:,}  ({best['alert_rate']:.2%} of traffic, "
      f"{best['n_alerts']/n_days:,.0f}/day)")
print(f"  precision      {best['precision']:.3f}")
print(f"  recall         {best['recall']:.3f}  (count)")
print(f"  value recall   {best['value_recall']:.3f}  (dollars)")
print(f"  saving vs do-nothing  ${no_model['total_cost']-best['total_cost']:,.0f} "
      f"({100*(no_model['total_cost']-best['total_cost'])/no_model['total_cost']:.1f}%)")

# %% [markdown]
# `recall` and `value_recall` differing is the interesting part: catching 60%
# of fraud *cases* is not the same as stopping 60% of fraud *dollars*. Risk
# teams are funded on the second one.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(curve["alert_rate"], curve["total_cost"], lw=2, label="total")
axes[0].plot(curve["alert_rate"], curve["fraud_loss"], lw=1.5, ls="--", label="fraud loss")
axes[0].plot(curve["alert_rate"], curve["review_cost"], lw=1.5, ls=":", label="review cost")
axes[0].axvline(best["alert_rate"], c="#C44E52", lw=1)
axes[0].set_xlabel("share of traffic alerted")
axes[0].set_ylabel("cost ($)")
axes[0].set_title("The two costs trade off. The minimum is the operating point.")
axes[0].set_xscale("log")
axes[0].legend()

axes[1].plot(curve["recall"], curve["precision"], lw=2)
axes[1].scatter([best["recall"]], [best["precision"]], c="#C44E52", zorder=5,
                label="cost-optimal")
axes[1].set_xlabel("recall")
axes[1].set_ylabel("precision")
axes[1].set_title("Where the cost optimum sits on the PR curve")
axes[1].legend()

plt.tight_layout()
plt.savefig(C.FIG_DIR / "03_cost_curve.png", dpi=140)
plt.show()

# %% [markdown]
# ## A preview of the amount-weighted rule
#
# The curve above finds one global cutoff. But the break-even probability
# depends on what is at stake:
#
#     flag when   p * (amount + fee)  >  review_cost
#
# A $4,000 transaction is worth reviewing at p = 0.25%; a $12 one needs
# p > 27%. A single threshold cannot express both. Day 23-26 develops this
# properly on calibrated probabilities -- this is just the shape of it.

# %%
flag_ev = E.expected_value_decision(p_lr, hold_df[C.AMOUNT_COL])
tp = int(((flag_ev) & (y_hold == 1)).sum())

ev_cost = (flag_ev.sum() * C.REVIEW_COST
           + (hold_df.loc[(~flag_ev) & (y_hold == 1), C.AMOUNT_COL].sum()
              + C.CHARGEBACK_FEE * int(((~flag_ev) & (y_hold == 1)).sum())))

print(f"amount-weighted rule   alerts {int(flag_ev.sum()):,}  "
      f"recall {tp/max(y_hold.sum(),1):.3f}  total cost ${ev_cost:,.0f}")
print(f"single-threshold rule  alerts {int(best['n_alerts']):,}  "
      f"recall {best['recall']:.3f}  total cost ${best['total_cost']:,.0f}")
print()
print("Expect the amount-weighted rule to look mediocre right now -- it relies")
print("on calibrated probabilities and these are not calibrated. Re-run this")
print("cell after isotonic regression on day 23-26; that before/after is the")
print("cleanest demonstration of why calibration matters in risk.")

# %%
print(pd.read_csv(C.RESULTS_CSV).to_string(index=False))

# %% [markdown]
# ## Days 1-3, done
#
# What exists now:
#
# - a splitter that reflects deployment, and a measurement of what the wrong
#   one would have claimed;
# - a locked holdout, scored once;
# - a baseline with a known fold-to-fold standard deviation, so later
#   improvements can be judged against noise;
# - metrics denominated in dollars and alerts/day, not just PR-AUC.
#
# **Day 4-8 next:** LightGBM properly, then the imbalance benchmark
# (`scale_pos_weight` vs. SMOTE vs. undersampling). Prediction to write down
# now and check later: SMOTE will underperform class weighting here, because
# interpolating between transactions in a mixed categorical/numeric space
# produces rows that could not occur. Being able to explain *why* the popular
# technique loses is worth more than using it.
