# %% [markdown]
# # 01 - EDA: what shape is this problem?
#
# Goal for today is *not* a model. It is to answer four questions that decide
# every later design choice:
#
# 1. How rare is fraud, and therefore which metrics are usable?
# 2. Is the data time-ordered, and does the fraud pattern drift?
# 3. Do accounts recur across the timeline? (decides the splitter)
# 4. What is missing, and is it missing for a reason?
#
# Run cells with Shift+Enter. In VS Code these `# %%` markers render as an
# interactive notebook without any .ipynb file.

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

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 60)
sns.set_theme(style="whitegrid", context="notebook")

# Set True to use the synthetic generator instead of the Kaggle download.
SMOKE = False          # data/raw is populated; set True to use the synthetic generator

# %%
df = D.load(smoke=SMOKE)
print(df.shape)
df.head(3)

# %% [markdown]
# ## Q1 - How rare is fraud?

# %%
n = len(df)
n_fraud = int(df[C.TARGET].sum())
rate = n_fraud / n

print(f"transactions : {n:,}")
print(f"fraud        : {n_fraud:,}")
print(f"fraud rate   : {rate:.3%}")
print()
print(f"A model that predicts 'never fraud' scores {1-rate:.3%} accuracy.")
print(f"Baseline (chance) average precision is exactly the fraud rate: {rate:.4f}.")
print(f"Baseline ROC-AUC is 0.500 regardless of imbalance -- which is why")
print(f"ROC-AUC will look reassuring here long after the model stops being useful.")

# %% [markdown]
# **Why this kills accuracy and softens ROC-AUC.**
#
# ROC-AUC's x-axis is FPR = FP / (all negatives). With ~96.5% negatives the
# denominator is enormous, so thousands of false alarms barely move the curve.
# Precision's denominator is FP + TP, which is small. PR-AUC therefore stays
# sensitive to exactly the errors that flood an analyst queue.
#
# Coming from DL: this is the same reason mIoU is preferred over pixel
# accuracy on sparse segmentation masks. Same failure, tabular clothing.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df[C.TARGET].value_counts().plot.bar(ax=axes[0], color=["#4C72B0", "#C44E52"])
axes[0].set_title(f"Class counts (fraud = {rate:.2%})")
axes[0].set_yscale("log")
axes[0].set_xticklabels(["legit", "fraud"], rotation=0)

for lab, g in df.groupby(C.TARGET):
    axes[1].hist(np.log1p(g[C.AMOUNT_COL]), bins=60, density=True, alpha=0.55,
                 label="fraud" if lab else "legit")
axes[1].set_xlabel("log1p(TransactionAmt)")
axes[1].set_title("Amount distribution by class")
axes[1].legend()

plt.tight_layout()
plt.savefig(C.FIG_DIR / "01_class_and_amount.png", dpi=130)
plt.show()

# %%
print(df.groupby(C.TARGET)[C.AMOUNT_COL].describe()[["count", "mean", "50%", "max"]])
print()
print("Fraud as a share of transactions, by amount decile:")
dec = pd.qcut(df[C.AMOUNT_COL], 10, labels=False, duplicates="drop")
print((df.groupby(dec)[C.TARGET].agg(["mean", "size"])
         .rename(columns={"mean": "fraud_rate", "size": "n"})))

# %% [markdown]
# ## Q2 - Is it time-ordered, and does the pattern drift?
#
# This is the question that decides the splitter, so it gets the most care.

# %%
span_days = df["day"].max() - df["day"].min()
print(f"span: {span_days:.0f} days  ({df['datetime'].min().date()} -> {df['datetime'].max().date()})")
print(f"TransactionDT monotonic after sort: {df[C.TIME_COL].is_monotonic_increasing}")

daily = (df.assign(d=df["day"].astype(int))
           .groupby("d")
           .agg(n=(C.TARGET, "size"), fraud_rate=(C.TARGET, "mean"),
                amt=(C.AMOUNT_COL, "median")))

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(daily.index, daily["n"], lw=1)
axes[0].set_ylabel("transactions/day")
axes[1].plot(daily.index, daily["fraud_rate"], lw=1, color="#C44E52")
axes[1].axhline(rate, ls="--", c="0.5", lw=1)
axes[1].set_ylabel("fraud rate")
axes[2].plot(daily.index, daily["amt"], lw=1, color="#55A868")
axes[2].set_ylabel("median amount")
axes[2].set_xlabel("day")
axes[0].set_title("Everything drifts. This is why a random split lies.")

plt.tight_layout()
plt.savefig(C.FIG_DIR / "01_drift.png", dpi=130)
plt.show()

# %%
# Quantify the drift rather than eyeballing it: compare the first and last
# thirds of the timeline.
q1, q2 = df["day"].quantile([1/3, 2/3])
early, late = df[df["day"] <= q1], df[df["day"] > q2]

print(f"fraud rate   first third {early[C.TARGET].mean():.3%}   last third {late[C.TARGET].mean():.3%}")
print(f"median amt   first third {early[C.AMOUNT_COL].median():.2f}   last third {late[C.AMOUNT_COL].median():.2f}")
print()
print("If these differ materially, a model validated on shuffled data is being")
print("scored on a distribution it will never actually see in deployment.")

# %% [markdown]
# ## Q3 - Do accounts recur?
#
# `uid` is the pseudo-account key from `src/data.py` (card1 + addr1 +
# registration day inferred from D1). If accounts recur heavily, a shuffled
# split will place the same card on both sides of the boundary and the model
# can score well by memorising *who* rather than learning *what fraud looks
# like*. Notebook 02 measures the damage; here we just establish that the
# condition holds.

# %%
vc = df["uid"].value_counts()
print(f"unique uids            : {vc.size:,}")
print(f"rows per uid  median   : {vc.median():.0f}   mean {vc.mean():.1f}   max {vc.max():,}")
print(f"rows in a uid seen >1x : {(df['uid'].map(vc) > 1).mean():.1%}")

# Is fraud concentrated in particular accounts?
per_uid = df.groupby("uid")[C.TARGET].agg(["sum", "size"])
frauded = per_uid[per_uid["sum"] > 0]
print()
print(f"uids with >=1 fraud    : {len(frauded):,} ({len(frauded)/len(per_uid):.2%} of accounts)")
print(f"share of all fraud in repeat-fraud accounts: "
      f"{frauded.loc[frauded['sum'] > 1, 'sum'].sum() / max(n_fraud,1):.1%}")
print()
print("A high number on that last line means account identity is directly")
print("predictive -- exactly the signal a random split leaks.")

# %% [markdown]
# ## Q4 - What is missing, and why?

# %%
miss = df.isna().mean().sort_values(ascending=False)
print(f"columns with any missing: {(miss > 0).sum()} / {df.shape[1]}")
print(miss.head(15).to_frame("pct_missing"))

# %%
# The key test: is missingness itself predictive? If yes, imputing to a
# median destroys information, and "was it missing" deserves to be a feature.
cands = [c for c in miss[(miss > 0.05) & (miss < 0.95)].index[:12]]
rows = []
for c in cands:
    m = df[c].isna()
    rows.append({"column": c, "pct_missing": m.mean(),
                 "fraud_when_missing": df.loc[m, C.TARGET].mean(),
                 "fraud_when_present": df.loc[~m, C.TARGET].mean()})
mt = pd.DataFrame(rows)
mt["ratio"] = mt["fraud_when_missing"] / mt["fraud_when_present"].replace(0, np.nan)
print(mt.sort_values("ratio", ascending=False).to_string(index=False))

print()
print("Any ratio far from 1.0 means missingness carries signal. That is why")
print("the baseline pipeline uses SimpleImputer(add_indicator=True) rather")
print("than a bare fillna.")

# %% [markdown]
# ## Q5 - Cardinality, which decides encoding strategy

# %%
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
card = df[cat_cols].nunique().sort_values(ascending=False)
print(card.to_frame("n_unique"))
print()
print("Anything above ~25 levels cannot be one-hot encoded for a linear model")
print("without handing it a per-value lookup table. Those columns are dropped")
print("from the baseline and return on day 9-15 as count / out-of-fold target")
print("encodings.")

# %% [markdown]
# ## Takeaways to carry into notebook 02
#
# - Fraud is rare -> report **PR-AUC** and **recall at a fixed FPR**, never accuracy.
# - The data is time-ordered and **drifting** -> validation must be a *future* block.
# - Accounts **recur** and fraud concentrates in them -> a shuffled split leaks identity.
# - Missingness is **informative** -> keep indicators, never blind-fill.
#
# Next notebook measures exactly how much a random split inflates the score.
