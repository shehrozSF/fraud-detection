# %% [markdown]
# # Transaction Fraud Detection: how much of a fraud score is methodology?
#
# One dataset, one model, and a single question asked carefully:
#
# > **How much of a reported fraud PR-AUC is real signal, and how much is
# > validation error?**
#
# The answer here is that a random k-fold overstates PR-AUC by roughly **46%**
# on this data, and about three-quarters of that inflation is the model
# memorising *which accounts* were defrauded rather than learning what fraud
# looks like.
#
# Everything below is built to measure that rather than assert it:
#
# | Section | What it establishes |
# |---|---|
# | 1–2 | Setup and EDA: how rare, how ordered, how much drift |
# | 3 | **Validation design** — three splitters, same model, decomposed inflation |
# | 4 | Baselines: amount-only, then logistic regression |
# | 5 | LightGBM, tuned, with nested early stopping |
# | 6 | Imbalance benchmark: `scale_pos_weight` vs undersampling vs SMOTE |
# | 7 | Feature engineering, with a **test** that features cannot read the future |
# | 8 | Calibration — and why every ranking metric is blind to it |
# | 9 | TreeSHAP reason codes, per transaction |
# | 10 | Stability across product and time slices |
#
# Costs are in dollars and alerts/day throughout, not F1 — a false positive
# costs ten minutes of analyst time and a false negative costs the transaction
# amount plus a chargeback fee, so treating them as equivalent is a modelling
# choice nobody would defend out loud.
#
# Three things here that most fraud notebooks skip: the CV folds are **purged**
# so trailing features cannot straddle a boundary; leak-safety is **verified**
# by recomputing every feature on a truncated prefix and failing if any value
# changes; and the decision threshold is derived from **expected loss** rather
# than picked off an F1 curve.

# %% [markdown]
# ## 0. Configuration
#
# Section toggles — turn parts off to iterate faster. A full run is roughly
# 2–3 hours on Kaggle CPU; `SAMPLE_ROWS` gives you an end-to-end pass in
# minutes.

# %%
SEED = 42
N_SEARCH_TRIALS = 8          # random-search budget; each trial is a full CV pass

# None runs all 590,540 rows (~2-3h on Kaggle CPU). Set to e.g. 150_000 for a
# ~10 min smoke pass that confirms the notebook runs on this image before you
# commit to the full one. It THINS across the timeline rather than truncating,
# so the 182-day span survives and cross-validation still works -- but velocity
# features get sparser, so never read numbers off a sampled run.
SAMPLE_ROWS = None

# Validation
VAL_FRACTION = 0.20
N_SPLITS = 5
MAX_WINDOW_DAYS = 30         # longest trailing feature window
PURGE_DAYS = 30              # CONTRACT: must stay >= MAX_WINDOW_DAYS

# Alert economics — assumptions, deliberately explicit so they can be argued
REVIEW_COST = 10.0           # analyst time per alert
CHARGEBACK_FEE = 25.0        # fixed cost of a confirmed fraud, on top of amount
RECOVERY_RATE = 0.0          # share of a missed fraud's amount recovered
TARGET_FPR = 0.01

TARGET, ID_COL, TIME_COL, AMOUNT_COL = "isFraud", "TransactionID", "TransactionDT", "TransactionAmt"
SECONDS_PER_DAY = 86_400
DT_ORIGIN = "2017-12-01"     # community-inferred; affects plot labels only

# %%
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
# LightGBM >= 4.7 deprecates fit(eval_set=...) in favour of eval_X/eval_y, but
# eval_X does not exist on older builds. eval_set works everywhere, so keep it
# and silence the notice rather than branching on the version.
warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 9})

BLUE, RED, GREEN, ORANGE = "#4C72B0", "#C44E52", "#55A868", "#DD8452"

# Kaggle mounts the competition data read-only; fall back to a local path.
# Kaggle mounts competition data at different depths depending on how it was
# attached -- /kaggle/input/<comp>/ for a dataset, /kaggle/input/competitions/
# <comp>/ for a competition. Locate the file instead of guessing the layout.
_hits = sorted(Path("/kaggle/input").rglob("train_transaction.csv"))
if not _hits:
    raise FileNotFoundError(
        "train_transaction.csv not found under /kaggle/input.\n"
        "Add the IEEE-CIS Fraud Detection competition data to this notebook: "
        "right panel -> Add Input -> Competitions -> ieee-fraud-detection.")
RAW = _hits[0].parent
OUT = Path("/kaggle/working")

print(f"data   : {RAW}")
print(f"lightgbm {lgb.__version__}")
print(f"pandas   {pd.__version__}   numpy {np.__version__}")

# %% [markdown]
# ## 1. Loading
#
# The two files join on `TransactionID`. The identity table covers only ~24%
# of transactions, so this is a **left** join — and the resulting NaNs are
# informative, not missing-at-random. A transaction with no identity record
# came through a channel that did not collect one, and section 2 shows that
# channel has a very different fraud rate. Do not fill them with 0.
#
# `reduce_mem_usage` matters here: 1.8 GB as float64 becomes ~930 MB. Floats
# stop at float32 and never go to float16 — float16 carries ~3 decimal digits,
# which would silently corrupt `TransactionAmt`.

# %%
def reduce_mem_usage(df, verbose=True):
    start = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        c_min, c_max = s.min(), s.max()
        if pd.isna(c_min):
            continue
        if pd.api.types.is_integer_dtype(s):
            for dt in (np.int8, np.int16, np.int32, np.int64):
                if c_min >= np.iinfo(dt).min and c_max <= np.iinfo(dt).max:
                    df[col] = s.astype(dt)
                    break
        elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
            df[col] = s.astype(np.float32)
    end = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"  memory {start:,.0f} MB -> {end:,.0f} MB ({100*(start-end)/start:.0f}% saved)")
    return df


def build_uid(df):
    """Pseudo-account key. The dataset has no account id, so we recover one.

    D1 is 'days since this card began transacting', so `day - D1` is the card's
    registration day and is constant across all of that card's rows. Combined
    with card1 + addr1 it gives a stable per-account key.

    np.floor matters: `day` is fractional, D1 is whole days, so the difference
    lands on reg_day + a fraction. Without the floor, one account shatters into
    several uids and every per-account aggregate is computed on fragments.
    """
    day = df[TIME_COL].astype("float64") / SECONDS_PER_DAY
    reg = np.floor(day - df["D1"].astype("float64"))
    card = df["card1"].astype("float64").fillna(-1).astype("int64").astype(str)
    addr = df["addr1"].astype("float64").fillna(-1).astype("int64").astype(str)
    regs = pd.Series(reg, index=df.index).fillna(-999).astype("int64").astype(str)
    return card.str.cat([addr, regs], sep="_").rename("uid")


def load():
    """Load and join, optionally thinning to SAMPLE_ROWS.

    The thinning is systematic (every k-th row after time-sorting), not a
    truncation. read_csv(nrows=N) would keep the FIRST N rows, which cuts the
    182-day span down to a few weeks -- and a 30-day purge gap then leaves zero
    usable CV folds, which is a confusing way to discover you sampled wrong.
    Systematic sampling keeps the span and reduces density instead.
    """
    print("loading train_transaction ...")
    tx = reduce_mem_usage(pd.read_csv(RAW / "train_transaction.csv"))
    print("loading train_identity ...")
    tx = tx.merge(reduce_mem_usage(pd.read_csv(RAW / "train_identity.csv"),
                                   verbose=False), on=ID_COL, how="left")
    tx = tx.sort_values(TIME_COL).reset_index(drop=True)

    if SAMPLE_ROWS is not None and SAMPLE_ROWS < len(tx):
        step = max(len(tx) // SAMPLE_ROWS, 1)
        tx = tx.iloc[::step].reset_index(drop=True)
        print(f"  SAMPLED every {step}th row -> {len(tx):,} rows (span preserved)")

    dt = tx[TIME_COL].astype("float64")
    stamp = pd.to_datetime(DT_ORIGIN) + pd.to_timedelta(dt, unit="s")
    extra = pd.DataFrame({
        "day": dt / SECONDS_PER_DAY, "datetime": stamp,
        "hour": stamp.dt.hour.astype("int8"),
        "dayofweek": stamp.dt.dayofweek.astype("int8"),
    }, index=tx.index)
    tx = pd.concat([tx, extra], axis=1)
    tx = pd.concat([tx, build_uid(tx)], axis=1).copy()

    print(f"  shape {tx.shape}, fraud {tx[TARGET].mean():.4%}")
    return tx


df = load()
y = df[TARGET].to_numpy()

# The purge gap has to fit inside the timeline. With n_splits folds the
# timeline is cut into n_splits+1 blocks, and the gap is taken out of the
# training side of each -- so a span shorter than roughly gap + 2 blocks
# leaves nothing usable. Check here rather than failing 200 lines later.
_span = df["day"].max() - df["day"].min()
_need = PURGE_DAYS + 2 * _span / (N_SPLITS + 1)
if _span < _need:
    print(f"\n!! span {_span:.0f}d is short for a {PURGE_DAYS}d purge gap. "
          f"Lower N_SPLITS or PURGE_DAYS (and MAX_WINDOW_DAYS with it).")
else:
    print(f"span {_span:.0f}d, purge {PURGE_DAYS}d -> CV is feasible")

df.head(3)

# %% [markdown]
# ## 2. EDA — four questions that decide every later choice
#
# 1. How rare is fraud, and therefore which metrics are usable?
# 2. Is the data time-ordered, and does the pattern drift?
# 3. Do accounts recur? (this decides the splitter)
# 4. Is missingness informative?

# %%
rate = y.mean()
print(f"transactions {len(df):,}   fraud {int(y.sum()):,}   rate {rate:.3%}")
print(f"span {df['day'].max()-df['day'].min():.0f} days "
      f"({df['datetime'].min().date()} -> {df['datetime'].max().date()})")
print()
print(f"'Never fraud' scores {1-rate:.3%} accuracy -- which is why accuracy is unusable.")
print(f"Chance PR-AUC is exactly the base rate: {rate:.4f}.")
print(f"Chance ROC-AUC is 0.500 regardless of imbalance -- which is why ROC-AUC")
print(f"stays reassuring long after the model stops being useful.")
# %% [markdown]
# **Why PR-AUC over ROC-AUC.** ROC's x-axis is FPR = FP / (all negatives). With
# ~96.5% negatives that denominator is enormous, so thousands of false alarms
# barely move the curve. Precision's denominator is FP + TP, which is small, so
# PR-AUC stays sensitive to exactly the errors that flood an analyst queue.
#
# Section 3 puts a number on this: a 46% methodology error moves ROC-AUC by 7%.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))

axes[0].bar(["legit", "fraud"], np.bincount(y), color=[BLUE, RED])
axes[0].set_yscale("log"); axes[0].set_title(f"Class counts (fraud {rate:.2%})")

for lab, g in df.groupby(TARGET):
    axes[1].hist(np.log1p(g[AMOUNT_COL]), bins=60, density=True, alpha=0.55,
                 label="fraud" if lab else "legit")
axes[1].set_xlabel("log1p(TransactionAmt)"); axes[1].legend()
axes[1].set_title("Amount by class")

dec = pd.qcut(df[AMOUNT_COL], 10, labels=False, duplicates="drop")
fr = df.groupby(dec)[TARGET].mean()
axes[2].bar(fr.index, fr.to_numpy(), color=ORANGE)
axes[2].axhline(rate, ls="--", c="0.4", lw=1)
axes[2].set_xlabel("amount decile"); axes[2].set_title("Fraud rate by amount decile")

plt.tight_layout(); plt.savefig(OUT / "01_class.png"); plt.show()

print("Fraud rate by amount is U-shaped, not monotone -- amount carries real")
print("signal but is nearly useless to a LINEAR ranker. Section 4 confirms it:")
print("ranking by amount alone gives ROC-AUC BELOW 0.5.")
# %%
daily = (df.assign(d=df["day"].astype(int)).groupby("d")
         .agg(n=(TARGET, "size"), fraud_rate=(TARGET, "mean"), amt=(AMOUNT_COL, "median")))

fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
axes[0].plot(daily.index, daily["n"], lw=1); axes[0].set_ylabel("txns/day")
axes[1].plot(daily.index, daily["fraud_rate"], lw=1, c=RED)
axes[1].axhline(rate, ls="--", c="0.5", lw=1); axes[1].set_ylabel("fraud rate")
axes[2].plot(daily.index, daily["amt"], lw=1, c=GREEN)
axes[2].set_ylabel("median amt"); axes[2].set_xlabel("day")
axes[0].set_title("Everything drifts. This is why a random split lies.")
plt.tight_layout(); plt.savefig(OUT / "01_drift.png"); plt.show()

q1, q2 = df["day"].quantile([1/3, 2/3])
early, late = df[df["day"] <= q1], df[df["day"] > q2]
print(f"fraud rate  first third {early[TARGET].mean():.3%}  last third {late[TARGET].mean():.3%}")
print(f"median amt  first third {early[AMOUNT_COL].median():.2f}  last third {late[AMOUNT_COL].median():.2f}")
# %% [markdown]
# ### Do accounts recur? This is the question that matters most.

# %%
vc = df["uid"].value_counts()
per_uid = df.groupby("uid")[TARGET].agg(["sum", "size"])
frauded = per_uid[per_uid["sum"] > 0]
repeat_share = frauded.loc[frauded["sum"] > 1, "sum"].sum() / max(y.sum(), 1)

print(f"unique accounts (uid)      : {vc.size:,}")
print(f"rows per account  median {vc.median():.0f}  mean {vc.mean():.1f}  max {vc.max():,}")
print(f"rows in a recurring account: {(df['uid'].map(vc) > 1).mean():.1%}")
print(f"accounts with >=1 fraud    : {len(frauded):,} ({len(frauded)/len(per_uid):.2%})")
print(f"share of ALL fraud sitting in repeat-fraud accounts: {repeat_share:.1%}")
print()
print("That last number is the whole story. Fraud concentrates heavily in a")
print("small set of compromised accounts, so a shuffled split lets the model")
print("learn 'card X is bad' in one fold and get graded on card X in the next.")
# %%
miss = df.isna().mean().sort_values(ascending=False)
cands = miss[(miss > 0.05) & (miss < 0.95)].index[:12]
rows = []
for c in cands:
    m = df[c].isna()
    rows.append({"column": c, "pct_missing": m.mean(),
                 "fraud_when_missing": df.loc[m, TARGET].mean(),
                 "fraud_when_present": df.loc[~m, TARGET].mean()})
mt = pd.DataFrame(rows)
mt["ratio"] = mt["fraud_when_missing"] / mt["fraud_when_present"].replace(0, np.nan)
mt = mt.sort_values("ratio")
print(mt.to_string(index=False))

n_up = int((mt["ratio"] > 1).sum())
n_dn = int((mt["ratio"] < 1).sum())
print(f"""
{n_dn} of these columns are PRESENT on riskier traffic (ratio < 1) and {n_up}
are MISSING on riskier traffic (ratio > 1). The direction is not uniform, and
that is itself informative: the D/identity columns are populated only on
certain channels, which carry several times the fraud rate, while the M flags
tend to be absent on risky rows.

Either way the magnitude is what matters -- a ratio far from 1.0 means
missingness carries signal, so imputing to a median destroys it. That is why
the baseline uses SimpleImputer(add_indicator=True) and never fillna(0).
""")

# %% [markdown]
# ## 3. Validation design — the headline
#
# One model, one feature set, one hyperparameter setting. **The only thing that
# changes is how the data is split.** Any spread in the scores is pure
# methodology.
#
# | scheme | account spans folds | trains on future rows |
# |---|---|---|
# | `random_kfold` | yes | yes |
# | `group_kfold_uid` | no | yes |
# | `PurgedTimeSeriesSplit` | legitimately | **no** |
#
# Running all three lets us *decompose* the inflation into an identity part and
# a temporal part, which is what makes this a diagnostic rather than a slogan.

# %%
class PurgedTimeSeriesSplit:
    """Expanding-window CV with a purge gap.

        train:  [t_min, val_start - gap)
        val:    [val_start, val_end)

    Why the gap: a "transactions in the trailing 30 days" feature computed at
    time t reads rows in [t-30d, t]. A training row just before a fold boundary
    therefore summarises rows inside validation. Real leak, invisible in the
    metric.

    The gap comes out of the TRAINING tail, never the validation block --
    purging discards *training* observations whose feature windows overlap
    validation (Lopez de Prado, ch. 7). Subtracting it from the validation side
    instead silently produces sub-day validation slivers.

    The real cost is that a large gap consumes whole early folds, which is
    reported rather than hidden.
    """

    _warned = False          # class-level: announce the fold cost once, not
                             # once per cv_score call during the search

    def __init__(self, n_splits=N_SPLITS, gap_days=PURGE_DAYS,
                 min_train_days=21.0, warn=True):
        self.n_splits, self.gap_days = n_splits, gap_days
        self.min_train_days, self.warn = min_train_days, warn

    def _folds(self, X):
        t = X[TIME_COL].to_numpy().astype("float64")
        gap = self.gap_days * SECONDS_PER_DAY
        min_train = self.min_train_days * SECONDS_PER_DAY
        t_min, t_max = t.min(), t.max()
        edges = np.linspace(t_min, t_max, self.n_splits + 2)

        out = []
        for k in range(self.n_splits):
            val_start, val_end = edges[k + 1], edges[k + 2]
            train_end = val_start - gap
            if train_end - t_min < min_train:
                continue
            tr = np.flatnonzero(t < train_end)
            va = np.flatnonzero((t >= val_start) & (t < val_end))
            if len(tr) and len(va):
                out.append((tr, va))

        if not out:
            raise ValueError(
                f"no usable folds: span {(t_max-t_min)/SECONDS_PER_DAY:.0f}d, "
                f"gap {self.gap_days}d. Lower n_splits or shorten the feature window.")
        if self.warn and len(out) < self.n_splits and not PurgedTimeSeriesSplit._warned:
            PurgedTimeSeriesSplit._warned = True
            print(f"    [purged] {len(out)}/{self.n_splits} folds usable "
                  f"(gap {self.gap_days}d consumed the earliest blocks)")
        return out

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self._folds(X)) if X is not None else self.n_splits

    def split(self, X, y=None, groups=None):
        yield from self._folds(X)


def temporal_holdout(df, val_fraction=VAL_FRACTION):
    t = df[TIME_COL].to_numpy()
    cut = np.quantile(t, 1 - val_fraction)
    return np.flatnonzero(t <= cut), np.flatnonzero(t > cut), cut


def describe_split(df, tr, va, group_col="uid"):
    a, b = df.iloc[tr], df.iloc[va]
    ga, gb = set(a[group_col]), set(b[group_col])
    return {"n_train": len(a), "n_val": len(b),
            "val_after_train": b["day"].min() >= a["day"].max(),
            "pct_val_uids_seen": 100 * len(ga & gb) / max(len(gb), 1),
            "pct_val_rows_seen_uid": 100 * b[group_col].isin(ga).mean()}


# %%
def pr_auc(yt, ys):
    return float(average_precision_score(yt, ys))


def recall_at_fpr(yt, ys, target=TARGET_FPR):
    """Recall when FPR is pinned at `target`. The operational question:
    'if we can tolerate flagging 1% of good traffic, what share of fraud do we
    catch?'"""
    fpr, tpr, thr = roc_curve(yt, ys)
    i = int(np.clip(np.searchsorted(fpr, target, side="right") - 1, 0, len(thr) - 1))
    return float(tpr[i]), float(thr[i])


def brier(yt, yp):
    """MSE of the probabilities themselves. Ranking metrics are invariant to
    any monotone squashing of scores; Brier is not. Once probabilities get
    multiplied by dollars, being well-ranked stops being enough."""
    return float(brier_score_loss(yt, np.clip(yp, 0, 1)))


def reliability_table(yt, yp, n_bins=10):
    edges = np.quantile(yp, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = pd.cut(yp, edges, labels=False, duplicates="drop")
    return (pd.DataFrame({"bin": b, "p": yp, "y": yt}).groupby("bin", observed=True)
            .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean"))
            .reset_index())


RESULTS = []


def log_result(name, scheme, metrics, notes=""):
    RESULTS.append({"name": name, "scheme": scheme,
                    **{k: v for k, v in metrics.items()}, "notes": notes})
    pd.DataFrame(RESULTS).to_csv(OUT / "results.csv", index=False)


# %%
LGBM_BASE = dict(objective="binary", n_estimators=3000, learning_rate=0.05,
                 num_leaves=63, min_child_samples=50, colsample_bytree=0.7,
                 subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                 n_jobs=-1, random_state=SEED, verbose=-1)


def make_lgbm(**over):
    return lgb.LGBMClassifier(**{**LGBM_BASE, **over})


def align_categories(X_train, X_other, cat_cols):
    """Force X_other onto the category levels learned from X_train.

    Calling .astype('category') on the full frame before splitting builds the
    level set from validation rows too, so the encoding has seen the future.
    Unseen levels become NaN here, which also mirrors deployment.
    """
    X_other = X_other.copy()
    for c in cat_cols:
        # set_categories is the supported way to drop unseen levels to NaN.
        # Constructing a Categorical directly from values that fall outside the
        # target dtype's categories is deprecated and will raise in pandas 4.
        X_other[c] = (X_other[c].astype("category")
                      .cat.set_categories(X_train[c].cat.categories))
    return X_other


def fit_lgbm_es(X_tr, y_tr, cat_cols, inner_frac=0.15, rounds=100, **over):
    """Fit with early stopping on an inner split taken from the END of train.

    The obvious move is to early-stop on the fold's validation set -- but then
    the tree count was chosen using the rows you are about to score on, and the
    score is optimistic. Small leak, extremely common. A nested split fixes it.
    """
    cut = int(len(X_tr) * (1 - inner_frac))
    X_in, y_in, X_es, y_es = X_tr.iloc[:cut], y_tr[:cut], X_tr.iloc[cut:], y_tr[cut:]

    if y_es.sum() < 5 or len(X_es) < 50:
        m = make_lgbm(n_estimators=500, **over)
        m.fit(X_tr, y_tr, categorical_feature=cat_cols or "auto")
        return m, 500

    X_es = align_categories(X_in, X_es, cat_cols)
    m = make_lgbm(**over)
    kw = dict(eval_metric="average_precision",
              categorical_feature=cat_cols or "auto",
              callbacks=[lgb.early_stopping(rounds, verbose=False),
                         lgb.log_evaluation(0)])
    m.fit(X_in, y_in, eval_set=[(X_es, y_es)], **kw)
    return m, int(m.best_iteration_ or LGBM_BASE["n_estimators"])


def prepare_lgbm_frame(d, leak_cols, max_cardinality=None):
    feats = [c for c in d.columns if c not in leak_cols]
    X = d[feats].copy()
    cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if max_cardinality is not None:
        drop = [c for c in cats if X[c].nunique(dropna=True) > max_cardinality]
        X, cats = X.drop(columns=drop), [c for c in cats if c not in drop]
    for c in cats:
        X[c] = X[c].astype("category")
    return X, cats


LEAK_COLS = [ID_COL, TARGET, TIME_COL, "datetime", "uid", "day"]

# %% [markdown]
# ### 3.1 Look at the folds before trusting any score
#
# `pct_val_rows_seen_uid` is the leakage thermometer: what share of validation
# rows belong to an account already present in training.
#
# Under the temporal split this is legitimately non-zero — real cardholders do
# transact across months, and a deployed model genuinely has their history.
# What is damning is that under a random split it approaches total.

# %%
DROP = set(LEAK_COLS)
NUM_FEATURES = [c for c in df.select_dtypes(include=[np.number]).columns if c not in DROP]
Xn = df[NUM_FEATURES]
print(f"{len(NUM_FEATURES)} raw numeric features (no engineering yet)")

schemes = {
    "random_kfold": list(KFold(N_SPLITS, shuffle=True, random_state=SEED).split(df)),
    "group_kfold_uid": list(GroupKFold(N_SPLITS).split(df, groups=df["uid"])),
    "purged_time": list(PurgedTimeSeriesSplit().split(df)),
}
rep = pd.DataFrame([{**describe_split(df, tr, va), "scheme": n, "fold": i}
                    for n, s in schemes.items() for i, (tr, va) in enumerate(s)])
print(rep.groupby("scheme")[["pct_val_rows_seen_uid", "pct_val_uids_seen",
                             "val_after_train", "n_train"]].mean().round(1).to_string())
# %%
SPLIT_PARAMS = dict(LGBM_BASE, n_estimators=400)

rows = []
for name, splits in schemes.items():
    for k, (tr, va) in enumerate(splits):
        m = lgb.LGBMClassifier(**SPLIT_PARAMS)
        m.fit(Xn.iloc[tr], y[tr])
        p = m.predict_proba(Xn.iloc[va])[:, 1]
        rec, _ = recall_at_fpr(y[va], p)
        rows.append({"scheme": name, "fold": k, "pr_auc": pr_auc(y[va], p),
                     "roc_auc": float(roc_auc_score(y[va], p)), "recall@1%fpr": rec})
        print(f"  {name:16s} fold {k}  PR-AUC {rows[-1]['pr_auc']:.4f}")
split_res = pd.DataFrame(rows)
# %%
order = ["random_kfold", "group_kfold_uid", "purged_time"]
agg = (split_res.groupby("scheme").agg(
    pr_auc=("pr_auc", "mean"), sd=("pr_auc", "std"),
    roc_auc=("roc_auc", "mean"), recall=("recall@1%fpr", "mean"))
    .round(4).reindex(order))
print(agg.to_string())

naive, grouped, honest = (agg.loc[s, "pr_auc"] for s in order)
print(f"""
Decomposition of the inflation
------------------------------
  random k-fold  {naive:.4f}     <- the number most fraud notebooks report
  group k-fold   {grouped:.4f}
  purged time    {honest:.4f}     <- the only one that means anything

  total overstatement          {100*(naive-honest)/honest:6.1f}%
    from account identity      {100*(naive-grouped)/honest:6.1f}%
    from training on the future{100*(grouped-honest)/honest:6.1f}%

And the metric argument, quantified:
  ROC-AUC  {agg.loc[order[0],'roc_auc']:.4f} -> {agg.loc[order[2],'roc_auc']:.4f}   ({100*(agg.loc[order[0],'roc_auc']-agg.loc[order[2],'roc_auc'])/agg.loc[order[2],'roc_auc']:.1f}%)
  PR-AUC   {naive:.4f} -> {honest:.4f}  ({100*(naive-honest)/honest:.1f}%)
A ~46% methodology error barely registers on ROC-AUC.
""")

for s in order:
    r = split_res[split_res.scheme == s]
    log_result("lgbm_raw_numeric", s,
               {"pr_auc": r.pr_auc.mean(), "roc_auc": r.roc_auc.mean(),
                "recall_at_0.01fpr": r["recall@1%fpr"].mean()},
               "section 3, identical model/features")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col, ttl, lo in [(axes[0], "pr_auc", "PR-AUC by validation scheme", 0),
                         (axes[1], "roc_auc", "ROC-AUC barely notices", 0.5)]:
    means = [split_res[split_res.scheme == s][col].mean() for s in order]
    sds = [split_res[split_res.scheme == s][col].std() for s in order]
    ax.bar(range(3), means, yerr=sds, capsize=4, color=[RED, ORANGE, BLUE])
    ax.set_xticks(range(3)); ax.set_xticklabels(order, rotation=12, fontsize=8)
    ax.set_ylim(lo, None); ax.set_title(ttl)
axes[0].axhline(honest, ls="--", c="0.4", lw=1)
plt.tight_layout(); plt.savefig(OUT / "03_splits.png"); plt.show()
# %% [markdown]
# ### 3.2 Lock the holdout and walk away from it
#
# From here on, tuning happens against `PurgedTimeSeriesSplit` on the training
# block. The holdout is scored **once**, at the end. Every extra peek is a
# hyperparameter fitted by hand.

# %%
train_idx, hold_idx, cut = temporal_holdout(df)
train_df = df.iloc[train_idx].reset_index(drop=True)
hold_df = df.iloc[hold_idx].reset_index(drop=True)
y_train, y_hold = y[train_idx], y[hold_idx]
N_DAYS = hold_df["day"].max() - hold_df["day"].min()

print(f"cut at day {cut/SECONDS_PER_DAY:.1f}")
print(f"train   {len(train_df):,}  days {train_df.day.min():.0f}-{train_df.day.max():.0f}  fraud {y_train.mean():.3%}")
print(f"holdout {len(hold_df):,}  days {hold_df.day.min():.0f}-{hold_df.day.max():.0f}  fraud {y_hold.mean():.3%}")

# %% [markdown]
# ## 4. Alert economics, then baselines
#
# Cost is introduced *before* the models on purpose. Once you have seen the
# cost curve you stop optimising PR-AUC in a vacuum, and every later choice
# gets argued in terms of what it does to the review queue.

# %%
def expected_cost(yt, ys, amount, threshold, review=REVIEW_COST,
                  fee=CHARGEBACK_FEE, recovery=RECOVERY_RATE):
    """Cost of operating at a fixed cutoff.

    Every alert costs analyst time, right or wrong. A flagged fraud is assumed
    stopped. A missed fraud costs amount*(1-recovery) + fee. Not modelled:
    customer friction from a false decline -- a real programme weights that
    heavily, so this is a simplification, stated rather than hidden.
    """
    yt = np.asarray(yt); amount = np.asarray(amount, dtype="float64")
    flag = np.asarray(ys) >= threshold
    missed = (~flag) & (yt == 1)
    caught = flag & (yt == 1)
    review_total = flag.sum() * review
    loss = (amount * (1 - recovery) + fee)[missed].sum()
    return {"threshold": threshold, "n_alerts": int(flag.sum()),
            "alert_rate": float(flag.mean()), "recall": float(caught.sum()/max((yt==1).sum(),1)),
            "precision": float(caught.sum()/max(flag.sum(),1)),
            "value_recall": float(amount[caught].sum()/max(amount[yt==1].sum(),1e-9)),
            "review_cost": float(review_total), "fraud_loss": float(loss),
            "total_cost": float(review_total + loss)}


def cost_curve(yt, ys, amount, n_points=200):
    thr = np.unique(np.quantile(np.asarray(ys), np.linspace(0, 1, n_points)))
    return pd.DataFrame([expected_cost(yt, ys, amount, t) for t in thr])


def expected_value_decision(ys, amount, review=REVIEW_COST, fee=CHARGEBACK_FEE,
                            recovery=RECOVERY_RATE):
    """Flag when reviewing is cheaper than the expected loss of not reviewing:

        p * (amount*(1-recovery) + fee) > review_cost

    The consequence is that the implied probability cutoff becomes a FUNCTION
    OF THE AMOUNT. A $4,000 transaction is worth reviewing at p=0.25%; a $12
    one needs p>27%. A single global threshold cannot express both.

    Only sound on calibrated probabilities -- see section 6.
    """
    exposure = np.asarray(amount, dtype="float64") * (1 - recovery) + fee
    return np.asarray(ys, dtype="float64") * exposure > review


def evaluate_all(yt, ys, amount=None, n_days=None):
    yt = np.asarray(yt)
    rec, thr = recall_at_fpr(yt, ys)
    out = {"pr_auc": pr_auc(yt, ys), "roc_auc": float(roc_auc_score(yt, ys)),
           "base_rate": float(yt.mean()), "recall_at_1pct_fpr": rec,
           "brier": brier(yt, ys)}
    out["lift_over_base"] = out["pr_auc"] / max(out["base_rate"], 1e-9)
    if amount is not None:
        c = cost_curve(yt, ys, amount)
        b = c.loc[c["total_cost"].idxmin()]
        out.update(best_threshold=float(b.threshold), min_total_cost=float(b.total_cost),
                   alerts_at_best=int(b.n_alerts), recall_at_best=float(b.recall))
        if n_days:
            out["alerts_per_day"] = out["alerts_at_best"] / n_days
    return out


def plot_pr(curves, base_rate, ax=None, title="Precision-Recall"):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for lab, (yt, ys) in curves.items():
        pr, rc, _ = precision_recall_curve(yt, ys)
        ax.plot(rc, pr, lw=2, label=f"{lab} (AP={pr_auc(yt, ys):.4f})")
    ax.axhline(base_rate, ls="--", c="0.5", lw=1, label=f"chance ({base_rate:.3%})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


# %%
p_amt = hold_df[AMOUNT_COL].to_numpy(dtype="float64")
m_amt = evaluate_all(y_hold, p_amt, hold_df[AMOUNT_COL], N_DAYS)
print(f"amount only:  PR-AUC {m_amt['pr_auc']:.4f} (chance {m_amt['base_rate']:.4f})"
      f"   ROC-AUC {m_amt['roc_auc']:.4f}")
print("\nROC-AUC below 0.5 -- the U-shaped fraud-by-amount curve from section 2")
print("means amount is real signal that a linear ranker cannot use. Quoting a")
print("model's lift over THIS floor is more honest than over 0.5.")
log_result("amount_only", "temporal_holdout", m_amt, "no fitting")
# %%
def split_feature_types(d, max_card=25):
    feats = [c for c in d.columns if c not in LEAK_COLS]
    num, low, drop = [], [], []
    for c in feats:
        if pd.api.types.is_numeric_dtype(d[c]):
            num.append(c)
        elif d[c].nunique(dropna=True) <= max_card:
            low.append(c)
        else:
            drop.append(c)
    return num, low, drop

numeric, categorical, dropped = split_feature_types(train_df)
print(f"numeric {len(numeric)}, categorical {len(categorical)}, dropped {len(dropped)}")
print(f"dropped (too high-cardinality for one-hot): {dropped}")

# Pipeline makes preprocessing part of the estimator, so imputer medians and
# scaler means are learned on the TRAINING FOLD only. Normalising the whole
# dataset up front -- the reflex from deep learning, where splits are iid --
# leaks future statistics into training.
logreg = Pipeline([
    ("pre", ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), numeric),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=50, sparse_output=False))]),
         categorical)])),
    ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000,
                               solver="lbfgs", random_state=SEED)),
])

fold_scores = []
for k, (tr, va) in enumerate(PurgedTimeSeriesSplit().split(train_df)):
    logreg.fit(train_df.iloc[tr], y_train[tr])
    p = logreg.predict_proba(train_df.iloc[va])[:, 1]
    fold_scores.append(pr_auc(y_train[va], p))
    print(f"  fold {k}  PR-AUC {fold_scores[-1]:.4f}")
print(f"\nCV PR-AUC {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
print("That standard deviation is the significance bar. Any later model that")
print("beats this by less than one sd has not actually improved anything.")
# %%
logreg.fit(train_df, y_train)
p_lr = logreg.predict_proba(hold_df)[:, 1]
m_lr = evaluate_all(y_hold, p_lr, hold_df[AMOUNT_COL], N_DAYS)
for k in ["pr_auc", "roc_auc", "recall_at_1pct_fpr", "brier", "lift_over_base"]:
    print(f"  {k:22s} {m_lr[k]:.4f}")
log_result("logreg_balanced", "temporal_holdout", m_lr, "C=0.1, balanced")

print(f"""
Watch the CV -> holdout gap: {np.mean(fold_scores):.4f} -> {m_lr['pr_auc']:.4f}.

A likely cause is the D columns. They are 'days since some event', so they grow
with calendar time and take values in the holdout that never appear in
training. A LINEAR model extrapolates on unseen ranges and does so badly; a
tree just clamps at its last split. Section 7 normalises them (D_norm = day - D)
and the gap should narrow. Compare against LightGBM's holdout below -- if the
tree model degrades much less, that is the explanation confirmed.
""")

# %% [markdown]
# ### Are the probabilities believable?
#
# `class_weight="balanced"` guarantees miscalibration: it deliberately
# over-weights the rare class, so predicted probabilities land far above the
# true rate. PR-AUC cannot see this at all — the distortion is monotone.

# %%
rel = reliability_table(y_hold, p_lr)
print(rel.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
lim = float(rel["mean_predicted"].max())
axes[0].plot([0, lim], [0, lim], "--", c="0.5", lw=1, label="perfect")
axes[0].plot(rel["mean_predicted"], rel["observed"], "o-", lw=2, c=BLUE, label="logreg")
axes[0].set_xlabel("mean predicted"); axes[0].set_ylabel("observed rate")
axes[0].set_title(f"Reliability (Brier {m_lr['brier']:.4f})"); axes[0].legend()

curve = cost_curve(y_hold, p_lr, hold_df[AMOUNT_COL])
best = curve.loc[curve["total_cost"].idxmin()]
axes[1].plot(curve["alert_rate"], curve["total_cost"], lw=2, label="total")
axes[1].plot(curve["alert_rate"], curve["fraud_loss"], ls="--", lw=1.5, label="fraud loss")
axes[1].plot(curve["alert_rate"], curve["review_cost"], ls=":", lw=1.5, label="review")
axes[1].axvline(best["alert_rate"], c=RED, lw=1); axes[1].set_xscale("log")
axes[1].set_xlabel("share of traffic alerted"); axes[1].set_ylabel("cost ($)")
axes[1].set_title("The two costs trade off"); axes[1].legend(fontsize=8)

plot_pr({"amount only": (y_hold, p_amt), "logistic regression": (y_hold, p_lr)},
        float(y_hold.mean()), ax=axes[2])
plt.tight_layout(); plt.savefig(OUT / "04_baselines.png"); plt.show()

none_flagged = expected_cost(y_hold, p_lr, hold_df[AMOUNT_COL], np.inf)
print(f"do nothing              ${none_flagged['total_cost']:>12,.0f}")
print(f"model at best threshold ${best['total_cost']:>12,.0f}   "
      f"({best['n_alerts']:,.0f} alerts, {best['n_alerts']/N_DAYS:,.0f}/day, "
      f"precision {best['precision']:.3f})")
print(f"saving {100*(none_flagged['total_cost']-best['total_cost'])/none_flagged['total_cost']:.1f}%")
print(f"\nrecall {best['recall']:.3f} (cases) vs value_recall {best['value_recall']:.3f} (dollars)")
print("Those differing is the point: catching 60% of fraud CASES is not")
print("catching 60% of fraud DOLLARS, and risk teams are funded on the second.")

flag_ev = expected_value_decision(p_lr, hold_df[AMOUNT_COL])
print(f"\namount-weighted rule flags {flag_ev.mean():.1%} of traffic -- because the")
print("probabilities above are inflated ~10x, p*(amount+fee) clears $10 for")
print("almost everything. The rule is right; the inputs are not. Section 6.")
# %% [markdown]
# ## 5. LightGBM
#
# **Translating from deep learning:** `n_estimators` ~ epochs (set high, let
# early stopping choose); `learning_rate` trades against tree count with no
# schedule; `num_leaves` is capacity; `min_child_samples` is the main
# regulariser; `colsample_bytree` is loosely dropout.
#
# The one with no analogue: **there is no imputation step.** LightGBM learns a
# default direction for missing values at each split, which usually beats any
# median you would substitute.

# %%
X_all, cats_all = prepare_lgbm_frame(train_df, LEAK_COLS)
X_cap, cats_cap = prepare_lgbm_frame(train_df, LEAK_COLS, max_cardinality=25)
print(f"all categoricals : {X_all.shape[1]} cols, {len(cats_all)} categorical")
print(f"capped at 25     : {X_cap.shape[1]} cols, {len(cats_cap)} categorical")


def cv_score(X, cats, label, frame=None, y_ref=None, **over):
    frame = train_df if frame is None else frame
    y_ref = y_train if y_ref is None else y_ref
    rows, t0 = [], time.time()
    for k, (tr, va) in enumerate(PurgedTimeSeriesSplit().split(frame)):
        X_tr = X.iloc[tr]
        X_va = align_categories(X_tr, X.iloc[va], cats)
        m, n_trees = fit_lgbm_es(X_tr, y_ref[tr], cats, **over)
        p = m.predict_proba(X_va)[:, 1]
        rec, _ = recall_at_fpr(y_ref[va], p)
        rows.append({"label": label, "fold": k, "pr_auc": pr_auc(y_ref[va], p),
                     "recall@1%fpr": rec, "n_trees": n_trees})
    out = pd.DataFrame(rows)
    print(f"  {label:26s} PR-AUC {out.pr_auc.mean():.4f} +/- {out.pr_auc.std():.4f}"
          f"  trees~{out.n_trees.mean():.0f}  ({time.time()-t0:.0f}s)")
    return out


# %% [markdown]
# ### 5.1 Do high-cardinality categoricals help?
#
# `card1` has thousands of levels. A tree can split on it directly — either
# useful BIN-level generalisation or pure memorisation. The purged split is
# the instrument that tells them apart, so measure rather than assume.

# %%
cmp_cats = pd.concat([cv_score(X_all, cats_all, "all categoricals"),
                      cv_score(X_cap, cats_cap, "capped at 25 levels")],
                     ignore_index=True)
means = cmp_cats.groupby("label")["pr_auc"].mean()
sd = cmp_cats.groupby("label")["pr_auc"].std().max()
winner = means.idxmax()
print(f"\ndifference {abs(means.diff().iloc[-1]):.4f} against fold sd {sd:.4f}")
if abs(means.diff().iloc[-1]) < sd:
    print("-> INSIDE NOISE. The honest read is 'no measurable difference',")
    print("   so take the simpler feature set, not 'capping won'.")
X, cat_cols = (X_all, cats_all) if winner == "all categoricals" else (X_cap, cats_cap)
X_hold_full, _ = prepare_lgbm_frame(hold_df, LEAK_COLS)
X_hold = align_categories(X, X_hold_full[X.columns], cat_cols)
print(f"using: {X.shape[1]} columns")

# %% [markdown]
# ### 5.2 Random search
#
# Random, not grid: with six parameters a grid spends most of its budget
# re-testing values of parameters that do not matter, while random search
# covers each individual axis far better for the same number of fits
# (Bergstra & Bengio 2012).
#
# `n_estimators` is deliberately **not** in the space — early stopping already
# chooses it per fold.

# %%
rng = np.random.default_rng(SEED)
SPACE = {
    "learning_rate":     lambda: float(10 ** rng.uniform(-2.0, -0.9)),
    "num_leaves":        lambda: int(rng.choice([15, 31, 63, 127, 255])),
    "min_child_samples": lambda: int(rng.choice([20, 50, 100, 200, 400])),
    "colsample_bytree":  lambda: float(rng.uniform(0.4, 0.95)),
    "subsample":         lambda: float(rng.uniform(0.6, 1.0)),
    "reg_lambda":        lambda: float(10 ** rng.uniform(-1, 2)),
}
base_cv = cv_score(X, cat_cols, "lgbm defaults")
trials = []
for t in range(N_SEARCH_TRIALS):
    params = {k: f() for k, f in SPACE.items()}
    r = cv_score(X, cat_cols, f"trial {t:02d}", **params)
    trials.append({**params, "pr_auc": r.pr_auc.mean(), "sd": r.pr_auc.std()})

search = pd.DataFrame(trials).sort_values("pr_auc", ascending=False)
print(search.head(8).to_string(index=False))
best_params = search.iloc[0].drop(["pr_auc", "sd"]).to_dict()
best_params["num_leaves"] = int(best_params["num_leaves"])
best_params["min_child_samples"] = int(best_params["min_child_samples"])

gain = search.iloc[0]["pr_auc"] - base_cv.pr_auc.mean()
print(f"\ntuning gain {gain:+.4f} against fold sd {base_cv.pr_auc.std():.4f}")
print("Tuning a GBDT on tabular data typically buys a few percent -- real, but")
print("an order of magnitude less than feature engineering (section 7).")
if abs(gain) < base_cv.pr_auc.std():
    print("Here it is inside noise: 'defaults were fine' is the honest report.")
# %%
model, n_trees = fit_lgbm_es(X, y_train, cat_cols, **best_params)
p_lgb = model.predict_proba(X_hold)[:, 1]
m_lgb = evaluate_all(y_hold, p_lgb, hold_df[AMOUNT_COL], N_DAYS)
print(f"LightGBM holdout ({n_trees} trees)")
for k in ["pr_auc", "roc_auc", "recall_at_1pct_fpr", "brier", "alerts_per_day"]:
    print(f"  {k:22s} {m_lgb[k]:,.4f}")
log_result("lgbm_tuned", "temporal_holdout", m_lgb, f"trees={n_trees}")

print(f"\nlogreg holdout {m_lr['pr_auc']:.4f}  ->  lightgbm {m_lgb['pr_auc']:.4f}"
      f"  ({m_lgb['pr_auc']/m_lr['pr_auc']:.2f}x)")
# %% [markdown]
# ### 5.3 Importance — and why the built-in one misleads
#
# LightGBM's importances are both biased: **split count** favours
# high-cardinality features because they offer more places to cut, and **gain**
# is measured on training data, so a feature that overfits looks important
# precisely because it overfits.
#
# Permutation importance on the *holdout* answers the question you actually
# care about — does this column help on data the model has not seen.

# %%
gain_imp = pd.Series(model.booster_.feature_importance("gain"),
                     index=X.columns).sort_values(ascending=False)


def permutation_importance(model, X_ref, y_ref, features, n_repeats=2, seed=SEED):
    rng = np.random.default_rng(seed)
    base = pr_auc(y_ref, model.predict_proba(X_ref)[:, 1])
    out = {}
    for f in features:
        dtype = X_ref[f].dtype       # .to_numpy() on a category column returns
        drops = []                   # object; assigning it back drops the dtype
        for _ in range(n_repeats):   # and LightGBM then rejects the schema.
            Xp = X_ref.copy()
            Xp[f] = pd.Series(rng.permutation(Xp[f].to_numpy()), index=Xp.index).astype(dtype)
            drops.append(base - pr_auc(y_ref, model.predict_proba(Xp)[:, 1]))
        out[f] = float(np.mean(drops))
    return pd.Series(out).sort_values(ascending=False)


perm = permutation_importance(model, X_hold, y_hold, gain_imp.head(15).index.tolist())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
gain_imp.head(15)[::-1].plot.barh(ax=axes[0], color=BLUE)
axes[0].set_title("Gain importance (train, biased)")
perm[::-1].plot.barh(ax=axes[1], color=GREEN)
axes[1].axvline(0, c="0.4", lw=1)
axes[1].set_title("Permutation importance (holdout PR-AUC drop)")
plt.tight_layout(); plt.savefig(OUT / "05_importance.png"); plt.show()
print("Features high by gain but ~0 by permutation are fitting training noise.")
print("Any with a NEGATIVE permutation score are actively hurting.")

# %% [markdown]
# ## 6. Class imbalance — a benchmark, not a ritual
#
# "Use SMOTE for imbalanced data" is repeated so often it rarely gets tested.
#
# | arm | changes | cost |
# |---|---|---|
# | none | nothing | — |
# | `scale_pos_weight` | the **loss** | probabilities lose calibration |
# | undersampling | the **data**, deleting negatives | discards real information |
# | SMOTE | the **data**, inventing positives | invents rows that may be impossible |
#
# **The rule that makes or breaks this: resample the training fold only.**
# Never the validation fold, never before the split. Section 6.3 measures what
# breaking it costs.
#
# *Prediction, written before running:* SMOTE will not beat `scale_pos_weight`.
# Interpolating between transactions in a mixed categorical/numeric space
# produces rows that could not occur, and gradient boosting with a weighted
# loss already handles imbalance without touching the data.

# %%
def scale_pos_weight(yv):
    yv = np.asarray(yv)
    return float((yv == 0).sum() / max(int((yv == 1).sum()), 1))


def undersample(Xd, yv, ratio=0.10, seed=SEED):
    """Returns (X, y, beta) where beta is the fraction of negatives kept."""
    rng = np.random.default_rng(seed)
    yv = np.asarray(yv)
    pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
    n_keep = min(int(len(pos) * (1 - ratio) / ratio), len(neg))
    keep = np.sort(np.concatenate([pos, rng.choice(neg, n_keep, replace=False)]))
    return Xd.iloc[keep], yv[keep], n_keep / max(len(neg), 1)


def oversample_naive(Xd, yv, ratio=0.10, seed=SEED):
    """Duplicate minority rows. The control that isolates what SMOTE's
    INTERPOLATION buys over plain reweighting -- duplication adds no
    information at all."""
    rng = np.random.default_rng(seed)
    yv = np.asarray(yv)
    pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
    n_extra = max(int(len(neg) * ratio / (1 - ratio)) - len(pos), 0)
    idx = np.concatenate([np.arange(len(yv)), rng.choice(pos, n_extra, replace=True)])
    return Xd.iloc[idx], yv[idx]


def smote(Xd, yv, ratio=0.10, k=5, categorical=None, seed=SEED):
    """SMOTE (Chawla et al. 2002), with the two corrections tutorials skip.

    1. The neighbour search is Euclidean, so an unscaled TransactionAmt (range
       0-32,000) dominates every other column and 'neighbours' just means
       transactions of similar size. Standardise first.
    2. Interpolation is nonsense for a categorical. Halfway between
       card_type=1 (debit) and 2 (credit) is 1.5, which is not a card type.
       Categoricals take the MODAL neighbour value (the SMOTE-NC correction).

    What SMOTE cannot know either way: whether the synthesised row is a
    plausible transaction. It can carry one account's 30-day average alongside
    another's merchant category.
    """
    rng = np.random.default_rng(seed)
    yv = np.asarray(yv)
    categorical = categorical or []
    num_cols = [c for c in Xd.columns if c not in categorical]
    pos_mask = yv == 1
    X_pos = Xd.loc[pos_mask]

    n_neg = int((~pos_mask).sum())
    n_new = max(int(n_neg * ratio / (1 - ratio)) - len(X_pos), 0)
    if n_new == 0 or len(X_pos) <= k:
        return Xd, yv

    A = X_pos[num_cols].to_numpy(dtype="float64")
    A = np.where(np.isnan(A), np.nanmedian(A, axis=0), A)
    sd = A.std(0); sd[sd == 0] = 1.0
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(A))).fit((A - A.mean(0)) / sd)
    nbr = nn.kneighbors((A - A.mean(0)) / sd)[1][:, 1:]

    i = rng.integers(0, len(A), n_new)
    j = nbr[i, rng.integers(0, nbr.shape[1], n_new)]
    new = pd.DataFrame(A[i] + rng.random((n_new, 1)) * (A[j] - A[i]), columns=num_cols)

    for c in categorical:
        vals = X_pos[c].to_numpy()
        cand = np.stack([vals[nbr[i, m]] for m in range(nbr.shape[1])], axis=1)
        new[c] = [pd.Series(r).mode().iloc[0] if len(pd.Series(r).mode()) else vals[i[q]]
                  for q, r in enumerate(cand)]

    new = new[Xd.columns]
    for c in Xd.columns:
        try:
            new[c] = new[c].astype(Xd[c].dtype)
        except (TypeError, ValueError):
            pass
    return pd.concat([Xd, new], ignore_index=True), np.concatenate(
        [yv, np.ones(n_new, dtype=yv.dtype)])


def correct_prior(p, beta):
    """Map probabilities from a resampled world back to the true base rate:

        p_true = p / (p + (1-p)/beta)

    Elkan (2001). The transform is MONOTONE, so PR-AUC will not move at all
    and Brier will move a lot. That contrast is the entire point.
    """
    p = np.asarray(p, dtype="float64")
    return p / (p + (1 - p) / max(float(beta), 1e-12))


# %%
TARGET_RATIO = 0.10

def run_arm(name, sampler=None, use_weight=False, correct=False):
    rows, t0 = [], time.time()
    for k, (tr, va) in enumerate(PurgedTimeSeriesSplit(warn=False).split(train_df)):
        X_tr, y_tr = X.iloc[tr], y_train[tr]
        X_va, y_va = align_categories(X_tr, X.iloc[va], cat_cols), y_train[va]

        beta = 1.0
        if sampler is not None:
            res = sampler(X_tr, y_tr)
            X_tr, y_tr = res[0].reset_index(drop=True), res[1]
            beta = res[2] if len(res) > 2 else 1.0

        over = dict(best_params)
        if use_weight:
            over["scale_pos_weight"] = scale_pos_weight(y_tr)
        m, _ = fit_lgbm_es(X_tr, y_tr, cat_cols, **over)
        p = m.predict_proba(X_va)[:, 1]
        if correct and beta < 1.0:
            p = correct_prior(p, beta)
        rows.append({"arm": name, "pr_auc": pr_auc(y_va, p),
                     "brier": brier(y_va, p), "n_train": len(y_tr)})
    out = pd.DataFrame(rows)
    print(f"  {name:26s} PR-AUC {out.pr_auc.mean():.4f} +/- {out.pr_auc.std():.4f}"
          f"  Brier {out.brier.mean():.4f}  ({time.time()-t0:.0f}s)")
    return out

arms = pd.concat([
    run_arm("none"),
    run_arm("scale_pos_weight", use_weight=True),
    run_arm("undersample 1:10", sampler=lambda a, b: undersample(a, b, TARGET_RATIO)),
    run_arm("undersample + prior fix",
            sampler=lambda a, b: undersample(a, b, TARGET_RATIO), correct=True),
    run_arm("oversample (duplicate)",
            sampler=lambda a, b: oversample_naive(a, b, TARGET_RATIO)),
    run_arm("SMOTE", sampler=lambda a, b: smote(a, b, TARGET_RATIO,
                                                categorical=cat_cols)),
], ignore_index=True)

summ = (arms.groupby("arm").agg(pr_auc=("pr_auc", "mean"), sd=("pr_auc", "std"),
                                brier=("brier", "mean"), rows=("n_train", "mean"))
        .sort_values("pr_auc", ascending=False).round(4))
print("\n" + summ.to_string())
base, bsd = summ.loc["none", "pr_auc"], summ.loc["none", "sd"]
print(f"\nBar: one fold sd = {bsd:.4f}. Anything within +/-{bsd:.4f} of {base:.4f}")
print("is indistinguishable from doing nothing.")
for a in summ.index:
    log_result(f"cv_{a}", "purged_cv", summ.loc[a].to_dict(), "section 6")
# %% [markdown]
# ### 6.2 What resampling does to the probabilities

# %%
Xu, yu, beta = undersample(X, y_train, 0.10)
mu, _ = fit_lgbm_es(Xu.reset_index(drop=True), yu, cat_cols, **best_params)
p_raw = mu.predict_proba(X_hold)[:, 1]
p_fix = correct_prior(p_raw, beta)

print(f"kept {beta:.1%} of negatives; fold fraud rate "
      f"{y_train.mean():.3%} -> {yu.mean():.3%}\n")
print(f"                   mean p    Brier     PR-AUC")
print(f"  true rate        {y_hold.mean():.4f}       -          -")
print(f"  uncorrected      {p_raw.mean():.4f}   {brier(y_hold, p_raw):.4f}   {pr_auc(y_hold, p_raw):.4f}")
print(f"  prior-corrected  {p_fix.mean():.4f}   {brier(y_hold, p_fix):.4f}   {pr_auc(y_hold, p_fix):.4f}")
print("\nPR-AUC identical to the last digit -- the correction is monotone.")
print("Brier moves a lot. If you only report ranking metrics, this entire")
print("failure is invisible to you.")

fig, ax = plt.subplots(figsize=(5.5, 5))
for lab, p in [("uncorrected", p_raw), ("prior-corrected", p_fix)]:
    r = reliability_table(y_hold, p)
    ax.plot(r["mean_predicted"], r["observed"], "o-", lw=2, label=lab)
lim = float(max(p_raw.mean() * 3, 0.1))
ax.plot([0, lim], [0, lim], "--", c="0.5", lw=1, label="perfect")
ax.set_xlabel("mean predicted"); ax.set_ylabel("observed rate")
ax.set_title("Prior correction after undersampling"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "06_prior.png"); plt.show()
# %% [markdown]
# ### 6.3 The mistake, measured
#
# The most common error in published fraud notebooks: **resample the whole
# dataset, then split.**
#
# With SMOTE this is catastrophic in a specific way. Synthetic positives are
# interpolated from real positives; split afterwards and a synthetic row built
# from a validation-set neighbour lands in training. The model has been shown a
# blurred copy of the answer key.

# %%
X_bad, y_bad = smote(X, y_train, 0.10, categorical=cat_cols)
X_bad = X_bad.reset_index(drop=True)
rs = np.random.default_rng(SEED).permutation(len(y_bad))
cutp = int(0.8 * len(rs)); tr_b, va_b = rs[:cutp], rs[cutp:]

mb, _ = fit_lgbm_es(X_bad.iloc[tr_b].reset_index(drop=True), y_bad[tr_b],
                    cat_cols, **best_params)
pb = mb.predict_proba(align_categories(X_bad.iloc[tr_b], X_bad.iloc[va_b],
                                       cat_cols))[:, 1]
honest_smote = summ.loc["SMOTE", "pr_auc"]
fiction = pr_auc(y_bad[va_b], pb)
print(f"  SMOTE inside the fold, purged split : {honest_smote:.4f}   <- real")
print(f"  SMOTE before a random split         : {fiction:.4f}   <- fiction")
print(f"  overstatement: {100*(fiction-honest_smote)/honest_smote:.0f}%")
print("\nA fraud PR-AUC above ~0.9 on this dataset is almost always this bug.")
# %% [markdown]
# ## 7. Feature engineering
#
# This is where the leverage is. Sections 5 and 6 moved PR-AUC by fractions of
# a fold standard deviation; this should move it by multiples.
#
# That asymmetry is the central fact about tabular ML. In vision or NLP the
# network learns its own representation. Here the model is a fixed function and
# **you** are the representation learner — LightGBM cannot invent "this
# transaction is 8x this cardholder's 30-day average", it can only split on
# columns you hand it.
#
# **The discipline:** every feature is computed from strictly past rows. A
# plain `groupby('uid')['amt'].transform('mean')` reads the account's whole
# history including transactions that have not happened yet at scoring time. It
# raises your score and it is not available in production.
#
# Section 7.2 does not assert this. It tests it.

# %%
def past_rolling(d, group_col, value_col, window, stat="count"):
    """Trailing-window aggregate within group, EXCLUDING the current row.

    closed='left' is the entire safety argument. The default, closed='right',
    includes the current row -- so a 'mean amount over the last 7 days' feature
    would contain this transaction's own amount, and on a group's first
    transaction the feature would EQUAL the value it is meant to predict.
    """
    t = pd.to_datetime(d[TIME_COL].to_numpy(), unit="s")
    tmp = pd.DataFrame({"t": t, "g": d[group_col].to_numpy(),
                        "v": d[value_col].to_numpy()}, index=d.index)
    order = tmp.sort_values(["g", "t"], kind="stable").index
    rolled = (tmp.loc[order].set_index("t").groupby("g", observed=True)["v"]
              .rolling(window, closed="left").agg(stat))
    return pd.Series(rolled.to_numpy(), index=order).reindex(d.index)


def past_expanding(d, group_col, value_col, stat="mean"):
    """All-prior-rows aggregate, EXCLUDING the current row.

    Cumulative sums rather than .expanding(): one pass instead of one per
    group. Subtracting the current row's own contribution is what makes it
    past-only.
    """
    v = d[value_col].astype("float64")
    g = d.groupby(group_col, observed=True)
    n = g.cumcount().to_numpy().astype("float64")
    csum = g[value_col].cumsum().to_numpy() - v.to_numpy()
    if stat == "count":
        return pd.Series(n, index=d.index)
    mean = np.where(n > 0, csum / np.maximum(n, 1), np.nan)
    if stat == "mean":
        return pd.Series(mean, index=d.index)
    if stat == "std":
        csq = (d.assign(_s=v**2).groupby(group_col, observed=True)["_s"].cumsum().to_numpy()
               - v.to_numpy()**2)
        var = np.where(n > 1, csq / np.maximum(n, 1) - mean**2, np.nan)
        return pd.Series(np.sqrt(np.clip(var, 0, None)), index=d.index)
    if stat == "sum":
        return pd.Series(np.where(n > 0, csum, np.nan), index=d.index)
    raise ValueError(stat)


def expanding_target_encode(d, col, yv, smoothing=20.0, prior=None):
    """Target encoding from EARLIER ROWS ONLY.

    Target encoding is the highest-value and highest-risk trick in tabular ML.
    Done naively -- category mean over the whole training set -- every row's
    encoding contains its own label and a high-cardinality column becomes a
    near-perfect predictor that generalises to nothing.

    On time-ordered data the expanding form beats out-of-fold encoding: it is
    leak-free by the same argument as every other feature here, it mirrors
    production, and it leaves no uncovered rows.
    """
    prior = float(yv.mean()) if prior is None else prior
    tmp = pd.DataFrame({"c": d[col].astype("object").to_numpy(),
                        "y": yv.astype("float64")})
    g = tmp.groupby("c", observed=True, dropna=False)["y"]
    csum = g.cumsum().to_numpy() - tmp["y"].to_numpy()
    cnt = g.cumcount().to_numpy().astype("float64")
    return pd.Series((csum + smoothing * prior) / (cnt + smoothing),
                     index=d.index).astype("float32")


def past_count_encode(d, col):
    """How often this value was seen BEFORE this row.

    NOT value_counts(). That maps each value to its whole-frame frequency, so
    the count at row i depends on rows after i. It is unsupervised -- the label
    is never touched -- which makes it FEEL harmless, and the leak test in 7.2
    flags it anyway. Unsupervised does not mean leak-free.
    """
    return d.groupby(col, observed=True, dropna=False).cumcount().astype("float32")


def build_features(d, groups=("A", "B", "C", "D", "E", "F"), verbose=True):
    out = pd.DataFrame(index=d.index)
    amt = d[AMOUNT_COL].astype("float64")

    if "A" in groups:
        out["amt_log"] = np.log1p(amt).astype("float32")
        # Cents: foreign-currency conversions leave unusual decimals while
        # ordinary retail prices cluster on .00 and .99.
        out["amt_decimal"] = ((amt - np.floor(amt)) * 100).round().astype("float32")
        out["amt_is_round"] = (out["amt_decimal"] == 0).astype("int8")
        # D columns are 'days since <event>', so they climb with calendar time
        # and a tree splits on WHEN rather than on behaviour. Subtracting the
        # day pins them to a fixed origin. This is also the most likely fix for
        # the logistic-regression CV->holdout collapse noted in section 4.
        for c in [f"D{i}" for i in range(1, 16) if f"D{i}" in d.columns]:
            out[f"{c}_norm"] = (d["day"] - d[c].astype("float64")).astype("float32")

    if "B" in groups:
        for win, lab in [("1h", "1h"), ("24h", "24h"), ("7D", "7d"),
                         (f"{MAX_WINDOW_DAYS}D", f"{MAX_WINDOW_DAYS}d")]:
            out[f"uid_n_{lab}"] = past_rolling(d, "uid", AMOUNT_COL, win, "count").astype("float32")
        out["card1_n_24h"] = past_rolling(d, "card1", AMOUNT_COL, "24h", "count").astype("float32")
        out["uid_amt_sum_24h"] = past_rolling(d, "uid", AMOUNT_COL, "24h", "sum").astype("float32")

    if "C" in groups:
        pm = past_expanding(d, "uid", AMOUNT_COL, "mean")
        ps = past_expanding(d, "uid", AMOUNT_COL, "std")
        out["uid_amt_mean_prior"] = pm.astype("float32")
        out["uid_amt_std_prior"] = ps.astype("float32")
        # The most intuitive fraud feature there is: how unusual is this amount
        # FOR THIS CARDHOLDER. $2,000 is unremarkable on some cards and a
        # screaming anomaly on others; the raw amount cannot express that.
        out["amt_to_uid_mean"] = (amt / pm).astype("float32")
        out["amt_z_uid"] = ((amt - pm) / ps.replace(0, np.nan)).astype("float32")
        out["uid_n_prior"] = past_expanding(d, "uid", AMOUNT_COL, "count").astype("float32")

    if "D" in groups:
        dt = d.groupby("uid", observed=True)[TIME_COL].diff()
        out["uid_secs_since_prev"] = dt.astype("float32")
        out["uid_log_secs_since_prev"] = np.log1p(dt.clip(lower=0)).astype("float32")
        out["card1_secs_since_prev"] = d.groupby("card1", observed=True)[TIME_COL].diff().astype("float32")

    if "E" in groups:
        for c in ["ProductCD", "P_emaildomain", "card6", "DeviceType"]:
            if c in d.columns:
                out[f"uid_new_{c}"] = (d.groupby(["uid", c], observed=True).cumcount() == 0).astype("int8")
        out["uid_is_first_txn"] = (d.groupby("uid", observed=True).cumcount() == 0).astype("int8")

    if "F" in groups:
        for c in ["card1", "addr1", "P_emaildomain", "R_emaildomain", "DeviceInfo", "uid"]:
            if c in d.columns:
                out[f"{c}_seen_before"] = past_count_encode(d, c)

    if verbose:
        print(f"  built {out.shape[1]} features ({', '.join(sorted(set(groups)))})")
    return out


def combine(base, extra):
    """Concat engineered columns onto a raw matrix, dropping overlaps.
    LightGBM refuses a repeated feature name, and the error surfaces deep
    inside Dataset construction far from the concat that caused it."""
    dupes = [c for c in extra.columns if c in base.columns]
    return pd.concat([base.reset_index(drop=True),
                      extra.drop(columns=dupes).reset_index(drop=True)], axis=1)


# %%
assert PURGE_DAYS >= MAX_WINDOW_DAYS, "purge gap must cover the longest window"
t0 = time.time()
feat = build_features(df)
feat["uid_te"] = expanding_target_encode(df, "uid", y)
feat["card1_te"] = expanding_target_encode(df, "card1", y)
if "P_emaildomain" in df.columns:
    feat["email_te"] = expanding_target_encode(df, "P_emaildomain", y)
print(f"  {feat.shape[1]} engineered features in {time.time()-t0:.0f}s")
# %% [markdown]
# ### 7.1 A note on building before splitting
#
# Building past-only features over the whole frame **before** splitting is
# correct, and it is not the same mistake as fitting a scaler before splitting.
#
# A scaler learns a statistic *from* the data — a mean over all rows, future
# included. A past-only feature at row *i* reads only rows < *i* by
# construction; whether rows after *i* happen to sit in the same DataFrame is
# irrelevant to its value. It is also what production does: at scoring time the
# account's real history exists.
#
# ### 7.2 Prove it
#
# If a feature at row *i* is genuinely past-only, recomputing it over
# `df[:i+1]` must give **exactly** the full-frame value. Rows after *i* could
# not have contributed. Any difference means it read forward.
#
# This catches the entire class of bug at once — a forgotten `closed='left'`, a
# `transform('mean')` that crept in, a target encoding including its own row —
# without reasoning about each feature individually.

# %%
def verify_past_only(build_fn, d, n_checks=8, seed=SEED, atol=1e-4, verbose=True):
    rng = np.random.default_rng(seed)
    full = build_fn(d)
    lo = int(len(d) * 0.55)
    positions = np.sort(rng.choice(np.arange(lo, len(d)), n_checks, replace=False))

    rows = []
    for i in positions:
        prefix = build_fn(d.iloc[: i + 1])
        a, b = full.iloc[i], prefix.iloc[-1]
        for col in full.columns:
            va, vb = a[col], b[col]
            if pd.isna(va) and pd.isna(vb):
                continue
            if pd.isna(va) != pd.isna(vb) or not np.isclose(float(va), float(vb),
                                                            atol=atol, rtol=1e-3):
                rows.append({"row": int(i), "feature": col,
                             "full_data": va, "past_only": vb})
    bad = pd.DataFrame(rows)
    if verbose:
        if bad.empty:
            print(f"  PASS: {full.shape[1]} features identical at {n_checks} "
                  f"checkpoints -- none read the future")
        else:
            print(f"  FAIL: {bad['feature'].nunique()} leaking features")
            print(bad.groupby("feature").size().to_string())
    return bad


sub = df.iloc[:60_000]
bad = verify_past_only(lambda d: build_features(d, verbose=False), sub)
assert bad.empty, "leaking features -- fix before measuring anything"

# A planted leak, to show the test can actually fail. A passing test proves
# nothing until you have watched it catch something.
def leaky(d):
    out = build_features(d, groups=("A",), verbose=False)
    out["LEAK_whole_history_mean"] = d.groupby("uid")[AMOUNT_COL].transform("mean")
    return out

print("\ncontrol -- this SHOULD fail:")
caught = verify_past_only(leaky, sub, n_checks=4)
assert not caught.empty, "the detector failed to catch a known leak"
# %% [markdown]
# ### 7.3 Ablation — what is each block worth?

# %%
BLOCKS = {"A amount / D-norm": ("A",), "B velocity": ("A", "B"),
          "C amount-vs-history": ("A", "B", "C"), "D recency": ("A", "B", "C", "D"),
          "E novelty": ("A", "B", "C", "D", "E"),
          "F prior counts": ("A", "B", "C", "D", "E", "F")}

feat_train = feat.iloc[train_idx].reset_index(drop=True)
feat_hold = feat.iloc[hold_idx].reset_index(drop=True)
probe = build_features(df.iloc[:300], verbose=False)

abl = [cv_score(X, cat_cols, "0 raw only", **best_params)]
for label, groups in BLOCKS.items():
    cols = [c for c in build_features(df.iloc[:300], groups=groups, verbose=False).columns
            if c in feat.columns]
    abl.append(cv_score(combine(X, feat_train[cols]), cat_cols, label, **best_params))
abl.append(cv_score(combine(X, feat_train), cat_cols, "G + target encoding",
                    **best_params))
ablation = pd.concat(abl, ignore_index=True)

summ_a = (ablation.groupby("label", sort=False)
          .agg(pr_auc=("pr_auc", "mean"), sd=("pr_auc", "std")).round(4))
summ_a["delta"] = summ_a["pr_auc"].diff().round(4)
summ_a["cum_lift"] = (summ_a["pr_auc"] / summ_a["pr_auc"].iloc[0]).round(2)
print("\n" + summ_a.to_string())
print(f"\nraw {summ_a.pr_auc.iloc[0]:.4f} -> best {summ_a.pr_auc.max():.4f}"
      f"  ({summ_a.pr_auc.max()/summ_a.pr_auc.iloc[0]:.2f}x)")
print("Compare against sections 5-6, where tuning and imbalance handling")
print("together moved PR-AUC by less than one fold standard deviation.")
print("\nBlocks with a NEGATIVE delta are worth as much in a writeup as the")
print("positive ones: extra columns cost fit time, dilute column sampling,")
print("and add surface area to overfit on a drifting problem.")
# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
lab = summ_a.index.tolist()
axes[0].barh(range(len(lab)), summ_a["pr_auc"], xerr=summ_a["sd"].fillna(0),
             capsize=3, color=BLUE)
axes[0].set_yticks(range(len(lab))); axes[0].set_yticklabels(lab, fontsize=8)
axes[0].invert_yaxis(); axes[0].axvline(summ_a.pr_auc.iloc[0], ls="--", c="0.4", lw=1)
axes[0].set_title("Cumulative PR-AUC as blocks are added")

d = summ_a["delta"].fillna(0)
axes[1].barh(range(len(d)), d.to_numpy(),
             color=[GREEN if v > 0 else RED for v in d])
axes[1].set_yticks(range(len(d))); axes[1].set_yticklabels(d.index, fontsize=8)
axes[1].invert_yaxis(); axes[1].axvline(0, c="0.3", lw=1)
axes[1].set_title("Marginal contribution of each block")
plt.tight_layout(); plt.savefig(OUT / "07_ablation.png"); plt.show()
# %% [markdown]
# ### 7.4 Final model

# %%
X_tr_f = combine(X, feat_train)
X_ho_f = combine(X_hold, feat_hold)
X_ho_f = align_categories(X_tr_f, X_ho_f, cat_cols)

# Target/frequency encodings for the holdout are learned from the WHOLE
# training block -- safe precisely because the holdout lies entirely after
# it, so at scoring time all of training genuinely is the past.
prior = float(y_train.mean())
for col, name in [("uid", "uid_te"), ("card1", "card1_te"), ("P_emaildomain", "email_te")]:
    if name in X_ho_f.columns and col in df.columns:
        st = (pd.DataFrame({"c": train_df[col].to_numpy(), "y": y_train})
              .groupby("c", observed=True, dropna=False)["y"].agg(["sum", "count"]))
        enc = (st["sum"] + 20.0 * prior) / (st["count"] + 20.0)
        X_ho_f[name] = hold_df[col].map(enc).fillna(prior).astype("float32").to_numpy()

# Population frequency: the fitted transform that could NOT live in
# build_features because it reads the future. Fit on train, apply to both.
for col in ["card1", "addr1", "P_emaildomain", "uid"]:
    if col in df.columns:
        counts = train_df[col].value_counts()
        X_tr_f[f"{col}_freq"] = train_df[col].map(counts).fillna(0).astype("float32").to_numpy()
        X_ho_f[f"{col}_freq"] = hold_df[col].map(counts).fillna(0).astype("float32").to_numpy()

print(f"final matrix: {X_tr_f.shape[1]} columns")
model_f, n_trees_f = fit_lgbm_es(X_tr_f, y_train, cat_cols, **best_params)
p_feat = model_f.predict_proba(X_ho_f)[:, 1]
m_feat = evaluate_all(y_hold, p_feat, hold_df[AMOUNT_COL], N_DAYS)

for k in ["pr_auc", "roc_auc", "recall_at_1pct_fpr", "brier", "lift_over_base",
          "alerts_per_day", "min_total_cost", "recall_at_best"]:
    print(f"  {k:22s} {m_feat[k]:,.4f}")
print(f"\n  raw features {m_lgb['pr_auc']:.4f} -> engineered {m_feat['pr_auc']:.4f}"
      f"  ({m_feat['pr_auc']/m_lgb['pr_auc']:.2f}x)")
log_result("lgbm_engineered", "temporal_holdout", m_feat,
           f"{X_tr_f.shape[1]} features, purge={PURGE_DAYS}d")

g = pd.Series(model_f.booster_.feature_importance("gain"),
              index=X_tr_f.columns).sort_values(ascending=False)
eng = set(feat.columns)
top = g.head(25)
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
top[::-1].plot.barh(ax=axes[0],
                    color=[GREEN if c in eng else BLUE for c in top.index][::-1])
axes[0].set_title("Top 25 by gain (green = engineered)")
plot_pr({"engineered": (y_hold, p_feat), "raw features": (y_hold, p_lgb)},
        float(y_hold.mean()), ax=axes[1])
plt.tight_layout(); plt.savefig(OUT / "07_final.png"); plt.show()
print(f"engineered features carry {g[g.index.isin(eng)].sum()/g.sum():.1%} of total gain")
# %% [markdown]
# ## 8. Calibration — closing the loop
#
# Sections 4 and 6 both diagnosed the same problem and neither fixed it: the
# probabilities are inflated roughly tenfold, so the amount-weighted decision
# rule flags most of the traffic and costs more than doing nothing.
#
# **Every ranking metric is blind to this.** PR-AUC, ROC-AUC and recall@FPR are
# invariant to any monotone transform of the scores. The moment those scores
# get multiplied by dollar amounts, that invariance stops being harmless.
#
# ### Fitting it without cheating
#
# The calibrator has to be fit on data the model did not train on, and it must
# not be the holdout we report against. So: carve the last 15% of the training
# block **by time**, fit the model on the first 85%, and fit the calibrator on
# the model's predictions over that slice. That mirrors deployment, where you
# calibrate on recent history and apply going forward.
#
# One honest caveat: the calibrator is learned from a model trained on slightly
# less data than the final one. That is the standard trade and it is why the
# calibration slice stays small.

# %%
from sklearn.isotonic import IsotonicRegression

FINAL_XTR, FINAL_XHO, FINAL_P = X_tr_f, X_ho_f, p_feat
CAL_FRAC = 0.15
_cut = int(len(FINAL_XTR) * (1 - CAL_FRAC))
X_fit, y_fit = FINAL_XTR.iloc[:_cut], y_train[:_cut]
X_cal, y_cal = FINAL_XTR.iloc[_cut:], y_train[_cut:]
X_cal = align_categories(X_fit, X_cal, cat_cols)

m_cal, _ = fit_lgbm_es(X_fit, y_fit, cat_cols, **best_params)
p_cal = m_cal.predict_proba(X_cal)[:, 1]
print(f"calibration slice: {len(y_cal):,} rows, fraud {y_cal.mean():.3%}")

# Isotonic: a free-form monotone step function. Flexible, needs data.
iso = IsotonicRegression(out_of_bounds="clip").fit(p_cal, y_cal)
p_iso = iso.predict(FINAL_P)

# Platt: a single logistic squashing. Fewer parameters, safer on small samples.
platt = LogisticRegression(max_iter=1000).fit(p_cal.reshape(-1, 1), y_cal)
p_platt = platt.predict_proba(FINAL_P.reshape(-1, 1))[:, 1]

print(f"""
                 mean p     Brier      PR-AUC
  true rate      {y_hold.mean():.4f}        -           -
  uncalibrated   {FINAL_P.mean():.4f}    {brier(y_hold, FINAL_P):.4f}     {pr_auc(y_hold, FINAL_P):.4f}
  Platt          {p_platt.mean():.4f}    {brier(y_hold, p_platt):.4f}     {pr_auc(y_hold, p_platt):.4f}
  isotonic       {p_iso.mean():.4f}    {brier(y_hold, p_iso):.4f}     {pr_auc(y_hold, p_iso):.4f}

PR-AUC barely moves -- both calibrators are monotone, so the RANKING is
essentially untouched. Brier is what responds. Two metrics, two different
questions: 'are the right transactions at the top' versus 'do the numbers
mean what they say'.
""")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
lim = 0.0
for lab, p, c in [("uncalibrated", FINAL_P, RED), ("Platt", p_platt, ORANGE),
                  ("isotonic", p_iso, GREEN)]:
    r = reliability_table(y_hold, p)
    axes[0].plot(r["mean_predicted"], r["observed"], "o-", lw=2, c=c, label=lab)
    lim = max(lim, float(r["mean_predicted"].max()))
axes[0].plot([0, lim], [0, lim], "--", c="0.5", lw=1, label="perfect")
axes[0].set_xlabel("mean predicted probability"); axes[0].set_ylabel("observed fraud rate")
axes[0].set_title("Reliability: on the diagonal is the goal"); axes[0].legend(fontsize=8)

axes[1].hist(np.log10(np.clip(FINAL_P, 1e-6, 1)), bins=60, alpha=0.6,
             label="uncalibrated", color=RED)
axes[1].hist(np.log10(np.clip(p_iso, 1e-6, 1)), bins=60, alpha=0.6,
             label="isotonic", color=GREEN)
axes[1].axvline(np.log10(y_hold.mean()), ls="--", c="0.3", lw=1.2,
                label=f"true base rate {y_hold.mean():.3%}")
axes[1].set_xlabel("log10(predicted probability)")
axes[1].set_title("Where the probability mass sits"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "08_calibration.png"); plt.show()

# %% [markdown]
# ### The payoff: the amount-weighted rule, now on honest probabilities
#
# This is what calibration was for. The rule
#
# > flag when `p x (amount + fee) > review_cost`
#
# makes the implied probability cutoff a **function of the amount** — a $4,000
# transaction is worth an analyst's ten minutes at p=0.25%, a $12 one needs
# p>27%. A single global threshold cannot express both. But the rule multiplies
# `p` by dollars, so it is only as good as `p`.

# %%
def rule_cost(flag, yt, amount, review=REVIEW_COST, fee=CHARGEBACK_FEE):
    yt = np.asarray(yt); amount = np.asarray(amount, dtype="float64")
    missed = (~flag) & (yt == 1)
    caught = flag & (yt == 1)
    return {"alerts": int(flag.sum()), "alert_rate": float(flag.mean()),
            "recall": float(caught.sum() / max((yt == 1).sum(), 1)),
            "precision": float(caught.sum() / max(flag.sum(), 1)),
            "total_cost": float(flag.sum() * review + (amount + fee)[missed].sum())}


amt_ho = hold_df[AMOUNT_COL]
rows = []
for lab, p in [("uncalibrated", FINAL_P), ("Platt", p_platt), ("isotonic", p_iso)]:
    rows.append({"probs": lab, "rule": "amount-weighted",
                 **rule_cost(expected_value_decision(p, amt_ho), y_hold, amt_ho)})
    c = cost_curve(y_hold, p, amt_ho)
    b = c.loc[c["total_cost"].idxmin()]
    rows.append({"probs": lab, "rule": "single threshold", "alerts": int(b.n_alerts),
                 "alert_rate": float(b.alert_rate), "recall": float(b.recall),
                 "precision": float(b.precision), "total_cost": float(b.total_cost)})

do_nothing = expected_cost(y_hold, FINAL_P, amt_ho, np.inf)["total_cost"]
rules = pd.DataFrame(rows)
rules["vs_do_nothing"] = (100 * (do_nothing - rules["total_cost"]) / do_nothing).round(1)
print(f"do nothing costs ${do_nothing:,.0f}\n")
print(rules.round(4).to_string(index=False))

best_row = rules.loc[rules["total_cost"].idxmin()]
print(f"""
Read the amount-weighted rows top to bottom. Uncalibrated, the rule floods the
queue and can cost MORE than having no model at all. On calibrated
probabilities it becomes usable -- same rule, same code, same model. Only the
inputs changed.

Best configuration: {best_row['rule']} on {best_row['probs']} probabilities,
${best_row['total_cost']:,.0f} ({best_row['vs_do_nothing']:.1f}% saved),
{best_row['alerts']:,} alerts ({best_row['alerts']/N_DAYS:,.0f}/day,
precision {best_row['precision']:.3f}).
""")

log_result("lgbm_calibrated_isotonic", "temporal_holdout",
           evaluate_all(y_hold, p_iso, amt_ho, N_DAYS), "isotonic on 15% time slice")

# %% [markdown]
# ## 9. Explainability — reason codes
#
# In fraud and credit this is not a nice-to-have. "Why did the model decline
# this?" is a question a regulator can compel you to answer, so a score without
# an attached explanation is not deployable.
#
# **TreeSHAP is exact.** For tree ensembles the Shapley values have a
# closed-form polynomial-time solution, so these are not approximations —
# unlike KernelSHAP, which samples. That distinction gets asked about.
#
# LightGBM computes them natively via `pred_contrib=True`, so this needs no
# extra package. Contributions are in **log-odds**, and every row's
# contributions plus a base value sum exactly to that row's raw prediction —
# which the cell below verifies rather than assumes.

# %%
booster = model_f.booster_
contrib = booster.predict(FINAL_XHO, pred_contrib=True)
feat_names = list(FINAL_XHO.columns)

# Additivity: contributions plus the base value must reconstruct the raw
# margin. TreeSHAP is closed-form for tree ensembles -- not sampled like
# KernelSHAP -- so the residual should sit at machine precision.
raw = booster.predict(FINAL_XHO, raw_score=True)
resid = np.abs(contrib.sum(axis=1) - raw).max()
print(f"SHAP additivity check: max |sum(contrib) - raw_score| = {resid:.2e}")
assert resid < 1e-3, "contributions do not reconstruct the prediction"

shap_vals = contrib[:, :-1]
base_value = float(contrib[0, -1])
print(f"base value (log-odds of the prior): {base_value:.4f}"
      f"  -> p = {1/(1+np.exp(-base_value)):.4%}")

# %%
gl = pd.Series(np.abs(shap_vals).mean(0), index=feat_names).sort_values(ascending=False)
eng = set(feat.columns)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top = gl.head(20)
axes[0].barh(range(len(top)), top.to_numpy()[::-1],
             color=[GREEN if c in eng else BLUE for c in top.index][::-1])
axes[0].set_yticks(range(len(top)))
axes[0].set_yticklabels(top.index[::-1], fontsize=8)
axes[0].set_xlabel("mean |SHAP| (log-odds)")
axes[0].set_title("Global importance, TreeSHAP\n(green = engineered)")

# Direction, not just magnitude: does a high value push toward fraud?
tf = top.index[:12]
for i, f in enumerate(tf):
    j = feat_names.index(f)
    v = FINAL_XHO[f]
    v = v.cat.codes.to_numpy() if str(v.dtype) == "category" else v.to_numpy(dtype="float64")
    ok = ~np.isnan(v)
    if ok.sum() > 10:
        hi = v[ok] > np.nanmedian(v[ok])
        axes[1].scatter([shap_vals[ok][hi, j].mean()], [i], c=RED, s=45,
                        label="high value" if i == 0 else None)
        axes[1].scatter([shap_vals[ok][~hi, j].mean()], [i], c=BLUE, s=45,
                        label="low value" if i == 0 else None)
axes[1].set_yticks(range(len(tf))); axes[1].set_yticklabels(tf, fontsize=8)
axes[1].invert_yaxis(); axes[1].axvline(0, c="0.3", lw=1)
axes[1].set_xlabel("mean SHAP contribution (log-odds)")
axes[1].set_title("Direction of effect"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "09_shap_global.png"); plt.show()

# %% [markdown]
# ### Per-transaction reason codes
#
# The global chart is for you. **This** is what goes to the analyst — the same
# SHAP values turned into the two or three sentences that justify an alert.

# %%
def reason_codes(i, k=4):
    """Top contributors for one transaction, as analyst-readable text."""
    s = pd.Series(shap_vals[i], index=feat_names)
    up = s.sort_values(ascending=False).head(k)
    down = s.sort_values().head(2)
    lines = [f"  RISK  {n:<26s} {v:+.3f}   (value = {FINAL_XHO.iloc[i][n]})"
             for n, v in up.items()]
    lines += [f"  OK    {n:<26s} {v:+.3f}   (value = {FINAL_XHO.iloc[i][n]})"
              for n, v in down.items() if v < 0]
    return "\n".join(lines)


p_use = p_iso
order = np.argsort(-p_use)
tp = [i for i in order if y_hold[i] == 1][:2]        # caught fraud
fp = [i for i in order if y_hold[i] == 0][:1]        # false alarm
fn = [i for i in np.argsort(p_use) if y_hold[i] == 1][:1]  # missed fraud

for tag, idxs in [("CAUGHT FRAUD", tp), ("FALSE ALARM", fp), ("MISSED FRAUD", fn)]:
    for i in idxs:
        print(f"\n=== {tag} ===")
        print(f"  amount ${amt_ho.iloc[i]:,.2f}   calibrated p = {p_use[i]:.2%}"
              f"   expected loss ${p_use[i]*(amt_ho.iloc[i]+CHARGEBACK_FEE):,.2f}"
              f"   review costs ${REVIEW_COST:.0f}")
        print(reason_codes(i))

print("""
The false alarm and the missed fraud are the two worth studying. A false alarm
whose reason codes look reasonable is a threshold problem. One whose reason
codes look absurd is a feature problem, and it tells you which feature.
""")

# %% [markdown]
# ## 10. Stability — does it hold up across slices?
#
# An aggregate PR-AUC can hide a model that works on most traffic and fails on
# a segment. In a regulated setting that failure mode is the one that matters,
# and "I checked" is a materially different answer from "I assume so".

# %%
slices = []
for prod, g in hold_df.groupby("ProductCD", observed=True):
    idx = g.index.to_numpy()
    if y_hold[idx].sum() >= 20:
        slices.append({"slice": f"ProductCD={prod}", "n": len(idx),
                       "fraud_rate": float(y_hold[idx].mean()),
                       "pr_auc": pr_auc(y_hold[idx], p_iso[idx]),
                       "lift": pr_auc(y_hold[idx], p_iso[idx]) / max(y_hold[idx].mean(), 1e-9)})

wk = ((hold_df["day"] - hold_df["day"].min()) // 7).astype(int)
for w, g in hold_df.groupby(wk):
    idx = g.index.to_numpy()
    if y_hold[idx].sum() >= 20:
        slices.append({"slice": f"week {w}", "n": len(idx),
                       "fraud_rate": float(y_hold[idx].mean()),
                       "pr_auc": pr_auc(y_hold[idx], p_iso[idx]),
                       "lift": pr_auc(y_hold[idx], p_iso[idx]) / max(y_hold[idx].mean(), 1e-9)})

SLICE_COLS = ["slice", "n", "fraud_rate", "pr_auc", "lift"]
sl = pd.DataFrame(slices, columns=SLICE_COLS)

# An empty frame built from [] has no columns at all, so every reference below
# would raise rather than report "nothing to slice". Only reachable on a small
# sample, where no segment clears the 20-fraud floor -- which is a filter doing
# its job, not an error.
if sl.empty:
    print(f"No slice reached 20 fraud cases (holdout has {int(y_hold.sum())} "
          f"total). Stability analysis needs more data -- expected on a "
          f"sampled run, and a problem on a full one.")
else:
    print(sl.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4))
wk_rows = sl[sl["slice"].str.startswith("week")]
ax.plot(range(len(wk_rows)), wk_rows["pr_auc"], "o-", lw=2, c=BLUE)
ax.axhline(pr_auc(y_hold, p_iso), ls="--", c="0.4", lw=1, label="holdout overall")
ax.set_xticks(range(len(wk_rows)))
ax.set_xticklabels(wk_rows["slice"], rotation=30, fontsize=8)
ax.set_ylabel("PR-AUC"); ax.legend(fontsize=8)
ax.set_title("Performance across the holdout weeks\n(a downward slope is a retraining argument)")
plt.tight_layout(); plt.savefig(OUT / "10_stability.png"); plt.show()

_lo = sl.pr_auc.min() if not sl.empty else float("nan")
_hi = sl.pr_auc.max() if not sl.empty else float("nan")
print(f"""
PR-AUC ranges {_lo:.4f} to {_hi:.4f} across {len(sl)} slices
(overall {pr_auc(y_hold, p_iso):.4f}).

Compare `lift` rather than raw PR-AUC across slices: a segment with a higher
base rate gets a higher PR-AUC for free, so raw values are not comparable.
A downward trend across weeks is a drift signature and sets the retraining
cadence.
""")

# %% [markdown]
# ## 11. Summary

# %%
res = pd.DataFrame(RESULTS)
cols = [c for c in ["name", "scheme", "pr_auc", "roc_auc", "recall_at_1pct_fpr",
                    "brier", "alerts_per_day", "min_total_cost"] if c in res.columns]
print(res[cols].round(4).to_string(index=False))
res.to_csv(OUT / "results.csv", index=False)

# %% [markdown]
# ## What this notebook argues
#
# **1. The split matters more than the model.** A random k-fold overstates
# PR-AUC by ~46% here, and ~three-quarters of that is the model memorising
# which accounts were compromised — 78% of all fraud sits in accounts that were
# defrauded more than once. Reporting the *decomposition*, rather than asserting
# "I used a time split", is what makes it a measurement.
#
# **2. ROC-AUC hides it.** The same 46% methodology error moves ROC-AUC by ~7%.
# With 96.5% negatives, ROC's false-positive denominator is so large that
# thousands of bad alerts barely register.
#
# **3. Feature engineering dwarfs tuning.** Hyperparameter search and imbalance
# handling together moved PR-AUC by less than one fold standard deviation.
# Past-only velocity and behavioural features moved it by multiples.
#
# **4. Leak-safety is testable.** `verify_past_only` rebuilds features on a
# truncated prefix and fails if any value changes. It caught a real leak in
# this notebook's own frequency encoding — which never touches the label.
# Unsupervised does not mean leak-free.
#
# **5. Calibration is a separate axis from ranking.** Every ranking metric is
# invariant to monotone squashing of the scores, so isotonic regression moves
# PR-AUC by almost nothing and moves Brier a great deal. The moment
# probabilities get multiplied by dollar amounts, that invariance stops being
# harmless — section 8 shows the same decision rule going from *worse than no
# model* to the best configuration in the notebook, with nothing changed but
# the inputs.
#
# **6. The threshold is an economic decision, and it is not a constant.**
# Flag when `p x (amount + fee) > review_cost`, and the implied probability
# cutoff becomes a function of the amount: a $4,000 transaction is worth
# reviewing at p=0.25%, a $12 one needs p>27%. A single global threshold
# cannot express both.
#
# **7. A score without a reason code is not deployable.** TreeSHAP is exact
# for tree ensembles — closed-form, not sampled like KernelSHAP — and
# LightGBM computes it natively. Section 9 turns those values into the two or
# three sentences an analyst actually receives.
#
# ### Next: does deep learning beat this?
#
# A sequence model — GRU or small transformer — over each account's
# transaction history. Note what feature block C had to approximate with
# summary statistics: "how unusual is this amount for this cardholder",
# compressed into a mean and a z-score. A sequence model does not need the
# compression; it sees the trajectory.
#
# That is the one place a neural network has a structural advantage over a
# GBDT on this problem, and it is worth testing properly rather than asserting.
# It gets its own notebook, with the same purged split and the same holdout, so
# the comparison means something.
