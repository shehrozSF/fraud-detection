# Transaction Fraud Detection: Gradient Boosting vs. Deep Learning

An end-to-end study on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
dataset (~590k transactions, 3.5% fraud), built around one question:

> **Does deep learning actually beat gradient boosting on transactional fraud —
> and if so, where?**

The interesting parts are the ones most fraud portfolios skip: leak-safe
validation under temporal drift, calibrated probabilities, and a decision rule
denominated in dollars rather than in F1.

---

## Status

| Phase | State |
|---|---|
| 1. EDA + validation design + baselines | **done** |
| 2. LightGBM + imbalance benchmark | **done** |
| 3. Feature engineering (velocity, leak-safe aggregates) | **done** |
| 4. Deep models (embedding MLP, GRU over card history) | — |
| 5. Calibration, cost thresholding, SHAP | in progress |
| 6. Streamlit demo | — |

## Results

From `results/results.csv`. The logged run is the single-file Kaggle version
([kaggle/fraud_detection_kaggle.py](kaggle/fraud_detection_kaggle.py)) on a
systematic 1-in-4 sample — every 4th row, so the 182-day span is preserved and
only the density drops. The temporal holdout is the last 20% of the timeline:
**29,527 transactions over ~42 days, 1,015 of them fraud (3.44%)**.

| Model | Split | PR-AUC | Recall @ 1% FPR | Alerts/day | Est. cost |
|---|---|---|---|---|---|
| chance | — | 0.034 | — | — | — |
| amount only | temporal holdout | 0.039 | 0.007 | 93 | $115,623 |
| logistic regression | temporal holdout | 0.169 | 0.010 | 131 | $110,825 |
| **LightGBM, tuned** | temporal holdout | **0.455** | **0.369** | **110** | **$92,633** |

Alerts/day and cost are quoted at each model's own cost-minimising threshold
(review $10/alert, chargeback fee $25, no recovery — see `src/config.py`).
LightGBM is 2.7× the linear baseline on PR-AUC, and at its operating point it
raises 16% *fewer* alerts than the logistic regression while catching 11% more
fraud (755 vs. 677 of 1,015).

---

## Phase 1 finding: the split is worth more than the model

The same LightGBM, the same features, three validation schemes:

| Scheme | Account spans folds | Trains on future rows | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| random k-fold | yes | yes | 0.664 | 0.929 |
| group k-fold on account | no | yes | 0.532 | 0.875 |
| purged expanding-window | legitimately | no | **0.462** | 0.862 |

Running all three separates *identity* leakage (the model memorising which
cards are compromised) from *temporal* leakage (training on rows that
postdate the validation period), as a share of the honest score:

| Source | PR-AUC overstatement |
|---|---|
| account identity | +28.7% |
| training on the future | +15.2% |
| **total, random k-fold** | **+43.9%** |

A shuffled k-fold — the tutorial default — would have overstated this model by
almost half, and it was the *validation design*, chosen before any modelling,
that bought the correction. Note how little ROC-AUC moves across the same three
rows (7% against PR-AUC's 30%): the headline metric has to be the sensitive one
or the whole diagnostic is invisible. Reporting the decomposition — rather than
just asserting "I used a time split" — is the point of
[notebooks/02_validation_design.py](notebooks/02_validation_design.py).

---

## Phase 2 finding: the imbalance benchmark is a null result

Six arms against the tuned LightGBM, resampling inside the fold only, under
purged CV on the training block:

| Arm | PR-AUC | Fold SD | Brier | Train rows |
|---|---|---|---|---|
| oversample (duplicate) | 0.486 | 0.045 | 0.0266 | 63,354 |
| SMOTE | 0.481 | 0.051 | 0.0259 | 63,354 |
| undersample 1:10 | 0.480 | 0.051 | 0.0267 | 18,810 |
| undersample + prior fix | 0.480 | 0.051 | 0.0259 | 18,810 |
| none | 0.476 | 0.043 | 0.0261 | 58,900 |
| `scale_pos_weight` | 0.461 | 0.054 | 0.0296 | 58,900 |

**Every arm is within one fold standard deviation (~0.045) of doing nothing.**
The honest headline is that none of them mattered here — gradient boosting
already handles a 3.4% positive rate, because each tree fits the residuals of
the last and those are mostly fraud. What the table *does* show: undersampling
reached the same PR-AUC on under a third of the rows, SMOTE lost to plain
duplication (so its interpolation bought nothing), and `scale_pos_weight` had
the worst Brier because it distorts the probabilities most.

Elkan's prior correction after undersampling leaves PR-AUC identical to four
decimals (0.4795 either way) and moves Brier from 0.0267 to 0.0259 — the
cleanest demonstration in the repo that ranking quality and probability quality
are different things, and that a ranking-only scorecard cannot see the
difference.

---

## Phase 3 finding: one feature block did everything, and it needs a retest

Ablation under purged CV, adding one block at a time
([results/07_ablation.png](results/07_ablation.png)):

| Blocks | PR-AUC |
|---|---|
| raw columns only | ~0.47 |
| + A–F (amount/D-norm, velocity, amount-vs-history, recency, novelty, prior counts) | ~0.47 |
| + G target encoding (`uid_te`, `card1_te`, `email_te`) | ~0.69 |

Blocks A–F each moved PR-AUC by less than 0.01 — inside the ±0.04 fold noise —
and C and F were slightly negative. The phase was predicted to beat
hyperparameter tuning by an order of magnitude and it did not. The likely
reason is that the dataset's hundreds of anonymised C/D/V columns already
encode much of this; the provider built them from the same raw logs.

**Treat the +0.21 from target encoding as an upper bound, not a result.**
`expanding_target_encode` is past-only in *row order*, and
`verify_past_only` passes it, but it consumes each earlier transaction's label
the instant that row goes by. Real fraud labels arrive from chargebacks and
investigations weeks later. If a card was compromised yesterday, production
does not know it yet and `uid_te` does. The prefix test cannot catch this —
the feature genuinely only reads past rows, it just reads past *labels* that
would not have landed yet. The outstanding work is a label-delay variant
(count only labels older than ~30 days) and a re-measurement.

---

## Setup

```bash
pip install -r requirements.txt
```

Get the data (accept the competition rules in the browser once first):

```bash
pip install kaggle
kaggle competitions download -c ieee-fraud-detection -p data/raw
cd data/raw && unzip ieee-fraud-detection.zip
```

No download yet? Every notebook has a `SMOKE = True` flag at the top that
swaps in a synthetic generator with the same structural properties, so the
whole pipeline runs end to end in seconds.

## Running

Notebooks are plain `.py` files using `# %%` cell markers — VS Code renders
them as interactive notebooks with no `.ipynb` to diff. Run them in order:

```
notebooks/01_eda.py                 what shape is this problem
notebooks/02_validation_design.py   measuring what a random split overstates
notebooks/03_baselines.py           amount-only, logistic regression, cost curve
notebooks/04_lightgbm.py            tuning, early stopping, honest importance
notebooks/05_imbalance.py           scale_pos_weight / undersample / SMOTE benchmark
notebooks/06_features.py            past-only features, the leak test, ablation
```

Order matters: 02 writes the locked train/holdout indices to `data/interim/`,
and 04 writes the tuned hyperparameters that 05 and 06 read back.

To get real `.ipynb` files instead: `pip install jupytext && jupytext --to
notebook notebooks/*.py`.

[kaggle/fraud_detection_kaggle.py](kaggle/fraud_detection_kaggle.py) is the
whole pipeline flattened into one self-contained script for a Kaggle kernel,
with a `SAMPLE_ROWS` knob that systematically thins the rows while keeping the
timeline intact. The committed `results.csv` came from it.

## Layout

```
src/
  config.py       paths, seeds, purge gap, and the alert economics
  data.py         loading, memory reduction, pseudo-account id, synthetic fallback
  validation.py   PurgedTimeSeriesSplit + the two wrong splitters, kept for contrast
  evaluate.py     PR-AUC, recall@FPR, calibration, cost curves, experiment log
  models.py       baseline pipelines + LightGBM with nested early stopping
  imbalance.py    scale_pos_weight / undersample / SMOTE, and Elkan prior correction
  features.py     past-only aggregates, encodings, and the leak detector
notebooks/      01-06, the phases in order
kaggle/         the same pipeline as one self-contained script
scripts/        smoke_test.py, to_ipynb.py
data/
  raw/          the Kaggle download (gitignored)
  interim/      locked split indices, tuned params, cached predictions
results/
  results.csv   every experiment, including the ones that lost
  figures/
```

**Leak-safety is tested, not asserted.** `features.verify_past_only` rebuilds
every feature on a truncated prefix of the data and fails if any value
changes — if a feature at row *i* is genuinely past-only, computing it over
`df[:i+1]` must give exactly the value computed over the full frame. It
caught a real leak in this repo's own frequency encoding, which never touches
the label and still read the future. Notebook 06 also runs a deliberately
leaky control, because a test you have never seen fail proves nothing.

**What that test cannot see**, stated here so nobody over-reads it:

- *Self-inclusion.* The prefix `df[:i+1]` still contains row *i*, so a feature
  that uses the current row's own amount or label returns the same value both
  ways and passes. A forgotten `closed="left"` slips straight through. That
  property needs its own assertion — an account's first transaction must have
  a prior count of 0 and a NaN prior mean.
- *Label availability.* Past-in-row-order is not the same as known-at-scoring-
  time once a feature reads labels. See the phase 3 caveat above.

## Design notes

**Why PR-AUC, not ROC-AUC.** With 96.5% negatives, ROC's false-positive axis
has an enormous denominator, so thousands of spurious alerts barely move the
curve. Precision's denominator is small, so PR-AUC stays sensitive to exactly
the errors that flood a review queue.

**Why a purge gap.** A "transactions by this card in the trailing 30 days"
feature computed at time *t* reads rows in [*t*−30d, *t*]. A training row just
before a fold boundary therefore summarises rows sitting inside validation —
a real leak, invisible in the metric. `config.PURGE_DAYS` is 30 and is a
contract, not a preference: it must stay ≥ `features.MAX_WINDOW_DAYS` (it was
raised from 7 when the 30-day velocity block landed, and adding a 60-day
window means moving it again). The gap comes out of the *training* tail, never
out of validation width — purging discards training observations whose feature
windows overlap validation (López de Prado, ch. 7), and an earlier version that
took it from the validation side silently produced 0.3-day validation slivers.

**Why cost, not F1.** F1 assumes a false positive and a false negative cost
the same. Here a false positive costs ten minutes of analyst time and a false
negative costs the transaction amount plus a chargeback fee, so the optimal
threshold is a function of the amount:

```
flag when   p(fraud) × (amount + fee)  >  review_cost
```

A $4,000 transaction is worth reviewing at p = 0.25%; a $12 one needs
p > 27%. A single global cutoff cannot express both. This only works on
calibrated probabilities, which is why calibration gets its own phase.

**Why the account id is a heuristic.** The dataset has no account key. `D1`
is days since the card began transacting, so `day − D1` recovers a
registration date that is constant per card; combined with `card1 + addr1` it
gives a usable pseudo-account. It collides occasionally, which is fine for
grouping folds and for aggregating velocity features.
