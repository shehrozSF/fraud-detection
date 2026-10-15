"""Paths, constants, and the economics of a fraud alert.

Everything that is a *choice* lives here so experiments stay comparable.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
RESULTS_CSV = RESULTS_DIR / "results.csv"

for _d in (RAW_DIR, INTERIM_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset facts (IEEE-CIS Fraud Detection)
# --------------------------------------------------------------------------
TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"        # seconds since an unknown origin
AMOUNT_COL = "TransactionAmt"

# The competition never published the origin date. The community consensus
# (recovered by aligning transaction-count seasonality to weekday/holiday
# patterns) is 2017-12-01. We only use it to make plots readable -- no model
# input depends on it, so a wrong guess costs nothing.
DT_ORIGIN = "2017-12-01"

SECONDS_PER_DAY = 24 * 60 * 60

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
# Fraction of the timeline held out as the final (future) validation block.
VAL_FRACTION = 0.20

# Purge gap between the end of a training fold and the start of its validation
# fold. Rolling/velocity features look backwards over a window; without a gap,
# a training row near the boundary can have been computed from rows that sit
# inside validation.
#
# This is a CONTRACT, not a preference: it must stay >= the longest trailing
# window in features.py (MAX_WINDOW_DAYS). Raised from 7 to 30 on day 9 when
# the 30-day velocity block was added. Add a 60-day window later and this has
# to move again, or the guarantee silently lapses.
PURGE_DAYS = 30

N_SPLITS = 5
SEED = 42

# --------------------------------------------------------------------------
# Alert economics  (Track D -- used from day 1 so metrics stay honest)
# --------------------------------------------------------------------------
# Analyst time to review one alert. ~10 min of a loaded $60/hr fraud analyst.
REVIEW_COST = 10.0

# Fixed cost of a confirmed fraudulent transaction on top of the amount:
# card-network chargeback fee + operational handling.
CHARGEBACK_FEE = 25.0

# Share of the transaction amount actually recoverable after a missed fraud.
# 0.0 = the full amount is lost. Kept explicit rather than hidden in a formula.
RECOVERY_RATE = 0.0

# Operating point used for headline reporting when a single number is needed.
TARGET_FPR = 0.01
