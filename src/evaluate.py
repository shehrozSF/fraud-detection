"""Metrics that survive a 0.2%-positive class, plus alert economics.

Accuracy is meaningless here and ROC-AUC is close to it: with 96.5% negatives,
the false-positive axis is so generously scaled that a model can look strong
while drowning the review queue. Everything below is either precision-recall
based or denominated in dollars and analyst-hours.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import config as C


# --------------------------------------------------------------------------
# Ranking quality
# --------------------------------------------------------------------------
def pr_auc(y_true, y_score) -> float:
    """Average precision. The headline number for this project.

    Unlike ROC-AUC its baseline is the positive rate itself, so it moves when
    a model actually improves ranking *among the rare class*.
    """
    return float(average_precision_score(y_true, y_score))


def recall_at_fpr(y_true, y_score, target_fpr: float = C.TARGET_FPR):
    """Recall when the false-positive rate is pinned at `target_fpr`.

    The operational question: "if the review team can tolerate flagging 1% of
    legitimate traffic, what share of fraud do we catch?" Returns
    (recall, threshold).
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    i = np.searchsorted(fpr, target_fpr, side="right") - 1
    i = int(np.clip(i, 0, len(thr) - 1))
    return float(tpr[i]), float(thr[i])


def precision_at_k(y_true, y_score, k: int):
    """Precision in the top-k highest-risk transactions -- the analyst queue.

    This is what a fraud lead actually feels: of the k alerts my team works
    today, how many were real?
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score))[:k]
    return float(y_true[order].mean())


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def brier(y_true, y_prob) -> float:
    """Mean squared error of the probabilities themselves.

    Ranking metrics are invariant to any monotone squashing of the scores;
    Brier is not. Once probabilities get multiplied by dollar amounts (below),
    being well-ranked is no longer enough -- they have to be *right*.
    """
    return float(brier_score_loss(y_true, y_prob))


def reliability_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs. observed fraud rate per probability decile."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = pd.cut(y_prob, edges, labels=False, duplicates="drop")

    return (pd.DataFrame({"bin": b, "p": y_prob, "y": y_true})
            .groupby("bin", observed=True)
            .agg(n=("y", "size"), mean_predicted=("p", "mean"),
                 observed=("y", "mean"))
            .reset_index())


# --------------------------------------------------------------------------
# Alert economics
# --------------------------------------------------------------------------
def expected_cost(y_true, y_score, amount, threshold: float,
                  review_cost: float = C.REVIEW_COST,
                  chargeback_fee: float = C.CHARGEBACK_FEE,
                  recovery_rate: float = C.RECOVERY_RATE) -> dict:
    """Total cost of operating at a fixed probability cutoff.

    Cost model, stated plainly so it can be argued with:
      * every alert costs `review_cost` in analyst time, right or wrong;
      * a flagged fraud is assumed stopped -- we pay only the review;
      * a missed fraud costs amount*(1-recovery) + chargeback_fee;
      * a legitimate transaction that is *not* flagged is free.

    Not modelled: customer friction from a false decline. Real programmes
    weight that heavily; leaving it out is a simplification, not an oversight.
    """
    y_true = np.asarray(y_true)
    amount = np.asarray(amount, dtype="float64")
    flagged = np.asarray(y_score) >= threshold

    loss_if_missed = amount * (1.0 - recovery_rate) + chargeback_fee
    missed = (~flagged) & (y_true == 1)

    review_total = flagged.sum() * review_cost
    fraud_loss = loss_if_missed[missed].sum()

    caught = flagged & (y_true == 1)
    return {
        "threshold": threshold,
        "n_alerts": int(flagged.sum()),
        "alert_rate": float(flagged.mean()),
        "n_caught": int(caught.sum()),
        "n_missed": int(missed.sum()),
        "recall": float(caught.sum() / max((y_true == 1).sum(), 1)),
        "precision": float(caught.sum() / max(flagged.sum(), 1)),
        "review_cost": float(review_total),
        "fraud_loss": float(fraud_loss),
        "total_cost": float(review_total + fraud_loss),
        "value_recall": float(amount[caught].sum()
                              / max(amount[y_true == 1].sum(), 1e-9)),
    }


def cost_curve(y_true, y_score, amount, n_points: int = 200, **kw) -> pd.DataFrame:
    """Sweep the cutoff and tabulate cost. The minimum is your operating point."""
    qs = np.linspace(0.0, 1.0, n_points)
    thresholds = np.unique(np.quantile(np.asarray(y_score), qs))
    return pd.DataFrame([expected_cost(y_true, y_score, amount, t, **kw)
                         for t in thresholds])


def expected_value_decision(y_score, amount,
                            review_cost: float = C.REVIEW_COST,
                            chargeback_fee: float = C.CHARGEBACK_FEE,
                            recovery_rate: float = C.RECOVERY_RATE) -> np.ndarray:
    """Flag when reviewing is cheaper than the expected loss of not reviewing.

        p * (amount*(1-recovery) + fee)  >  review_cost

    Note what this does: the implied probability cutoff becomes a *function of
    the amount*. A $12 transaction needs far more confidence to be worth an
    analyst's ten minutes than a $4,000 one does. A single global threshold
    cannot express that, and most fraud projects never try.

    Only sound on calibrated probabilities -- see `brier`.
    """
    p = np.asarray(y_score, dtype="float64")
    exposure = np.asarray(amount, dtype="float64") * (1.0 - recovery_rate) + chargeback_fee
    return p * exposure > review_cost


# --------------------------------------------------------------------------
# One-call summary + experiment log
# --------------------------------------------------------------------------
def evaluate_all(y_true, y_score, amount=None, n_days: float | None = None,
                 target_fpr: float = C.TARGET_FPR) -> dict:
    """Every headline number for one set of predictions."""
    y_true = np.asarray(y_true)
    rec, thr = recall_at_fpr(y_true, y_score, target_fpr)

    out = {
        "pr_auc": pr_auc(y_true, y_score),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "base_rate": float(y_true.mean()),
        f"recall_at_{target_fpr:g}fpr": rec,
        "threshold_at_fpr": thr,
        "brier": brier(y_true, np.clip(y_score, 0, 1)),
    }
    out["lift_over_base"] = out["pr_auc"] / max(out["base_rate"], 1e-9)

    if amount is not None:
        curve = cost_curve(y_true, y_score, amount)
        best = curve.loc[curve["total_cost"].idxmin()]
        out["best_threshold"] = float(best["threshold"])
        out["min_total_cost"] = float(best["total_cost"])
        out["alerts_at_best"] = int(best["n_alerts"])
        out["recall_at_best"] = float(best["recall"])
        if n_days:
            out["alerts_per_day"] = out["alerts_at_best"] / n_days
    return out


def log_result(name: str, scheme: str, metrics: dict, notes: str = "",
               path=C.RESULTS_CSV) -> pd.DataFrame:
    """Append one experiment to results/results.csv.

    Log every run, including the ones that lose. The README table at the end
    is built from this file, and "SMOTE underperformed class weighting" is
    only a credible claim if you can show the row.
    """
    row = {"name": name, "scheme": scheme,
           "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
           **{k: (round(v, 6) if isinstance(v, float) else v)
              for k, v in metrics.items()},
           "notes": notes}

    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_pr_curves(curves: dict, ax=None, base_rate: float | None = None):
    """curves: {label: (y_true, y_score)}."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for label, (yt, ys) in curves.items():
        prec, rec, _ = precision_recall_curve(yt, ys)
        ax.plot(rec, prec, lw=2, label=f"{label}  (AP={pr_auc(yt, ys):.4f})")
    if base_rate is not None:
        ax.axhline(base_rate, ls="--", c="0.5", lw=1,
                   label=f"chance ({base_rate:.3%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall")
    ax.legend(fontsize=8)
    return ax
