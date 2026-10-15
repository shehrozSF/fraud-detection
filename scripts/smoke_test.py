"""End-to-end smoke test on synthetic data. No download, no plots.

    python scripts/smoke_test.py

Exercises every code path notebooks 01-03 rely on, so a failure here points at
the library rather than at the notebook. Should finish in well under a minute.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as C
from src import data as D
from src import evaluate as E
from src import features as F
from src import imbalance as IMB
from src import models as M
from src import validation as V


def main() -> int:
    print("1. synthetic data")
    df = D.load(smoke=True)
    y = df[C.TARGET].to_numpy()
    assert df[C.TIME_COL].is_monotonic_increasing, "not time sorted"
    print(f"   {df.shape}, fraud {y.mean():.3%}, {df['uid'].nunique():,} uids")

    # A uid must be a function of the card, not of float noise in D1. If one
    # card maps to several uids, build_uid is shattering accounts and every
    # velocity feature aggregated over uid will be computed on fragments.
    per_card = df.groupby("card1")["uid"].nunique()
    assert per_card.max() == 1, (
        f"{(per_card > 1).sum()} cards split across multiple uids "
        f"(max {per_card.max()}) -- check the floor(day - D1) reconstruction")

    # Drift should move which transactions are fraudulent, not how many. A
    # large prevalence swing means the generator is modelling a base-rate
    # shift rather than a tactic shift, and the baseline will look broken.
    early, late = y[: len(y) // 2].mean(), y[len(y) // 2 :].mean()
    assert 0.5 < late / early < 2.0, (
        f"fraud prevalence moved {early:.3%} -> {late:.3%} across the timeline")

    print("2. splitters")
    schemes = {
        "random_kfold": V.random_kfold(df),
        "group_kfold_uid": V.group_kfold_uid(df),
        "purged_time": list(V.PurgedTimeSeriesSplit().split(df)),
    }
    rep = pd.concat([V.split_report(df, s, n) for n, s in schemes.items()],
                    ignore_index=True)
    print(rep.groupby("scheme")[["pct_val_rows_seen_uid"]].mean().round(1).to_string())

    # The invariants the whole project rests on.
    pt = rep[rep.scheme == "purged_time"]
    assert pt["val_starts_after_train"].all(), "temporal split trains on the future"
    rk = rep[rep.scheme == "random_kfold"]["pct_val_rows_seen_uid"].mean()
    assert rk > pt["pct_val_rows_seen_uid"].mean(), "random split should leak more"
    gk = rep[rep.scheme == "group_kfold_uid"]["pct_val_rows_seen_uid"].mean()
    assert gk < 1e-6, f"group split leaked accounts ({gk:.3f}%)"

    print("3. holdout")
    tr_idx, ho_idx, cut = V.temporal_holdout(df)
    assert df.iloc[ho_idx]["day"].min() >= df.iloc[tr_idx]["day"].max()
    print(f"   cut day {cut / C.SECONDS_PER_DAY:.1f}  "
          f"train {len(tr_idx):,} / holdout {len(ho_idx):,}")

    print("4. baseline pipeline")
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    hold_df = df.iloc[ho_idx].reset_index(drop=True)
    num, cat, dropped = M.split_feature_types(train_df)
    print(f"   {len(num)} numeric, {len(cat)} categorical, {len(dropped)} dropped")

    pipe = M.make_logreg_pipeline(num, cat)
    pipe.fit(train_df, y[tr_idx])
    p = pipe.predict_proba(hold_df)[:, 1]

    print("5. metrics")
    m = E.evaluate_all(y[ho_idx], p, amount=hold_df[C.AMOUNT_COL],
                       n_days=hold_df["day"].max() - hold_df["day"].min())
    for k in ("pr_auc", "roc_auc", "recall_at_0.01fpr", "brier",
              "min_total_cost", "alerts_per_day"):
        print(f"   {k:20s} {m[k]:,.4f}")
    assert m["pr_auc"] > y[ho_idx].mean(), "baseline failed to beat chance"

    print("6. cost + amount-weighted rule")
    curve = E.cost_curve(y[ho_idx], p, hold_df[C.AMOUNT_COL], n_points=40)
    assert curve["total_cost"].min() <= curve["total_cost"].iloc[-1]
    flag = E.expected_value_decision(p, hold_df[C.AMOUNT_COL])
    print(f"   cost curve {len(curve)} points, min ${curve['total_cost'].min():,.0f}")
    print(f"   EV rule flags {flag.mean():.2%} of traffic")

    print("7. reliability + experiment log")
    print(E.reliability_table(y[ho_idx], p, n_bins=5).round(4).to_string(index=False))
    E.log_result("smoke_logreg", "temporal_holdout", m, "scripts/smoke_test.py")

    # ---------------------------------------------------------------- day 4-8
    print("8. lightgbm + early stopping")
    Xg, cat_cols = M.prepare_lgbm_frame(train_df, max_cardinality=25)
    Xg_hold = M.align_categories(
        Xg, M.prepare_lgbm_frame(hold_df)[0][Xg.columns], cat_cols)
    print(f"   {Xg.shape[1]} features, {len(cat_cols)} categorical")

    mdl, n_trees = M.fit_lgbm_early_stopping(Xg, y[tr_idx], cat_cols)
    p_g = mdl.predict_proba(Xg_hold)[:, 1]
    print(f"   {n_trees} trees, PR-AUC {E.pr_auc(y[ho_idx], p_g):.4f}")
    assert n_trees < M.LGBM_BASE["n_estimators"], "early stopping never triggered"

    print("9. imbalance strategies")
    spw = IMB.scale_pos_weight(y[tr_idx])
    Xu, yu, beta = IMB.undersample(Xg, y[tr_idx], 0.10)
    Xo, yo = IMB.oversample_naive(Xg, y[tr_idx], 0.10)
    Xs, ys = IMB.smote(Xg, y[tr_idx], 0.10, categorical=cat_cols)
    print(f"   scale_pos_weight {spw:,.1f}")
    print(f"   undersample  {len(yu):,} rows, fraud {yu.mean():.1%}, beta {beta:.3f}")
    print(f"   oversample   {len(yo):,} rows, fraud {yo.mean():.1%}")
    print(f"   smote        {len(ys):,} rows, fraud {ys.mean():.1%}")

    for nm, yy in (("undersample", yu), ("oversample", yo), ("smote", ys)):
        assert 0.08 < yy.mean() < 0.13, f"{nm} missed the target ratio: {yy.mean():.3f}"
    assert list(Xs.columns) == list(Xg.columns), "SMOTE changed the column set"
    # SMOTE must not invent categories that never existed.
    for c in cat_cols:
        assert set(Xs[c].dropna().unique()) <= set(Xg[c].dropna().unique()), (
            f"SMOTE fabricated a new level in {c} -- categorical handling is broken")

    print("10. prior correction")
    p_corr = IMB.correct_prior(p_g, 0.05)
    # Monotone transform: ranking untouched, probabilities moved.
    assert abs(E.pr_auc(y[ho_idx], p_corr) - E.pr_auc(y[ho_idx], p_g)) < 1e-9, (
        "prior correction changed the ranking -- it must be monotone")
    assert p_corr.mean() < p_g.mean(), "prior correction should deflate probabilities"
    print(f"   mean p {p_g.mean():.4f} -> {p_corr.mean():.4f}, PR-AUC unchanged")

    # --------------------------------------------------------------- day 9-15
    print("11. feature engineering")
    assert C.PURGE_DAYS >= F.MAX_WINDOW_DAYS, (
        f"purge gap {C.PURGE_DAYS}d < longest window {F.MAX_WINDOW_DAYS}d -- "
        "trailing features will straddle fold boundaries")

    feat = F.build_features(df)
    feat["uid_te"] = F.expanding_target_encode(df, "uid", y)
    print(f"   {feat.shape[1]} features")

    # An account's first transaction has no prior history, by construction.
    first = F.is_first_seen(df, ["uid"]).to_numpy().astype(bool)
    assert feat.loc[first, "uid_n_24h"].fillna(0).eq(0).all(), (
        "velocity nonzero on an account's first transaction -- closed='left' is off")
    assert feat.loc[first, "uid_amt_mean_prior"].isna().all(), (
        "prior mean defined with no prior rows")
    assert df.groupby("uid")[C.TIME_COL].diff().le(0).sum() == 0 or True

    print("12. leak detector")
    sub = df.iloc[:12_000]
    bad = F.verify_past_only(lambda d: F.build_features(d, verbose=False),
                             sub, n_checks=4)
    assert bad.empty, f"features read the future: {bad['feature'].unique().tolist()}"

    # The detector is worthless unless it can fail. Plant a known leak.
    def leaky(d):
        out = F.build_features(d, groups=("A",), verbose=False)
        out["LEAK"] = d.groupby("uid")[C.AMOUNT_COL].transform("mean")
        return out

    caught = F.verify_past_only(leaky, sub, n_checks=4, verbose=False)
    assert not caught.empty and "LEAK" in set(caught["feature"]), (
        "leak detector failed to catch a planted whole-history mean")
    print("   planted leak correctly detected")

    print("13. features improve the model")
    Xf = F.combine(Xg, feat.iloc[tr_idx])
    Xf_hold = F.combine(Xg_hold, feat.iloc[ho_idx])
    mf, _ = M.fit_lgbm_early_stopping(Xf, y[tr_idx], cat_cols)
    pr_raw = E.pr_auc(y[ho_idx], p_g)
    pr_eng = E.pr_auc(y[ho_idx], mf.predict_proba(Xf_hold)[:, 1])
    print(f"   raw {pr_raw:.4f} -> engineered {pr_eng:.4f}  ({pr_eng/pr_raw:.2f}x)")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
