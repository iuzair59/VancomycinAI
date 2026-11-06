"""
VancomycinAI - Phase 4: hybrid PopPK + ML models, and paired bootstrap comparison
=================================================================================
Inputs (all in ./data):
  vanco_pk_nonmem.csv, vanco_patients.csv, vanco_truth.csv   - simulated data
  poppk_train_auc24.csv, poppk_test_auc24.csv                - Pumas one-compartment PopPK results

Methods compared on the SAME 120 held-out test patients
  1. PopPK + TDM            : Pumas empirical-Bayes AUC24 (parameters fixed from the train fit)
  2. ML (gradient boosting) : features = covariates + regimen + TDM concentrations
  3. Hybrid A (stacked)     : ML features + the PopPK AUC24 / typical AUC24 as extra inputs
  4. Hybrid B (residual)    : PopPK AUC24 x exp(ML-predicted log-correction)

Leakage rules
  - Test patients are used once, at the end.
  - Model selection uses 5-fold CV GROUPED by patient inside the train set.
  - vanco_truth.csv is used only for the training label and for scoring.
  - Known limitation: PopPK values for TRAIN patients come from a model fitted on those same
    patients (in-sample), while test patients' values are out-of-sample. PopPK accuracy was
    similar in both (see README), so the mismatch is small, but cross-fitting would be cleaner.

Research use only. Simulated data. Not for clinical dosing.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
SEED = 42
N_BOOT = 5000

ev = pd.read_csv(DATA / "vanco_pk_nonmem.csv")
pts = pd.read_csv(DATA / "vanco_patients.csv")
truth = pd.read_csv(DATA / "vanco_truth.csv")
pk_tr = pd.read_csv(DATA / "poppk_train_auc24.csv")
pk_te = pd.read_csv(DATA / "poppk_test_auc24.csv")


# ---------------------------------------------------------------------------
# Features available at TDM time (same as ml_baseline.py)
# ---------------------------------------------------------------------------
def build_features(ev, pts):
    rows = []
    for pid, g in ev.groupby("ID"):
        doses = g[g.EVID == 1]
        obs = g[g.EVID == 0].sort_values("TIME")
        last_dose_start = doses.TIME.max()
        f = {"ID": pid, "N_DOSES": len(doses)}
        t_rel = (obs.TIME - last_dose_start).values
        c = obs.DV.values
        interval = float(pts.loc[pts.ID == pid, "INTERVAL"].iloc[0])
        i_late = int(np.argmax(t_rel))
        f.update(C_LATE=c[i_late], T_LATE=t_rel[i_late], TIME_TO_NEXT_DOSE=interval - t_rel[i_late])
        if len(obs) > 1:
            i_early = int(np.argmin(t_rel))
            f.update(C_EARLY=c[i_early], T_EARLY=t_rel[i_early], HAS_TWO=1)
            dt = t_rel[i_late] - t_rel[i_early]
            f["KE_OBS"] = (np.log(c[i_early] / c[i_late]) / dt
                           if dt > 0.5 and c[i_early] > 0 and c[i_late] > 0 else np.nan)
        else:
            f.update(C_EARLY=np.nan, T_EARLY=np.nan, KE_OBS=np.nan, HAS_TWO=0)
        rows.append(f)
    df = pts.merge(pd.DataFrame(rows), on="ID")
    df["SEX_NUM"] = (df.SEX == "F").astype(int)
    df["DAILY_DOSE"] = df.DOSE * 24 / df.INTERVAL
    df["DOSE_PER_KG"] = df.DOSE / df.WT
    df["LOG_DAILY_DOSE"] = np.log(df.DAILY_DOSE)
    df["LOG_CLEFF"] = np.log(df.CRCL)
    df["C_LATE_X_INT"] = df.C_LATE * df.INTERVAL
    return df


BASE = ["AGE", "SEX_NUM", "WT", "HT", "SCR", "CRCL", "ICU", "DOSE", "INTERVAL", "INFDUR",
        "N_DOSES", "DAILY_DOSE", "DOSE_PER_KG", "LOG_DAILY_DOSE", "LOG_CLEFF",
        "C_LATE", "T_LATE", "TIME_TO_NEXT_DOSE", "C_LATE_X_INT",
        "HAS_TWO", "C_EARLY", "T_EARLY", "KE_OBS"]
POPPK_FEATS = ["LOG_AUC_POPPK", "LOG_AUC_TYPICAL", "LOG_RATIO_POPPK_TYPICAL"]

df = build_features(ev, pts).merge(truth[["ID", "TRUE_AUC24"]], on="ID")
pk = pd.concat([pk_tr[["ID", "AUC24_POPPK", "AUC24_TYPICAL"]],
                pk_te[["ID", "AUC24_POPPK", "AUC24_TYPICAL"]]])
df = df.merge(pk, on="ID", how="left")
assert df.AUC24_POPPK.notna().all(), "missing PopPK predictions"
df["LOG_AUC_POPPK"] = np.log(df.AUC24_POPPK)
df["LOG_AUC_TYPICAL"] = np.log(df.AUC24_TYPICAL)
df["LOG_RATIO_POPPK_TYPICAL"] = df.LOG_AUC_POPPK - df.LOG_AUC_TYPICAL

train = df[df.SPLIT == "train"].reset_index(drop=True)
test = df[df.SPLIT == "test"].reset_index(drop=True)
assert len(train) == 480 and len(test) == 120


def metrics(pred, truth_):
    err = pred - truth_
    in_t = lambda a: (a >= 400) & (a <= 600)
    return dict(RMSE=np.sqrt(np.mean(err ** 2)), MAE=np.mean(np.abs(err)),
                BIAS_PCT=100 * np.mean(err / truth_),
                WITHIN20=100 * np.mean(np.abs(err / truth_) <= 0.20),
                TARGET_AGREE=100 * np.mean(in_t(pred) == in_t(truth_)))


def fill(X, med):
    X = X.copy()
    for c, m in med.items():
        X[c] = X[c].fillna(m)
    return X


def gb():
    return GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                     subsample=0.8, random_state=SEED)


def gb_residual():
    # shallower and slower: the correction should be small
    return GradientBoostingRegressor(n_estimators=200, learning_rate=0.02, max_depth=2,
                                     subsample=0.8, min_samples_leaf=10, random_state=SEED)


# method name -> (feature list, builder, how to turn the model output into AUC24)
log_true_tr = np.log(train.TRUE_AUC24.values)
METHODS = {
    "ML (gradient boosting)": dict(feats=BASE, model=gb, target="log_auc"),
    "Hybrid A (stacked)": dict(feats=BASE + POPPK_FEATS, model=gb, target="log_auc"),
    "Hybrid A-ridge": dict(feats=BASE + POPPK_FEATS,
                           model=lambda: make_pipeline(StandardScaler(),
                                                       RidgeCV(alphas=np.logspace(-3, 3, 25))),
                           target="log_auc"),
    "Hybrid B (residual)": dict(feats=BASE + POPPK_FEATS, model=gb_residual, target="residual"),
}


def target_vec(spec, d):
    y = np.log(d.TRUE_AUC24.values)
    return y - d.LOG_AUC_POPPK.values if spec["target"] == "residual" else y


def to_auc(spec, d, raw):
    return np.exp(raw + d.LOG_AUC_POPPK.values) if spec["target"] == "residual" else np.exp(raw)


# ---------------------------------------------------------------------------
# 1. Grouped CV on TRAIN
# ---------------------------------------------------------------------------
print("=== 5-fold grouped CV on TRAIN (out-of-fold) ===")
cv = GroupKFold(n_splits=5)
cv_rows = {}
for name, spec in METHODS.items():
    oof = np.zeros(len(train))
    y = target_vec(spec, train)
    for i_fit, i_val in cv.split(train, y, groups=train.ID):
        med = train.loc[i_fit, spec["feats"]].median().to_dict()
        m = spec["model"]()
        m.fit(fill(train.loc[i_fit, spec["feats"]], med), y[i_fit])
        oof[i_val] = m.predict(fill(train.loc[i_val, spec["feats"]], med))
    cv_rows[name] = metrics(to_auc(spec, train, oof), train.TRUE_AUC24.values)
print(pd.DataFrame(cv_rows).T.round(1).to_string())

# ---------------------------------------------------------------------------
# 2. Fit on all TRAIN, evaluate on TEST once
# ---------------------------------------------------------------------------
preds = {"PopPK + TDM": test.AUC24_POPPK.values}
for name, spec in METHODS.items():
    med = train[spec["feats"]].median().to_dict()
    m = spec["model"]()
    m.fit(fill(train[spec["feats"]], med), target_vec(spec, train))
    preds[name] = to_auc(spec, test, m.predict(fill(test[spec["feats"]], med)))

y_te = test.TRUE_AUC24.values
print("\n=== HELD-OUT TEST (120 patients, evaluated once) ===")
print(pd.DataFrame({k: metrics(v, y_te) for k, v in preds.items()}).T.round(1).to_string())

out = test[["ID", "TRUE_AUC24"]].copy()
for k, v in preds.items():
    out[k] = v
out.to_csv(DATA / "hybrid_test_predictions.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Paired bootstrap over patients (differences, 95% CI)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
n = len(test)
idx = rng.integers(0, n, size=(N_BOOT, n))
in_t = lambda a: (a >= 400) & (a <= 600)


def stat_arrays(p):
    err = np.abs(p - y_te)
    sq = (p - y_te) ** 2
    w20 = (np.abs(p - y_te) / y_te <= 0.20).astype(float)
    tg = (in_t(p) == in_t(y_te)).astype(float)
    return dict(MAE=err, MSE=sq, WITHIN20=w20, TARGET=tg)


S = {k: stat_arrays(v) for k, v in preds.items()}


def boot_diff(a, b, stat):
    """mean over bootstrap samples of (stat_a - stat_b); a negative value favours a for errors."""
    da = S[a][stat][idx].mean(axis=1)
    db = S[b][stat][idx].mean(axis=1)
    if stat == "MSE":
        da, db = np.sqrt(da), np.sqrt(db)   # report as RMSE
    d = da - db
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


pairs = [("ML (gradient boosting)", "PopPK + TDM"),
         ("Hybrid A (stacked)", "PopPK + TDM"),
         ("Hybrid B (residual)", "PopPK + TDM"),
         ("Hybrid A (stacked)", "ML (gradient boosting)"),
         ("Hybrid B (residual)", "ML (gradient boosting)")]
rows = []
for a, b in pairs:
    for stat, label in [("MAE", "MAE"), ("MSE", "RMSE"), ("WITHIN20", "within20 (frac)"),
                        ("TARGET", "target-agree (frac)")]:
        m, lo, hi = boot_diff(a, b, stat)
        rows.append(dict(comparison=f"{a}  minus  {b}", metric=label,
                         diff=round(m, 3), ci_low=round(lo, 3), ci_high=round(hi, 3),
                         ci_excludes_0=bool(lo > 0 or hi < 0)))
res = pd.DataFrame(rows)
print(f"\n=== Paired bootstrap ({N_BOOT} resamples of the 120 test patients) ===")
print("For MAE/RMSE a NEGATIVE difference means the first method is better; for within20 and "
      "target-agree a POSITIVE difference means the first is better.")
print(res.to_string(index=False))
res.to_csv(DATA / "bootstrap_comparison.csv", index=False)
print("\nSaved data/hybrid_test_predictions.csv and data/bootstrap_comparison.csv")
