"""
VancomycinAI - Phase 3: machine-learning baseline for AUC24
===========================================================
Predicts steady-state AUC24 from information available at the time of TDM:
  - baseline covariates (age, weight, height, SCr, CrCL, sex, ICU)
  - the dosing regimen (dose, interval, infusion duration, number of doses given)
  - the patient's own TDM concentrations and how long after the last dose started
    each sample was taken.

NOT used as features: anything in vanco_truth.csv (true CL, V, AUC). The true AUC24 is
used only as the training label and for scoring.

Evaluation protocol
  - Split is fixed by patient (SPLIT column). Test patients are touched ONCE, at the end.
  - Model comparison and tuning use 5-fold cross-validation GROUPED by patient inside train.

Research use only. Simulated data. Not for clinical dosing.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data")
SEED = 42

ev = pd.read_csv(DATA / "vanco_pk_nonmem.csv")
pts = pd.read_csv(DATA / "vanco_patients.csv")
truth = pd.read_csv(DATA / "vanco_truth.csv")


# --------------------------------------------------------------------------
# 1. One row per patient, using only information available at the TDM time
# --------------------------------------------------------------------------
def build_features(ev, pts):
    rows = []
    for pid, g in ev.groupby("ID"):
        doses = g[g.EVID == 1]
        obs = g[g.EVID == 0].sort_values("TIME")
        last_dose_start = doses.TIME.max()
        f = {"ID": pid, "N_DOSES": len(doses)}
        # every sample is in the final dosing interval: time since the start of the last dose
        t_rel = (obs.TIME - last_dose_start).values
        c = obs.DV.values
        interval = float(pts.loc[pts.ID == pid, "INTERVAL"].iloc[0])
        # late sample (trough-like) = the one closest to the end of the interval
        i_late = int(np.argmax(t_rel))
        f["C_LATE"] = c[i_late]
        f["T_LATE"] = t_rel[i_late]
        f["TIME_TO_NEXT_DOSE"] = interval - t_rel[i_late]
        # early sample (peak-like), only if the patient has two samples
        if len(obs) > 1:
            i_early = int(np.argmin(t_rel))
            f["C_EARLY"] = c[i_early]
            f["T_EARLY"] = t_rel[i_early]
            f["HAS_TWO"] = 1
            # slope between the two samples -> elimination rate estimate
            dt = t_rel[i_late] - t_rel[i_early]
            f["KE_OBS"] = np.log(c[i_early] / c[i_late]) / dt if dt > 0.5 and c[i_early] > 0 and c[i_late] > 0 else np.nan
        else:
            f["C_EARLY"] = f["T_EARLY"] = f["KE_OBS"] = np.nan
            f["HAS_TWO"] = 0
        rows.append(f)
    tdm = pd.DataFrame(rows)

    df = pts.merge(tdm, on="ID")
    df["SEX_NUM"] = (df.SEX == "F").astype(int)
    df["DAILY_DOSE"] = df.DOSE * 24 / df.INTERVAL
    df["DOSE_PER_KG"] = df.DOSE / df.WT
    df["LOG_DAILY_DOSE"] = np.log(df.DAILY_DOSE)
    df["LOG_CLEFF"] = np.log(df.CRCL)           # kidney function on a log scale
    df["C_LATE_X_INT"] = df.C_LATE * df.INTERVAL  # crude concentration-to-exposure scale
    # fill missing second-sample features with column medians learned later (inside the pipeline)
    return df


FEATURES = ["AGE", "SEX_NUM", "WT", "HT", "SCR", "CRCL", "ICU", "DOSE", "INTERVAL", "INFDUR",
            "N_DOSES", "DAILY_DOSE", "DOSE_PER_KG", "LOG_DAILY_DOSE", "LOG_CLEFF",
            "C_LATE", "T_LATE", "TIME_TO_NEXT_DOSE", "C_LATE_X_INT",
            "HAS_TWO", "C_EARLY", "T_EARLY", "KE_OBS"]
# note: SPLIT, INFECTION (string) and everything from vanco_truth.csv are excluded.

df = build_features(ev, pts).merge(truth[["ID", "TRUE_AUC24", "TRUE_CL"]], on="ID")
train = df[df.SPLIT == "train"].reset_index(drop=True)
test = df[df.SPLIT == "test"].reset_index(drop=True)


def fill_missing(X, medians):
    X = X.copy()
    for col, m in medians.items():
        X[col] = X[col].fillna(m)
    return X


# --------------------------------------------------------------------------
# 2. Metrics (same definitions as the Julia PopPK script)
# --------------------------------------------------------------------------
def metrics(pred, truth_):
    err = pred - truth_
    in_t = lambda a: (a >= 400) & (a <= 600)
    return dict(
        RMSE=np.sqrt(np.mean(err ** 2)),
        MAE=np.mean(np.abs(err)),
        BIAS_PCT=100 * np.mean(err / truth_),
        WITHIN20=100 * np.mean(np.abs(err / truth_) <= 0.20),
        TARGET_AGREE=100 * np.mean(in_t(pred) == in_t(truth_)),
    )


# --------------------------------------------------------------------------
# 3. Models. Target = log(AUC24) so errors are proportional and predictions stay positive.
# --------------------------------------------------------------------------
def make_models():
    return {
        "Ridge (linear)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))),
        "Random forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                               random_state=SEED, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.03,
                                                       max_depth=3, subsample=0.8,
                                                       random_state=SEED),
    }


# --------------------------------------------------------------------------
# 4. Grouped cross-validation inside TRAIN only
# --------------------------------------------------------------------------
y_tr = np.log(train.TRUE_AUC24.values)
cv = GroupKFold(n_splits=5)
oof = {name: np.zeros(len(train)) for name in make_models()}

for fold, (i_fit, i_val) in enumerate(cv.split(train, y_tr, groups=train.ID)):
    X_fit, X_val = train.loc[i_fit, FEATURES], train.loc[i_val, FEATURES]
    med = X_fit.median().to_dict()                  # medians from the fitting fold only
    X_fit, X_val = fill_missing(X_fit, med), fill_missing(X_val, med)
    for name, model in make_models().items():
        model.fit(X_fit, y_tr[i_fit])
        oof[name][i_val] = model.predict(X_val)

print("=== 5-fold grouped CV on TRAIN (out-of-fold predictions) ===")
cv_rows = {}
for name, p in oof.items():
    cv_rows[name] = metrics(np.exp(p), train.TRUE_AUC24.values)
print(pd.DataFrame(cv_rows).T.round(1).to_string())

# --------------------------------------------------------------------------
# 5. Final fit on all TRAIN, ONE evaluation on TEST
# --------------------------------------------------------------------------
med_all = train[FEATURES].median().to_dict()
X_tr = fill_missing(train[FEATURES], med_all)
X_te = fill_missing(test[FEATURES], med_all)

test_out = test[["ID", "TRUE_AUC24"]].copy()
te_rows = {}
fitted = {}
for name, model in make_models().items():
    model.fit(X_tr, y_tr)
    fitted[name] = model
    pred = np.exp(model.predict(X_te))
    test_out["AUC24_" + name.split()[0].upper()] = pred
    te_rows[name] = metrics(pred, test.TRUE_AUC24.values)

print("\n=== HELD-OUT TEST (120 patients, evaluated once) ===")
print(pd.DataFrame(te_rows).T.round(1).to_string())

# feature importance (random forest) - which patient characteristics drive exposure
imp = pd.Series(fitted["Random forest"].feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n=== Top random-forest feature importances ===")
print(imp.head(8).round(3).to_string())

test_out.to_csv(DATA / "ml_test_auc24.csv", index=False)
oof_df = train[["ID", "TRUE_AUC24"]].copy()
for name, p in oof.items():
    oof_df["AUC24_" + name.split()[0].upper()] = np.exp(p)
oof_df.to_csv(DATA / "ml_train_oof_auc24.csv", index=False)
print("\nSaved data/ml_test_auc24.csv and data/ml_train_oof_auc24.csv")
