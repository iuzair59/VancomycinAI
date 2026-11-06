"""
VancomycinAI - Phase 5: dosing-regimen simulation
=================================================
Question: if each model's estimate of a patient's clearance were used to choose a regimen,
how often would the patient's TRUE steady-state AUC24 land in 400-600 mg*h/L?

Method (held-out test patients only)
  1. Each method gives an AUC24 for the patient's CURRENT regimen. Because exposure is
     linear in dose, the implied clearance is CL_hat = daily dose / AUC24_hat.
  2. For every candidate regimen (dose x interval) the predicted AUC24 is
        AUC24_pred = candidate daily dose / CL_hat
  3. The regimen whose predicted AUC24 is closest to 500 is "selected" (ties -> lower daily dose).
  4. The selected regimen is scored with the TRUE clearance (from the simulation):
        AUC24_true = candidate daily dose / CL_true
     and we count how often that lands in 400-600 (and below / above).
  5. Reference points: the regimen the patient actually received, and an oracle that knows the true CL.

This is a research simulation on simulated data. It does NOT produce dosing recommendations.
Steady state and linear kinetics are assumed (true in the simulation, a simplification in practice).
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
TARGET_LO, TARGET_HI, TARGET_MID = 400.0, 600.0, 500.0
DOSES = np.arange(500, 2001, 250)          # mg
INTERVALS = np.array([8, 12, 24])          # h
MAX_DAILY_DOSE = 4500                      # mg/day cap on candidate regimens

truth = pd.read_csv(DATA / "vanco_truth.csv")[["ID", "TRUE_CL", "TRUE_AUC24"]]
pts = pd.read_csv(DATA / "vanco_patients.csv")[["ID", "DOSE", "INTERVAL", "SPLIT"]]
pred = pd.read_csv(DATA / "hybrid_test_predictions.csv")

df = pred.merge(truth[["ID", "TRUE_CL"]], on="ID").merge(pts[["ID", "DOSE", "INTERVAL"]], on="ID")
df["DAILY_DOSE_CURRENT"] = df.DOSE * 24 / df.INTERVAL

CANDIDATES = pd.DataFrame(
    [(d, t, d * 24 / t) for d in DOSES for t in INTERVALS if d * 24 / t <= MAX_DAILY_DOSE],
    columns=["DOSE", "INTERVAL", "DAILY"],
)

METHOD_COLS = {
    "PopPK + TDM": "PopPK + TDM",
    "ML (gradient boosting)": "ML (gradient boosting)",
    "Hybrid A (stacked)": "Hybrid A (stacked)",
    "Hybrid B (residual)": "Hybrid B (residual)",
}


def classify(auc):
    return np.where(auc < TARGET_LO, "below", np.where(auc > TARGET_HI, "above", "in target"))


rows, per_patient = [], []
for _, p in df.iterrows():
    rec = dict(ID=int(p.ID), CURRENT_DOSE=p.DOSE, CURRENT_INTERVAL=int(p.INTERVAL),
               TRUE_AUC24_CURRENT=p.TRUE_AUC24)
    # reference: oracle that knows the true clearance
    pred_true = CANDIDATES.DAILY.values / p.TRUE_CL
    j = int(np.argmin(np.abs(pred_true - TARGET_MID) + 1e-6 * CANDIDATES.DAILY.values))
    rec["ORACLE_DOSE"], rec["ORACLE_INTERVAL"] = CANDIDATES.DOSE[j], int(CANDIDATES.INTERVAL[j])
    rec["ORACLE_TRUE_AUC24"] = CANDIDATES.DAILY[j] / p.TRUE_CL
    for m, col in METHOD_COLS.items():
        cl_hat = p.DAILY_DOSE_CURRENT / p[col]
        pa = CANDIDATES.DAILY.values / cl_hat
        k = int(np.argmin(np.abs(pa - TARGET_MID) + 1e-6 * CANDIDATES.DAILY.values))
        rec[f"{m}|DOSE"], rec[f"{m}|INTERVAL"] = CANDIDATES.DOSE[k], int(CANDIDATES.INTERVAL[k])
        rec[f"{m}|PRED_AUC24"] = pa[k]
        rec[f"{m}|TRUE_AUC24"] = CANDIDATES.DAILY[k] / p.TRUE_CL
    per_patient.append(rec)
pp = pd.DataFrame(per_patient)

summary = []


def add(label, auc_true):
    c = classify(auc_true)
    summary.append(dict(
        strategy=label,
        pct_in_target=round(100 * np.mean(c == "in target"), 1),
        pct_below=round(100 * np.mean(c == "below"), 1),
        pct_above=round(100 * np.mean(c == "above"), 1),
        median_abs_dev_from_500=round(float(np.median(np.abs(auc_true - TARGET_MID))), 1),
    ))


add("Regimen actually received", pp.TRUE_AUC24_CURRENT.values)
for m in METHOD_COLS:
    add(f"Model-guided: {m}", pp[f"{m}|TRUE_AUC24"].values)
add("Oracle (knows true clearance)", pp.ORACLE_TRUE_AUC24.values)
summ = pd.DataFrame(summary)

print(f"Held-out test patients: {len(pp)}; candidate regimens: {len(CANDIDATES)} "
      f"(doses {DOSES.min()}-{DOSES.max()} mg x intervals {[int(x) for x in INTERVALS]} h)")
print("\nShare of patients whose TRUE steady-state AUC24 would fall in 400-600:\n")
print(summ.to_string(index=False))

pp.to_csv(DATA / "dosing_sim_patients.csv", index=False)
summ.to_csv(DATA / "dosing_sim_summary.csv", index=False)
CANDIDATES.to_csv(DATA / "dosing_candidates.csv", index=False)
print("\nSaved data/dosing_sim_patients.csv, dosing_sim_summary.csv, dosing_candidates.csv")
