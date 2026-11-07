"""
VancomycinAI - paired bootstrap including the two-compartment PopPK benchmark
=============================================================================
Inputs: data/hybrid_test_predictions.csv (one-compartment PopPK, ML, hybrids)
        data/poppk2cmt_test_auc24.csv    (two-compartment PopPK, Pumas)
Same 120 held-out test patients. Differences are first method minus second method.
For MAE/RMSE a NEGATIVE difference favours the first; for within-20% and target-agreement a
POSITIVE difference favours the first.
Research use only. Simulated data.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
N_BOOT, SEED = 10000, 42

a = pd.read_csv(DATA / "hybrid_test_predictions.csv")
b = pd.read_csv(DATA / "poppk2cmt_test_auc24.csv")[["ID", "AUC24_POPPK2", "AUC24_TYPICAL2"]]
d = a.merge(b, on="ID")
assert len(d) == 120
d = d.rename(columns={"AUC24_POPPK2": "PopPK 2-cmt + TDM"})
y = d.TRUE_AUC24.values
METHODS = ["PopPK + TDM", "PopPK 2-cmt + TDM", "ML (gradient boosting)",
           "Hybrid A (stacked)", "Hybrid B (residual)"]
in_t = lambda x: (x >= 400) & (x <= 600)


def metrics(p):
    e = p - y
    return dict(RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)),
                BIAS_PCT=100 * np.mean(e / y), WITHIN20=100 * np.mean(np.abs(e / y) <= .2),
                TARGET_AGREE=100 * np.mean(in_t(p) == in_t(y)))


print("=== Held-out test (120 patients) ===")
print(pd.DataFrame({m: metrics(d[m].values) for m in METHODS}).T.round(1).to_string())

rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
S = {}
for m in METHODS:
    p = d[m].values
    S[m] = dict(MAE=np.abs(p - y), MSE=(p - y) ** 2,
                WITHIN20=(np.abs(p - y) / y <= .2).astype(float),
                TARGET=(in_t(p) == in_t(y)).astype(float))


def diff(m1, m2, stat):
    x1, x2 = S[m1][stat][idx].mean(1), S[m2][stat][idx].mean(1)
    if stat == "MSE":
        x1, x2 = np.sqrt(x1), np.sqrt(x2)
    dd = x1 - x2
    return dd.mean(), np.percentile(dd, 2.5), np.percentile(dd, 97.5)


pairs = [("PopPK 2-cmt + TDM", "PopPK + TDM"),
         ("ML (gradient boosting)", "PopPK 2-cmt + TDM"),
         ("Hybrid A (stacked)", "PopPK 2-cmt + TDM"),
         ("Hybrid B (residual)", "PopPK 2-cmt + TDM")]
rows = []
for m1, m2 in pairs:
    for stat, lab in [("MAE", "MAE"), ("MSE", "RMSE"), ("WITHIN20", "within20 (frac)"),
                      ("TARGET", "target-agree (frac)")]:
        mean, lo, hi = diff(m1, m2, stat)
        rows.append(dict(comparison=f"{m1}  minus  {m2}", metric=lab, diff=round(mean, 3),
                         ci_low=round(lo, 3), ci_high=round(hi, 3), ci_excludes_0=bool(lo > 0 or hi < 0)))
res = pd.DataFrame(rows)
print(f"\n=== Paired bootstrap ({N_BOOT} resamples) ===")
print(res.to_string(index=False))
res.to_csv(DATA / "bootstrap_vs_twocmt.csv", index=False)

# Where does the one-compartment model fail? Absolute error by true exposure band
band = pd.cut(y, [0, 400, 600, 5000], labels=["<400", "400-600", ">600"])
tab = pd.DataFrame({m: pd.Series(np.abs(d[m].values - y)).groupby(band, observed=True).mean()
                    for m in METHODS}).round(1)
print("\n=== Mean absolute error by TRUE exposure band ===")
print(tab.to_string())
print("\nSaved data/bootstrap_vs_twocmt.csv")
