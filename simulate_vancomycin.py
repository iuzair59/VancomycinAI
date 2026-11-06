"""
VancomycinAI - Phase 1: simulated vancomycin TDM dataset
=========================================================
Generates virtual adult inpatients with vancomycin dosing and sparse TDM samples.

DESIGN PRINCIPLE (deliberate model misspecification)
----------------------------------------------------
The "truth" model is a TWO-compartment model with NONLINEAR covariate effects and
an UNMEASURED-in-the-model covariate (ICU status affects V1). The PopPK model you fit
in Pumas will start as a simple ONE-compartment model with a power-law CrCL effect,
so it is structurally wrong in realistic ways. This gives ML / hybrid approaches
something genuine to gain, and avoids the circular "fit the model you simulated" trap.

GROUND TRUTH
------------
true AUC24 (steady state) = daily dose / individual true CL
This is the reference for ALL model comparisons. It is never given to a model as an input.

OUTPUTS (written to ./data)
---------------------------
vanco_pk_nonmem.csv   NONMEM/Pumas-style event dataset (dose + observation records)
vanco_patients.csv    One row per patient: baseline covariates + dosing regimen
vanco_truth.csv       True individual parameters and true AUC24 (ground truth, evaluation only)

RESEARCH USE ONLY. Simulated data. Not for clinical dosing.
"""
import numpy as np
import pandas as pd
from scipy.linalg import expm
from pathlib import Path

SEED = 20261105
N_PATIENTS = 600
OUT = Path("data")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Typical values and variability (documented assumptions)
# ---------------------------------------------------------------------------
TV_CL = 5.0      # L/h at CrCL = 100 mL/min, age 60 (reference patient)
TV_V1 = 30.0     # L central volume (70 kg reference)
TV_Q = 6.0       # L/h inter-compartmental clearance
TV_V2 = 40.0     # L peripheral volume
OMEGA_CL = 0.30  # approx CV on log scale
OMEGA_V1 = 0.25
OMEGA_Q = 0.30
OMEGA_V2 = 0.30
PROP_ERR = 0.10  # proportional residual error (10%)
ADD_ERR = 1.0    # additive residual error (mg/L)

# ---------------------------------------------------------------------------
# 2. Patient covariates
# ---------------------------------------------------------------------------
def simulate_patients(n):
    age = np.clip(rng.normal(60, 16, n), 18, 95).round().astype(int)
    sex = rng.choice(["M", "F"], size=n, p=[0.55, 0.45])
    height = np.where(sex == "M", rng.normal(174, 7, n), rng.normal(161, 6.5, n))
    bmi = np.clip(rng.lognormal(np.log(27), 0.22, n), 16, 55)
    weight = (bmi * (height / 100) ** 2).round(1)
    icu = rng.binomial(1, 0.30, n)

    # latent "true" renal function declines with age; SCr is a noisy readout of it
    gfr_true = np.clip(
        rng.lognormal(np.log(95), 0.35, n) * (1 - 0.006 * (age - 40).clip(min=0)), 8, 200
    )
    gfr_true = gfr_true * np.where(icu == 1, rng.lognormal(-0.10, 0.25, n), 1.0)
    scr_raw = (88 - 0.4 * (age - 40)) / gfr_true * np.where(sex == "M", 1.0, 0.85)
    scr = np.clip(scr_raw * rng.lognormal(0, 0.12, n) * 1.0, 0.4, 6.0).round(2)

    # Cockcroft-Gault CrCL (mL/min) using total body weight, as would be calculated in practice
    crcl = ((140 - age) * weight) / (72 * scr)
    crcl = np.where(sex == "F", crcl * 0.85, crcl)
    crcl = np.clip(crcl, 5, 250).round(1)

    infection = rng.choice(
        ["bacteraemia", "pneumonia", "skin_soft_tissue", "bone_joint", "other"],
        size=n, p=[0.30, 0.20, 0.25, 0.10, 0.15],
    )
    return pd.DataFrame(dict(ID=np.arange(1, n + 1), AGE=age, SEX=sex, WT=weight,
                             HT=height.round(0), SCR=scr, CRCL=crcl, ICU=icu,
                             INFECTION=infection)), gfr_true


# ---------------------------------------------------------------------------
# 3. Dosing regimen (rule-based, mimics typical hospital practice)
# ---------------------------------------------------------------------------
def assign_regimen(df):
    # loading-ish maintenance 15 mg/kg rounded to nearest 250 mg, capped at 2000 mg
    dose = (np.round(15 * df.WT / 250) * 250).clip(500, 2000)
    # interval by CrCL band (as many protocols do)
    interval = np.select([df.CRCL >= 90, df.CRCL >= 50, df.CRCL >= 25], [8, 12, 24], default=48)
    # real-world heterogeneity: some clinicians deviate
    flip = rng.random(len(df)) < 0.12
    interval = np.where(flip & (interval == 12), 8, interval)
    interval = np.where(flip & (interval == 8), 12, interval)
    infdur = np.where(dose <= 1000, 1.0, 2.0)  # hours; 1 g/h max rate rule
    return dose.astype(float), interval.astype(int), infdur


# ---------------------------------------------------------------------------
# 4. Individual true parameters (NONLINEAR covariate effects = misspecification)
# ---------------------------------------------------------------------------
def true_parameters(df):
    n = len(df)
    # saturating (Emax-like) CrCL effect on CL instead of a simple power law
    crcl_eff = 0.35 + 0.65 * (df.CRCL / 100) ** 0.9 / ((df.CRCL / 100) ** 0.9 + 0.35)
    crcl_eff = crcl_eff / (0.35 + 0.65 / 1.35)  # rescale so effect = 1.0 exactly at CRCL = 100
    cl_cov = crcl_eff * (1 - 0.004 * (df.AGE - 60))
    v1_cov = (df.WT / 70) ** 0.75 * np.where(df.ICU == 1, 1.30, 1.0)  # ICU effect on V1
    q_cov = (df.WT / 70) ** 0.75
    v2_cov = df.WT / 70

    CL = TV_CL * cl_cov * np.exp(rng.normal(0, OMEGA_CL, n))
    V1 = TV_V1 * v1_cov * np.exp(rng.normal(0, OMEGA_V1, n))
    Q = TV_Q * q_cov * np.exp(rng.normal(0, OMEGA_Q, n))
    V2 = TV_V2 * v2_cov * np.exp(rng.normal(0, OMEGA_V2, n))
    return CL.clip(0.3, 20), V1.clip(8, 150), Q.clip(1, 25), V2.clip(10, 200)


# ---------------------------------------------------------------------------
# 5. Exact two-compartment multi-dose solver (piecewise-constant infusion, matrix exponential)
# ---------------------------------------------------------------------------
def simulate_conc(CL, V1, Q, V2, dose, tau, dur, n_doses, obs_times):
    k10, k12, k21 = CL / V1, Q / V1, Q / V2
    A = np.array([[-(k10 + k12), k21], [k12, -k21]])
    # dose start times
    starts = np.arange(n_doses) * tau
    # event grid: dose starts, infusion ends, observation times
    grid = sorted(set(list(starts) + list(starts + dur) + list(obs_times)))
    x = np.zeros(2)
    t_prev = 0.0
    out = {}
    for t in grid:
        dt = t - t_prev
        if dt > 0:
            # rate during (t_prev, t): active if inside an infusion
            rate = 0.0
            for s in starts:
                if s <= t_prev + 1e-9 and t_prev + 1e-9 < s + dur:
                    rate = dose / dur
                    break
            if rate > 0:
                # x(t) = e^{A dt} x + A^{-1}(e^{A dt} - I) b, b=[rate,0]
                E = expm(A * dt)
                x = E @ x + np.linalg.solve(A, (E - np.eye(2)) @ np.array([rate, 0.0]))
            else:
                x = expm(A * dt) @ x
        out[t] = x[0] / V1
        t_prev = t
    return np.array([out[t] for t in obs_times])


# ---------------------------------------------------------------------------
# 6. Build dataset
# ---------------------------------------------------------------------------
def main():
    pts, _ = simulate_patients(N_PATIENTS)
    dose, tau, dur = assign_regimen(pts)
    pts["DOSE"], pts["INTERVAL"], pts["INFDUR"] = dose, tau, dur
    CL, V1, Q, V2 = true_parameters(pts)

    # reference exposure: steady-state AUC24 = daily dose / CL
    daily_dose = dose * (24 / tau)
    auc24_true = daily_dose / CL

    rows = []
    sample_info = []
    for i, p in pts.iterrows():
        n_doses = rng.integers(4, 9)  # patient has received 4-8 doses before TDM window ends
        tau_i, dur_i, dose_i = float(p.INTERVAL), float(p.INFDUR), float(p.DOSE)
        # dose records
        for d in range(n_doses):
            rows.append(dict(ID=p.ID, TIME=d * tau_i, EVID=1, AMT=dose_i, RATE=dose_i / dur_i,
                             CMT=1, DV=np.nan, DOSE_NO=d + 1))
        # TDM sampling: sparse, at the LAST dose interval (realistic, near steady state)
        last_start = (n_doses - 1) * tau_i
        times = []
        # trough: within 30 min before next dose (i.e. end of final interval) - 85% of patients
        # peak-ish: 1-2h post end of infusion - 55% of patients
        if rng.random() < 0.85:
            times.append(last_start + tau_i - rng.uniform(0.1, 0.5))
        if rng.random() < 0.55:
            times.append(last_start + dur_i + rng.uniform(1.0, 2.0))
        if not times:
            times.append(last_start + tau_i - rng.uniform(0.1, 0.5))
        times = sorted(times)
        conc = simulate_conc(CL[i], V1[i], Q[i], V2[i], dose_i, tau_i, dur_i, n_doses, times)
        for t, c in zip(times, conc):
            err = rng.normal(0, np.sqrt((PROP_ERR * c) ** 2 + ADD_ERR ** 2))
            dv = max(c + err, 0.2)  # assay LLOQ floor
            rows.append(dict(ID=p.ID, TIME=round(t, 3), EVID=0, AMT=np.nan, RATE=np.nan,
                             CMT=1, DV=round(dv, 2), DOSE_NO=np.nan))
        sample_info.append(len(times))

    ev = pd.DataFrame(rows)
    ev = ev.merge(pts[["ID", "AGE", "SEX", "WT", "HT", "SCR", "CRCL", "ICU", "INFECTION"]], on="ID")
    ev["SEX_NUM"] = (ev.SEX == "F").astype(int)
    ev = ev.sort_values(["ID", "TIME", "EVID"], ascending=[True, True, False]).reset_index(drop=True)

    truth = pd.DataFrame(dict(ID=pts.ID, TRUE_CL=CL.round(3), TRUE_V1=V1.round(2),
                              TRUE_Q=Q.round(3), TRUE_V2=V2.round(2),
                              TRUE_AUC24=auc24_true.round(1)))
    truth["IN_TARGET_400_600"] = truth.TRUE_AUC24.between(400, 600).astype(int)

    # train / test split at PATIENT level (never split within a patient -> leakage)
    # "temporal" validation proxy: last 20% of IDs treated as a later admission cohort
    pts["SPLIT"] = np.where(pts.ID <= int(0.8 * N_PATIENTS), "train", "test")

    ev = ev.merge(pts[["ID", "SPLIT"]], on="ID")
    ev.to_csv(OUT / "vanco_pk_nonmem.csv", index=False)
    pts.to_csv(OUT / "vanco_patients.csv", index=False)
    truth.to_csv(OUT / "vanco_truth.csv", index=False)
    print(f"Wrote {len(pts)} patients, {int((ev.EVID == 0).sum())} concentrations, "
          f"{int((ev.EVID == 1).sum())} dose records to {OUT.resolve()}")


if __name__ == "__main__":
    main()
