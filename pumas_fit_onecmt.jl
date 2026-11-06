# VancomycinAI - Phase 2, script 1: one-compartment PopPK model in Pumas
# ---------------------------------------------------------------------
# Run from the VancomycinAI folder (the folder that contains data/).
# Research use only. Simulated data. Not for clinical dosing.
#
# What this does:
#   1. Reads the simulated TDM data and keeps TRAIN patients only (IDs 1-480).
#   2. Fits a one-compartment IV infusion model: CL ~ CrCL (power), V ~ weight (linear),
#      random effects on CL and V, combined (additive + proportional) error.
#      (Deliberately simpler than the two-compartment "truth" model.)
#   3. Estimates each patient's CL from their own TDM samples (empirical Bayes).
#   4. Converts CL to AUC24 = daily dose / CL and compares it with the TRUE AUC24.
#   5. Saves results to data/poppk_train_auc24.csv

using Pumas
using LinearAlgebra
using Statistics
using CSV, DataFrames

# ---- 1. Load data -----------------------------------------------------------
df  = CSV.read(joinpath("data", "vanco_pk_nonmem.csv"), DataFrame)
pts = CSV.read(joinpath("data", "vanco_patients.csv"), DataFrame)
tru = CSV.read(joinpath("data", "vanco_truth.csv"), DataFrame)

train_df = df[df.SPLIT .== "train", [:ID, :TIME, :EVID, :AMT, :RATE, :CMT, :DV, :WT, :CRCL, :AGE]]
println("Train rows: ", nrow(train_df), "  patients: ", length(unique(train_df.ID)))

pop = read_pumas(
    train_df;
    id = :ID, time = :TIME, amt = :AMT, evid = :EVID, cmt = :CMT, rate = :RATE,
    covariates = [:WT, :CRCL, :AGE],
    observations = [:DV],
)

# ---- 2. Model ---------------------------------------------------------------
model = @model begin
    @param begin
        tvcl    ∈ RealDomain(lower = 0.1, init = 4.5)    # L/h at CrCL = 100
        tvv     ∈ RealDomain(lower = 1.0, init = 40.0)   # L at 70 kg
        θcrcl   ∈ RealDomain(lower = 0.0, init = 0.7)    # CrCL exponent on CL
        Ω       ∈ PDiagDomain([0.09, 0.09])              # IIV on CL and V
        σ_add   ∈ RealDomain(lower = 0.001, init = 1.0)  # mg/L
        σ_prop  ∈ RealDomain(lower = 0.001, init = 0.1)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @covariates WT CRCL

    @pre begin
        CL = tvcl * (CRCL / 100)^θcrcl * exp(η[1])
        Vc = tvv * (WT / 70) * exp(η[2])
    end

    @dynamics Central1

    @derived begin
        cp := @. Central / Vc
        DV ~ @. Normal(cp, sqrt(σ_add^2 + (σ_prop * cp)^2))
    end
end

# ---- 3. Fit -----------------------------------------------------------------
println("\nFitting model (this can take a few minutes)...")
fpm = fit(model, pop, init_params(model), FOCE())

println("\n=== Estimated parameters ===")
display(coef(fpm))
println("\n=== Fit summary ===")
try
    println("OFV = ", round(-2 * loglikelihood(fpm), digits = 1),
            "   AIC = ", round(aic(fpm), digits = 1),
            "   BIC = ", round(bic(fpm), digits = 1))
catch err
    println("(fit statistics not available: ", err, ")")
end

# ---- 4. Individual CL (empirical Bayes) and AUC24 ---------------------------
θ    = coef(fpm)
ebes = empirical_bayes(fpm)
ids  = parse.(Int, [s.id for s in pop])

est = DataFrame(
    ID     = ids,
    eta_cl = [e.η[1] for e in ebes],
    eta_v  = [e.η[2] for e in ebes],
)

est = leftjoin(est, pts[:, [:ID, :CRCL, :WT, :DOSE, :INTERVAL]], on = :ID)
est = leftjoin(est, tru[:, [:ID, :TRUE_CL, :TRUE_AUC24]], on = :ID)

est.CL_TYPICAL   = @. θ.tvcl * (est.CRCL / 100)^θ.θcrcl
est.CL_HAT       = @. est.CL_TYPICAL * exp(est.eta_cl)
est.DAILY_DOSE   = @. est.DOSE * 24 / est.INTERVAL
est.AUC24_POPPK  = @. est.DAILY_DOSE / est.CL_HAT        # uses the patient's own TDM
est.AUC24_TYPICAL = @. est.DAILY_DOSE / est.CL_TYPICAL   # covariates only, no TDM

# ---- 5. Compare with truth (TRAIN set, in-sample = optimistic) ---------------
function report(label, pred, truth)
    err  = pred .- truth
    rmse = sqrt(mean(err .^ 2))
    mae  = mean(abs.(err))
    bias = 100 * mean(err ./ truth)
    w20  = 100 * mean(abs.(err ./ truth) .<= 0.20)
    tp   = (pred .>= 400) .& (pred .<= 600)
    tt   = (truth .>= 400) .& (truth .<= 600)
    acc  = 100 * mean(tp .== tt)
    println(rpad(label, 22),
            "RMSE=", round(rmse, digits = 1),
            "  MAE=", round(mae, digits = 1),
            "  bias%=", round(bias, digits = 1),
            "  within20%=", round(w20, digits = 1),
            "  target-agreement%=", round(acc, digits = 1))
end

println("\n=== AUC24 vs TRUE AUC24 (train set, in-sample) ===")
report("PopPK + TDM (EBE)", est.AUC24_POPPK, est.TRUE_AUC24)
report("Typical (covariates)", est.AUC24_TYPICAL, est.TRUE_AUC24)

CSV.write(joinpath("data", "poppk_train_auc24.csv"), est)
println("\nSaved data/poppk_train_auc24.csv")
