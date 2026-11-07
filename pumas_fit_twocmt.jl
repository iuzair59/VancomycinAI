# VancomycinAI - benchmark: TWO-compartment PopPK model in Pumas
# ---------------------------------------------------------------
# Why: the data were simulated from a two-compartment model. The earlier PopPK model was
# one-compartment, so part of the hybrids' advantage may just be fixing that structure.
# This script fits the CORRECT structure (still with simple power-law covariate effects and
# no ICU effect, i.e. the covariate model remains wrong) so the comparison is fairer.
#
# Run from a folder that contains data/ (VancomycinAI or VancomycinAI-deep):
#   include("pumas_fit_twocmt.jl")
# Research use only. Simulated data. Not for clinical dosing.

using Pumas
using LinearAlgebra
using Statistics
using CSV, DataFrames

# ---- 1. Data -------------------------------------------------------------------
df  = CSV.read(joinpath("data", "vanco_pk_nonmem.csv"), DataFrame)
pts = CSV.read(joinpath("data", "vanco_patients.csv"), DataFrame)
tru = CSV.read(joinpath("data", "vanco_truth.csv"), DataFrame)

keep = [:ID, :TIME, :EVID, :AMT, :RATE, :CMT, :DV, :WT, :CRCL, :AGE]
train_df = df[df.SPLIT .== "train", keep]
test_df  = df[df.SPLIT .== "test", keep]

pop = read_pumas(train_df;
    id = :ID, time = :TIME, amt = :AMT, evid = :EVID, cmt = :CMT, rate = :RATE,
    covariates = [:WT, :CRCL, :AGE], observations = [:DV])
test_pop = read_pumas(test_df;
    id = :ID, time = :TIME, amt = :AMT, evid = :EVID, cmt = :CMT, rate = :RATE,
    covariates = [:WT, :CRCL, :AGE], observations = [:DV])
println("Train patients: ", length(pop), "   Test patients: ", length(test_pop))

# ---- 2. Two-compartment model --------------------------------------------------
# Random effects on CL and Vc only: with 1-2 samples per patient, Q and Vp cannot get
# their own between-patient variability. Q is fixed effect only; Vp scales with weight.
model2 = @model begin
    @param begin
        tvcl    ∈ RealDomain(lower = 0.1, init = 5.0)
        tvvc    ∈ RealDomain(lower = 1.0, init = 30.0)
        tvq     ∈ RealDomain(lower = 0.1, init = 6.0)
        tvvp    ∈ RealDomain(lower = 1.0, init = 40.0)
        θcrcl   ∈ RealDomain(lower = 0.0, init = 0.7)
        Ω       ∈ PDiagDomain([0.09, 0.09])
        σ_add   ∈ RealDomain(lower = 0.001, init = 1.0)
        σ_prop  ∈ RealDomain(lower = 0.001, init = 0.1)
    end
    @random begin
        η ~ MvNormal(Ω)
    end
    @covariates WT CRCL
    @pre begin
        CL = tvcl * (CRCL / 100)^θcrcl * exp(η[1])
        Vc = tvvc * (WT / 70) * exp(η[2])
        Q  = tvq
        Vp = tvvp * (WT / 70)
    end
    @dynamics Central1Periph1
    @derived begin
        cp := @. Central / Vc
        DV ~ @. Normal(cp, sqrt(σ_add^2 + (σ_prop * cp)^2))
    end
end

println("\nFitting the two-compartment model (a few minutes)...")
fpm2 = fit(model2, pop, init_params(model2), FOCE())
println("\n=== Estimated parameters ===")
display(coef(fpm2))
try
    println("OFV = ", round(-2 * loglikelihood(fpm2), digits = 1),
            "   AIC = ", round(aic(fpm2), digits = 1))
catch err
    println("(fit statistics not available: ", err, ")")
end

# ---- 3. Score the held-out TEST patients (parameters fixed from the train fit) ------
θ  = coef(fpm2)
ic = try
    icoef(model2, test_pop, θ, FOCE())
catch
    icoef(model2, test_pop, θ)
end
cl_hat = [first(x.u.CL) for x in ic]

ids = parse.(Int, [s.id for s in test_pop])
res = DataFrame(ID = ids, CL_HAT = cl_hat)
res = leftjoin(res, pts[:, [:ID, :CRCL, :DOSE, :INTERVAL]], on = :ID)
res = leftjoin(res, tru[:, [:ID, :TRUE_AUC24]], on = :ID)
res.DAILY_DOSE     = @. res.DOSE * 24 / res.INTERVAL
res.AUC24_POPPK2   = @. res.DAILY_DOSE / res.CL_HAT
res.AUC24_TYPICAL2 = @. res.DAILY_DOSE / (θ.tvcl * (res.CRCL / 100)^θ.θcrcl)

function report(label, pred, truth)
    err  = pred .- truth
    in_t = a -> (a .>= 400) .& (a .<= 600)
    println(rpad(label, 30),
            "RMSE=", round(sqrt(mean(err .^ 2)), digits = 1),
            "  MAE=", round(mean(abs.(err)), digits = 1),
            "  bias%=", round(100 * mean(err ./ truth), digits = 1),
            "  within20%=", round(100 * mean(abs.(err ./ truth) .<= 0.20), digits = 1),
            "  target-agreement%=", round(100 * mean(in_t(pred) .== in_t(truth)), digits = 1))
end

println("\n=== HELD-OUT TEST (120 patients): two-compartment PopPK ===")
report("PopPK 2-cmt + TDM (EBE)", res.AUC24_POPPK2, res.TRUE_AUC24)
report("PopPK 2-cmt typical", res.AUC24_TYPICAL2, res.TRUE_AUC24)
CSV.write(joinpath("data", "poppk2cmt_test_auc24.csv"),
          res[:, [:ID, :TRUE_AUC24, :AUC24_POPPK2, :AUC24_TYPICAL2]])

# ---- 4. Side by side with the other methods (if their file is present) -------------
hp = joinpath("data", "hybrid_test_predictions.csv")
if isfile(hp)
    other = CSV.read(hp, DataFrame)
    both  = innerjoin(res[:, [:ID, :AUC24_POPPK2]], other, on = :ID)
    println("\n=== SAME 120 TEST PATIENTS: all methods ===")
    report("PopPK 2-cmt + TDM (EBE)", both.AUC24_POPPK2, both.TRUE_AUC24)
    for c in names(other)
        c in ("ID", "TRUE_AUC24") && continue
        report(c, both[!, c], both.TRUE_AUC24)
    end
else
    println("\n(Copy data/hybrid_test_predictions.csv into data/ to see the full comparison.)")
end
println("\nSaved data/poppk2cmt_test_auc24.csv")
