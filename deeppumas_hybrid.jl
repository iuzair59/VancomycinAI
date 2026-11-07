# VancomycinAI - DeepPumas hybrid: neural-network covariate model inside the PopPK model
# --------------------------------------------------------------------------------------
# Run in the VancomycinAI-deep folder (the folder that contains data/), in the DeepPumas
# Julia environment:   include("deeppumas_hybrid.jl")
# Research use only. Simulated data. Not for clinical dosing.
#
# Idea (DeepPumas covariate workflow)
#   1. Fit the same one-compartment PopPK model as before on the TRAIN patients.
#   2. Build a target: each train patient's empirical-Bayes random effects (eta) vs their covariates.
#   3. Train a small neural network that predicts those etas from covariates
#      (age, weight, height, SCr, CrCl, sex, ICU) - i.e. it learns the covariate effects
#      the simple power-law model could not express.
#   4. Embed the network in the model (augment), refit the random-effect variances,
#      and score the held-out TEST patients exactly as in the earlier PopPK script.
# The network sees covariates only. TDM concentrations still enter through each patient's
# own empirical-Bayes estimate, as in standard PopPK.

using DeepPumas
using LinearAlgebra
using Statistics
using CSV, DataFrames

# ---- 1. Data -------------------------------------------------------------------
df  = CSV.read(joinpath("data", "vanco_pk_nonmem.csv"), DataFrame)
pts = CSV.read(joinpath("data", "vanco_patients.csv"), DataFrame)
tru = CSV.read(joinpath("data", "vanco_truth.csv"), DataFrame)

covcols = [:WT, :CRCL, :AGE, :HT, :SCR, :SEX_NUM, :ICU]
keep    = vcat([:ID, :TIME, :EVID, :AMT, :RATE, :CMT, :DV], covcols)
train_df = df[df.SPLIT .== "train", keep]
test_df  = df[df.SPLIT .== "test", keep]

pop = read_pumas(train_df;
    id = :ID, time = :TIME, amt = :AMT, evid = :EVID, cmt = :CMT, rate = :RATE,
    covariates = covcols, observations = [:DV])
test_pop = read_pumas(test_df;
    id = :ID, time = :TIME, amt = :AMT, evid = :EVID, cmt = :CMT, rate = :RATE,
    covariates = covcols, observations = [:DV])
println("Train patients: ", length(pop), "   Test patients: ", length(test_pop))

# ---- 2. Base PopPK model (same as pumas_fit_onecmt.jl) ---------------------------
model = @model begin
    @param begin
        tvcl    ∈ RealDomain(lower = 0.1, init = 4.5)
        tvv     ∈ RealDomain(lower = 1.0, init = 40.0)
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
        Vc = tvv * (WT / 70) * exp(η[2])
    end
    @dynamics Central1
    @derived begin
        cp := @. Central / Vc
        DV ~ @. Normal(cp, sqrt(σ_add^2 + (σ_prop * cp)^2))
    end
end

println("\nFitting base PopPK model...")
fpm = fit(model, pop, init_params(model), MAP(FOCE()))
println("Base estimates:")
display(coef(fpm))

# ---- 3. Neural-network covariate model ----------------------------------------
try
    set_mlp_backend(:simplechains)      # fast CPU backend for small networks
catch err
    println("(set_mlp_backend skipped: ", err, ")")
end

target = preprocess(fpm; covs = (:AGE, :WT, :CRCL, :HT, :SCR, :SEX_NUM, :ICU))
println("\nNeural-network target: ", numinputs(target), " inputs -> ", numoutputs(target), " outputs")

nn  = MLPDomain(numinputs(target), 8, 8, (numoutputs(target), identity); reg = L2(1.0))
println("Training the covariate network...")
fnn = fit(nn, target; optim_options = (; epochs = 2000))

aug = augment(fpm, fnn)

println("\nRefitting the random-effect variances with the network fixed...")
fpm_hyb = fit(aug.model, aug.data, coef(aug), MAP(FOCE());
              constantcoef = filter(i -> i !== :Ω, keys(coef(aug))))
println("Hybrid model estimates (network weights omitted):")
display(NamedTuple{(:tvcl, :tvv, :θcrcl, :Ω, :σ_add, :σ_prop)}(
    map(k -> coef(fpm_hyb)[k], (:tvcl, :tvv, :θcrcl, :Ω, :σ_add, :σ_prop))))

# ---- 4. Score the held-out TEST patients -----------------------------------------
# Parameters are FIXED from the train fit; each test patient's own samples give their eta.
# (Pumas 2.8.1 does not accept the empirical_bayes(...) argument, so ask icoef for FOCE directly;
#  fall back to the plain 3-argument form if that is not accepted either.)
θb   = coef(fpm)
ic_b = try
    icoef(model, test_pop, θb, FOCE())
catch
    icoef(model, test_pop, θb)
end
cl_b = [first(x.u.CL) for x in ic_b]

θh   = coef(fpm_hyb)
ic_h = try
    icoef(fpm_hyb.model, test_pop, θh, FOCE())
catch
    icoef(fpm_hyb.model, test_pop, θh)
end
cl_h = [first(x.u.CL) for x in ic_h]

ids = parse.(Int, [s.id for s in test_pop])
res = DataFrame(ID = ids, CL_BASE = cl_b, CL_DEEP = cl_h)
res = leftjoin(res, pts[:, [:ID, :DOSE, :INTERVAL]], on = :ID)
res = leftjoin(res, tru[:, [:ID, :TRUE_AUC24]], on = :ID)
res.DAILY_DOSE         = @. res.DOSE * 24 / res.INTERVAL
res.AUC24_POPPK_BASE   = @. res.DAILY_DOSE / res.CL_BASE
res.AUC24_DEEPPUMAS    = @. res.DAILY_DOSE / res.CL_DEEP

# Sanity check: the base model should reproduce the earlier PopPK test result (MAE about 77)
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

println("\n=== HELD-OUT TEST (120 patients) ===")
report("PopPK base (re-fit here)", res.AUC24_POPPK_BASE, res.TRUE_AUC24)
report("DeepPumas hybrid", res.AUC24_DEEPPUMAS, res.TRUE_AUC24)

CSV.write(joinpath("data", "deeppumas_test_auc24.csv"), res[:, [:ID, :TRUE_AUC24, :AUC24_POPPK_BASE, :AUC24_DEEPPUMAS]])

# ---- 5. Side by side with the Python methods (if their file is present) -----------
hp = joinpath("data", "hybrid_test_predictions.csv")
if isfile(hp)
    other = CSV.read(hp, DataFrame)
    both  = innerjoin(res[:, [:ID, :AUC24_DEEPPUMAS]], other, on = :ID)
    println("\n=== SAME 120 TEST PATIENTS: all methods ===")
    report("DeepPumas hybrid", both.AUC24_DEEPPUMAS, both.TRUE_AUC24)
    for c in names(other)
        c in ("ID", "TRUE_AUC24") && continue
        report(c, both[!, c], both.TRUE_AUC24)
    end
else
    println("\n(Copy data/hybrid_test_predictions.csv into this folder's data/ to see the full comparison.)")
end
println("\nSaved data/deeppumas_test_auc24.csv")
