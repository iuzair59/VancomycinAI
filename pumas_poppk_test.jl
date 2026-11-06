# VancomycinAI - Phase 2, script 2: score the PopPK model on the HELD-OUT test patients
# --------------------------------------------------------------------------------------
# Run from the VancomycinAI folder AFTER pumas_fit_onecmt.jl works.
# The model parameters are fixed from the TRAIN fit. For each test patient, the model
# uses only that patient's own covariates, dosing and TDM samples to estimate clearance.
# Research use only. Simulated data. Not for clinical dosing.

include("pumas_fit_onecmt.jl")   # re-fits on train (takes seconds); keeps model, θ, pop, est, report

println("\n\n==================== TEST-SET EVALUATION ====================")

# ---- 1. Test population -----------------------------------------------------
test_df = df[df.SPLIT .== "test", [:ID, :TIME, :EVID, :AMT, :RATE, :CMT, :DV, :WT, :CRCL, :AGE]]
println("Test rows: ", nrow(test_df), "  patients: ", length(unique(test_df.ID)))

test_pop = read_pumas(
    test_df;
    id = :ID, time = :TIME, amt = :AMT, evid = :EVID, cmt = :CMT, rate = :RATE,
    covariates = [:WT, :CRCL, :AGE],
    observations = [:DV],
)

# ---- 2. Individual clearance with the parameters FIXED from the train fit -------
# Individual parameters = empirical-Bayes estimates (each patient's own TDM samples,
# with the population parameters θ held fixed), then evaluated through the @pre block.
# icoef returns, per patient, a small table whose CL is stored in x.u.CL.

# Safety check: the same steps applied to TRAIN patients must reproduce the
# empirical-Bayes clearances from script 1. If not, do NOT trust the test numbers.
ic_train = icoef(model, pop, θ, empirical_bayes(model, pop, θ, FOCE()))
cl_train_check = [first(x.u.CL) for x in ic_train]
r_ebe = cor(log.(cl_train_check), log.(est.CL_HAT))
r_typ = cor(log.(cl_train_check), log.(est.CL_TYPICAL))
println("\nSanity check on TRAIN: correlation with EBE clearance = ", round(r_ebe, digits = 4),
        " | with typical clearance = ", round(r_typ, digits = 4))
if r_ebe < 0.99
    println("WARNING: icoef did not reproduce the empirical-Bayes clearances. ",
            "Send this output to Claude before using the test numbers.")
end

test_ids = parse.(Int, [s.id for s in test_pop])
ic_test  = icoef(model, test_pop, θ, empirical_bayes(model, test_pop, θ, FOCE()))
cl_test  = [first(x.u.CL) for x in ic_test]

# ---- 3. AUC24 on the test patients ----------------------------------------------
tst = DataFrame(ID = test_ids, CL_HAT = cl_test)
tst = leftjoin(tst, pts[:, [:ID, :CRCL, :DOSE, :INTERVAL]], on = :ID)
tst = leftjoin(tst, tru[:, [:ID, :TRUE_AUC24]], on = :ID)
tst.DAILY_DOSE    = @. tst.DOSE * 24 / tst.INTERVAL
tst.AUC24_POPPK   = @. tst.DAILY_DOSE / tst.CL_HAT
tst.AUC24_TYPICAL = @. tst.DAILY_DOSE / (θ.tvcl * (tst.CRCL / 100)^θ.θcrcl)

println("\n=== HELD-OUT TEST: PopPK (parameters fixed from train) ===")
report("PopPK + TDM (EBE)", tst.AUC24_POPPK, tst.TRUE_AUC24)
report("Typical (covariates)", tst.AUC24_TYPICAL, tst.TRUE_AUC24)

CSV.write(joinpath("data", "poppk_test_auc24.csv"), tst)

# ---- 4. Side-by-side with the machine-learning models -------------------------------
ml_path = joinpath("data", "ml_test_auc24.csv")
if isfile(ml_path)
    ml = CSV.read(ml_path, DataFrame)
    both = innerjoin(tst[:, [:ID, :AUC24_POPPK, :AUC24_TYPICAL]], ml, on = :ID)
    println("\n=== SAME 120 TEST PATIENTS: PopPK vs ML ===")
    report("PopPK + TDM (EBE)", both.AUC24_POPPK, both.TRUE_AUC24)
    for c in names(ml)
        startswith(c, "AUC24_") || continue
        report("ML " * replace(c, "AUC24_" => ""), both[!, c], both.TRUE_AUC24)
    end
else
    println("\n(No data/ml_test_auc24.csv found - put it in the data folder to see the comparison.)")
end
println("\nSaved data/poppk_test_auc24.csv")
