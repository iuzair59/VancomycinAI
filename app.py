"""
VancomycinAI dashboard (Streamlit)
Run from the VancomycinAI folder:   streamlit run app.py
RESEARCH USE ONLY. Simulated data. Not for clinical dosing.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).parent / "data"
TARGET_LO, TARGET_HI = 400, 600
TV_V_70KG = 49.08          # typical volume (L at 70 kg) from the Pumas one-compartment fit

# Reference categorical palette (slots 1-2) and neutral ink
BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#898781", "#c3c2b7"

METHODS = ["PopPK + TDM", "ML (gradient boosting)", "Hybrid A (stacked)", "Hybrid B (residual)"]

st.set_page_config(page_title="VancomycinAI", layout="wide")


# ----------------------------------------------------------------------------- data
@st.cache_data
def load():
    d = {
        "pred": pd.read_csv(DATA / "hybrid_test_predictions.csv"),
        "pts": pd.read_csv(DATA / "vanco_patients.csv"),
        "truth": pd.read_csv(DATA / "vanco_truth.csv"),
        "ev": pd.read_csv(DATA / "vanco_pk_nonmem.csv"),
        "boot": pd.read_csv(DATA / "bootstrap_comparison.csv"),
        "sim_pat": pd.read_csv(DATA / "dosing_sim_patients.csv"),
        "sim_sum": pd.read_csv(DATA / "dosing_sim_summary.csv"),
    }
    return d


def metrics(pred, truth):
    err = pred - truth
    in_t = lambda a: (a >= TARGET_LO) & (a <= TARGET_HI)
    return {
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAE": np.mean(np.abs(err)),
        "Bias %": 100 * np.mean(err / truth),
        "Within ±20 %": 100 * np.mean(np.abs(err / truth) <= 0.20),
        "Target group correct %": 100 * np.mean(in_t(pred) == in_t(truth)),
    }


def conc_profile(t, dose_events, cl, v):
    """One-compartment IV infusion, superposition. dose_events = [(start_h, dose_mg, dur_h)]."""
    k = cl / v
    c = np.zeros_like(t, dtype=float)
    for s, dose, dur in dose_events:
        rate = dose / dur
        dt = t - s
        during = (dt > 0) & (dt <= dur)
        after = dt > dur
        c[during] += rate / cl * (1 - np.exp(-k * dt[during]))
        c[after] += rate / cl * (1 - np.exp(-k * dur)) * np.exp(-k * (dt[after] - dur))
    return c


def infusion_hours(dose):
    return 1.0 if dose <= 1000 else 2.0


def altair_defaults(chart):
    return (chart.configure_axis(gridColor=MUTED, gridOpacity=0.18, domainColor=GRID,
                                 tickColor=GRID, labelColor=MUTED, titleColor=MUTED)
            .configure_view(strokeWidth=0)
            .configure_legend(orient="top", labelColor=MUTED, titleColor=MUTED))


D = load()
pred, pts, truth, ev = D["pred"], D["pts"], D["truth"], D["ev"]

# ----------------------------------------------------------------------------- header
st.title("VancomycinAI")
st.caption("Hybrid population-PK and machine-learning estimation of vancomycin AUC24 · "
           "Pumas one-compartment model, gradient boosting, and two hybrids")
st.warning(
    "**Research output only. Not for clinical use.** All patients here are simulated. "
    "Nothing on this page is a dosing recommendation for any real patient.",
    icon="⚠️",
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Model comparison", "Patient explorer", "Dosing simulation", "About & limits"])

# ----------------------------------------------------------------------------- tab 1
with tab1:
    st.subheader("How well does each method estimate AUC24?")
    st.caption(f"{len(pred)} held-out test patients, evaluated once. "
               "Reference = true AUC24 from the simulation (mg·h/L).")
    rows = {m: metrics(pred[m].values, pred.TRUE_AUC24.values) for m in METHODS}
    mt = pd.DataFrame(rows).T.round(1)
    st.dataframe(mt.style.format("{:.1f}"), width="stretch")

    st.markdown("**Predicted vs true AUC24** — the dashed line is perfect agreement. "
                "Dotted lines mark the 400–600 target.")
    cols = st.columns(2)
    for i, m in enumerate(METHODS):
        df_m = pd.DataFrame({"True AUC24": pred.TRUE_AUC24, "Predicted AUC24": pred[m],
                             "Patient": pred.ID})
        pts_c = alt.Chart(df_m).mark_circle(size=45, opacity=0.75, color=BLUE).encode(
            x=alt.X("True AUC24:Q", scale=alt.Scale(domain=[0, 1600]), title="True AUC24"),
            y=alt.Y("Predicted AUC24:Q", scale=alt.Scale(domain=[0, 1600]), title="Predicted AUC24"),
            tooltip=["Patient:N", alt.Tooltip("True AUC24:Q", format=".0f"),
                     alt.Tooltip("Predicted AUC24:Q", format=".0f")])
        diag = alt.Chart(pd.DataFrame({"x": [0, 1600]})).mark_line(
            color=MUTED, strokeDash=[5, 4]).encode(x="x:Q", y="x:Q")
        bands = alt.Chart(pd.DataFrame({"v": [TARGET_LO, TARGET_HI]})).mark_rule(
            color=GRID, strokeDash=[1, 3]).encode(y="v:Q")
        bandsx = alt.Chart(pd.DataFrame({"v": [TARGET_LO, TARGET_HI]})).mark_rule(
            color=GRID, strokeDash=[1, 3]).encode(x="v:Q")
        ch = (bands + bandsx + diag + pts_c).properties(title=m, height=300)
        with cols[i % 2]:
            st.altair_chart(altair_defaults(ch), width="stretch")

    st.markdown("**Is the difference real?** Paired bootstrap over the test patients "
                "(difference in error; negative MAE/RMSE difference = first method better).")
    b = D["boot"].copy()
    b = b[b.metric.isin(["MAE", "RMSE"])]
    b["diff"] = b["diff"].round(1)
    b["95% interval"] = b.ci_low.round(1).astype(str) + " to " + b.ci_high.round(1).astype(str)
    b["clearly different"] = np.where(b.ci_excludes_0, "yes", "no")
    st.dataframe(b[["comparison", "metric", "diff", "95% interval", "clearly different"]]
                 .rename(columns={"diff": "difference"}), width="stretch", hide_index=True)

# ----------------------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Patient explorer")
    c_sel, c_m = st.columns([1, 2])
    pid = c_sel.selectbox("Test patient", pred.ID.tolist(), index=0)
    method = c_m.radio("Estimate clearance with", METHODS, index=3, horizontal=True)

    p = pts[pts.ID == pid].iloc[0]
    pr = pred[pred.ID == pid].iloc[0]
    tr = truth[truth.ID == pid].iloc[0]
    daily = p.DOSE * 24 / p.INTERVAL
    auc_hat = float(pr[method])
    cl_hat = daily / auc_hat
    v_hat = TV_V_70KG * p.WT / 70

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Age / sex", f"{int(p.AGE)} y / {p.SEX}")
    k2.metric("Weight", f"{p.WT:.0f} kg")
    k3.metric("CrCl (Cockcroft-Gault)", f"{p.CRCL:.0f} mL/min")
    k4.metric("Current regimen", f"{p.DOSE:.0f} mg q{int(p.INTERVAL)}h")

    r1, r2, r3 = st.columns(3)
    r1.metric(f"Estimated AUC24 · {method}", f"{auc_hat:.0f}")
    r2.metric("Estimated clearance", f"{cl_hat:.1f} L/h")
    r3.metric("True AUC24 (simulation only)", f"{tr.TRUE_AUC24:.0f}")

    # concentration-time profile over the patient's observed dosing history
    pe = ev[ev.ID == pid]
    doses = pe[pe.EVID == 1]
    obs = pe[pe.EVID == 0]
    dur = infusion_hours(p.DOSE)
    events = [(float(r.TIME), float(r.AMT), dur) for r in doses.itertuples()]
    t_end = float(obs.TIME.max()) + 4
    t = np.linspace(0, t_end, 400)
    prof = pd.DataFrame({"Hours since first dose": t,
                         "Concentration (mg/L)": conc_profile(t, events, cl_hat, v_hat),
                         "Series": "Model estimate"})
    ob = pd.DataFrame({"Hours since first dose": obs.TIME, "Concentration (mg/L)": obs.DV,
                       "Series": "Measured TDM sample"})
    band = pd.DataFrame({"lo": [TARGET_LO / 24], "hi": [TARGET_HI / 24]})
    shade = alt.Chart(band).mark_rect(opacity=0.14, color=MUTED).encode(
        y=alt.Y("lo:Q", title="Concentration (mg/L)"), y2="hi:Q")
    line = alt.Chart(prof).mark_line(strokeWidth=2).encode(
        x="Hours since first dose:Q", y="Concentration (mg/L):Q",
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Model estimate", "Measured TDM sample"],
                                                    range=[BLUE, ORANGE]), title=None))
    dots = alt.Chart(ob).mark_circle(size=110, opacity=1).encode(
        x="Hours since first dose:Q", y="Concentration (mg/L):Q",
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Model estimate", "Measured TDM sample"],
                                                    range=[BLUE, ORANGE]), title=None),
        tooltip=[alt.Tooltip("Hours since first dose:Q", format=".1f"),
                 alt.Tooltip("Concentration (mg/L):Q", format=".1f")])
    st.altair_chart(altair_defaults((shade + line + dots).properties(
        height=320, title="Estimated concentration-time curve vs measured samples")),
        width="stretch")
    st.caption("Grey band = mean concentration equivalent to AUC24 400–600 (≈ 16.7–25 mg/L). "
               "The curve is an approximation: one compartment, with the volume taken from the "
               "population fit scaled by weight; the clearance comes from the method selected above.")

    st.markdown("#### Try another regimen")
    a1, a2 = st.columns(2)
    alt_dose = a1.select_slider("Dose (mg)", options=list(range(500, 2501, 250)),
                                value=int(p.DOSE))
    alt_int = a2.select_slider("Interval (h)", options=[8, 12, 24, 48], value=int(p.INTERVAL))
    alt_daily = alt_dose * 24 / alt_int
    alt_auc = alt_daily / cl_hat
    alt_true = alt_daily / tr.TRUE_CL
    cls = "in target" if TARGET_LO <= alt_auc <= TARGET_HI else ("below target" if alt_auc < TARGET_LO
                                                                  else "above target")
    s1, s2, s3 = st.columns(3)
    s1.metric("Predicted steady-state AUC24", f"{alt_auc:.0f}", cls, delta_color="off")
    s2.metric("True AUC24 under this regimen (simulation only)", f"{alt_true:.0f}")
    s3.metric("Regimen", f"{alt_dose} mg q{alt_int}h")
    ev_alt = [(float(i * alt_int), float(alt_dose), infusion_hours(alt_dose))
              for i in range(int(120 // alt_int) + 1)]
    t2 = np.linspace(0, 120, 500)
    cur = pd.DataFrame({"Hours": t2, "Concentration (mg/L)": conc_profile(t2, ev_alt, cl_hat, v_hat)})
    st.write("")
    shade2 = alt.Chart(band).mark_rect(opacity=0.14, color=MUTED).encode(
        y=alt.Y("lo:Q", title="Concentration (mg/L)"), y2="hi:Q")
    ln2 = alt.Chart(cur).mark_line(strokeWidth=2, color=BLUE).encode(
        x=alt.X("Hours:Q", title="Hours since first dose"), y="Concentration (mg/L):Q",
        tooltip=[alt.Tooltip("Hours:Q", format=".0f"), alt.Tooltip("Concentration (mg/L):Q", format=".1f")])
    st.altair_chart(altair_defaults((shade2 + ln2).properties(height=240,
                    title=f"Simulated profile · {alt_dose} mg every {alt_int} h")),
                    width="stretch")

    sp = D["sim_pat"]
    sp = sp[sp.ID == pid]
    if len(sp):
        r = sp.iloc[0]
        sel = f"{int(r[f'{method}|DOSE'])} mg q{int(r[f'{method}|INTERVAL'])}h"
        st.info(f"In the dosing simulation, **{method}** would select **{sel}** for this patient "
                f"(true AUC24 under that regimen: **{r[f'{method}|TRUE_AUC24']:.0f}**). "
                "This is a simulation exercise, not advice.")

    with st.expander("Table view of this patient's TDM samples"):
        st.dataframe(obs[["TIME", "DV"]].rename(columns={"TIME": "hours since first dose",
                                                         "DV": "concentration (mg/L)"}),
                     hide_index=True)

# ----------------------------------------------------------------------------- tab 3
with tab3:
    st.subheader("If a model chose the regimen, how often would exposure be on target?")
    st.caption("Each method's estimate of the patient's clearance picks, from 19 candidate regimens "
               "(500–2000 mg × q8/12/24 h), the one predicted closest to AUC24 500. "
               "The pick is then scored with the patient's TRUE clearance. Steady state, linear kinetics.")
    ss = D["sim_sum"].copy()
    ss["kind"] = np.where(ss.strategy.str.startswith("Model-guided"), "Model-guided", "Reference")
    ss["label"] = ss.strategy.str.replace("Model-guided: ", "", regex=False)
    order = ss.label.tolist()
    bars = alt.Chart(ss).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=20).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=280)),
        x=alt.X("pct_in_target:Q", scale=alt.Scale(domain=[0, 100]), title="% of patients with true AUC24 in 400–600"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Model-guided", "Reference"],
                                                  range=[BLUE, MUTED]), title=None),
        tooltip=["label:N", "pct_in_target:Q", "pct_below:Q", "pct_above:Q"])
    txt = alt.Chart(ss).mark_text(align="left", dx=6, color=MUTED).encode(
        y=alt.Y("label:N", sort=order), x="pct_in_target:Q",
        text=alt.Text("pct_in_target:Q", format=".0f"))
    st.altair_chart(altair_defaults((bars + txt).properties(height=300)), width="stretch")
    show = ss[["label", "pct_in_target", "pct_below", "pct_above", "median_abs_dev_from_500"]].rename(
        columns={"label": "strategy", "pct_in_target": "% in target", "pct_below": "% below",
                 "pct_above": "% above", "median_abs_dev_from_500": "median |AUC24 − 500|"})
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("The 'regimen actually received' row comes from a simple simulated hospital protocol "
               "(≈15 mg/kg, interval by CrCl band) that does not use drug levels, so it is a weak "
               "comparator. The oracle knows each patient's true clearance and is the ceiling.")

# ----------------------------------------------------------------------------- tab 4
with tab4:
    st.subheader("About and limits")
    st.markdown("""
**What this is.** A portfolio research project comparing four ways to estimate vancomycin AUC24:
a Pumas one-compartment population PK model, gradient-boosted machine learning, and two hybrids
(PopPK estimates fed into ML; ML correcting the PopPK estimate).

**Data.** 600 simulated adult inpatients from a two-compartment model with nonlinear covariate
effects and an ICU effect on volume. The PopPK model is deliberately simpler than this truth,
so a correction layer has something to fix. Patients 1–480 train, 481–600 are the held-out test set.

**Main limits**
- Simulated data cannot show clinical validity; external validation on de-identified clinical data is needed.
- The ML models were trained on the true simulated AUC24, which real data does not provide.
- PopPK estimates for training patients are in-sample; cross-fitting would be cleaner.
- One simulated dataset and 120 test patients; the dosing simulation assumes steady state and linear kinetics.
- The concentration-time curves on this page are approximations (one compartment, scaled typical volume).

**Not for clinical use.** Do not use any output here to dose a real patient.
""")
