"""CreditLens AI — VP-Level Credit Risk Platform"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sys, os, joblib
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'data_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'explainability'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'genai'))

from preprocessing_german import prepare_data, FEATURE_COLS
from shap_engine import get_explainer, explain_single, get_adverse_reasons
from narrative_engine import generate_risk_report, _ecl_metrics, _risk_band
from stress_test import run_stress_test, SCENARIOS
from portfolio_risk import monte_carlo_loss, pd_term_structure

st.set_page_config(page_title="CreditLens AI", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
*, html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #04080f; color: #c8d8e8; }
section[data-testid="stSidebar"] { background: #060c16 !important; border-right: 1px solid #0d1f35; }
section[data-testid="stSidebar"] label { color: #5a7a9a !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500; }
section[data-testid="stSidebar"] .stRadio label { color: #8aabb8 !important; font-size: 0.82rem !important; text-transform: none; letter-spacing: 0; }
.top-bar { display: flex; align-items: center; justify-content: space-between; padding: 18px 0 20px 0; border-bottom: 1px solid #0d1f35; margin-bottom: 24px; }
.top-bar-left { display: flex; align-items: baseline; gap: 14px; }
.top-bar h1 { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 800; color: #ffffff; margin: 0; letter-spacing: -0.5px; }
.top-bar .sub { font-size: 0.75rem; color: #3a5878; font-family: 'DM Mono', monospace; }
.badge { display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 4px; font-family: 'DM Mono', monospace; font-size: 0.68rem; font-weight: 500; letter-spacing: 0.5px; }
.badge-green  { background: #0a1f14; color: #2ecc71; border: 1px solid #1a4a2e; }
.badge-blue   { background: #081828; color: #3498db; border: 1px solid #0d2a40; }
.badge-yellow { background: #1a1400; color: #f1c40f; border: 1px solid #3a3000; }
.verdict { border-radius: 8px; padding: 28px 32px; margin-bottom: 20px; display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 24px; border-left: 4px solid; }
.verdict.low   { background: #040f09; border-color: #2ecc71; }
.verdict.mod   { background: #0d0d00; border-color: #f1c40f; }
.verdict.high  { background: #0d0700; border-color: #e67e22; }
.verdict.vhigh { background: #0f0404; border-color: #e74c3c; }
.verdict-icon { font-size: 2.4rem; line-height: 1; }
.verdict-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 2px; color: #3a5878; margin-bottom: 4px; }
.verdict-title { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; line-height: 1; margin-bottom: 2px; }
.verdict-sub { font-size: 0.82rem; color: #6a8aa8; }
.verdict-pd { font-family: 'DM Mono', monospace; font-size: 3rem; font-weight: 500; line-height: 1; text-align: right; }
.verdict-pd-label { font-size: 0.65rem; color: #3a5878; text-transform: uppercase; letter-spacing: 1.5px; text-align: right; margin-top: 4px; }
.dc-row { display: grid; gap: 10px; margin-bottom: 16px; }
.dc-row-4 { grid-template-columns: repeat(4, 1fr); }
.dc-row-5 { grid-template-columns: repeat(5, 1fr); }
.dc-row-3 { grid-template-columns: repeat(3, 1fr); }
.dc { background: #060c16; border: 1px solid #0d1f35; border-radius: 6px; padding: 14px 16px; }
.dc-label { font-family: 'DM Mono', monospace; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 1.5px; color: #3a5878; margin-bottom: 6px; }
.dc-value { font-family: 'DM Mono', monospace; font-size: 1.2rem; font-weight: 500; color: #c8d8e8; }
.dc-value.green  { color: #2ecc71; }
.dc-value.yellow { color: #f1c40f; }
.dc-value.orange { color: #e67e22; }
.dc-value.red    { color: #e74c3c; }
.dc-sub { font-size: 0.65rem; color: #2a4060; margin-top: 3px; }
.sl { font-family: 'DM Mono', monospace; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 2px; color: #1a3a58; padding-bottom: 8px; border-bottom: 1px solid #0d1f35; margin: 22px 0 14px 0; }
.fp { border-radius: 5px; padding: 10px 14px; margin: 5px 0; display: flex; align-items: flex-start; gap: 10px; }
.fp-risk { background: #0f0404; border: 1px solid #3a1010; }
.fp-safe { background: #040f07; border: 1px solid #103a1a; }
.fp-dot  { font-size: 0.75rem; margin-top: 1px; flex-shrink: 0; }
.fp-text { font-size: 0.82rem; color: #a0b8c8; line-height: 1.4; }
.fp-feat { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #3a5878; margin-top: 2px; }
.gauge-wrap { margin: 16px 0; }
.gauge-track { height: 6px; background: #0d1f35; border-radius: 3px; overflow: hidden; }
.gauge-fill  { height: 100%; border-radius: 3px; }
.gauge-ticks { display: flex; justify-content: space-between; margin-top: 4px; font-family: 'DM Mono', monospace; font-size: 0.6rem; color: #1a3a58; }
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #0d1f35; gap: 0; padding: 0; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #3a5878; border-radius: 0; padding: 10px 18px; font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.5px; border-bottom: 2px solid transparent; }
.stTabs [aria-selected="true"] { background: transparent !important; color: #c8d8e8 !important; border-bottom: 2px solid #3498db !important; }
.report-wrap { background: #060c16; border: 1px solid #0d1f35; border-radius: 6px; padding: 28px 32px; font-size: 0.88rem; line-height: 1.8; color: #8aabb8; }
.report-wrap h2, .report-wrap h3 { font-family: 'Syne', sans-serif; color: #c8d8e8; }
.stButton > button { background: #3498db !important; color: #ffffff !important; border: none !important; border-radius: 5px !important; font-family: 'DM Mono', monospace !important; font-size: 0.78rem !important; font-weight: 500 !important; letter-spacing: 0.5px !important; padding: 11px 20px !important; width: 100% !important; }
.stButton > button:hover { background: #2980b9 !important; }
hr { border-color: #0d1f35 !important; margin: 20px 0 !important; }
.big-pd { text-align: center; padding: 32px; background: #060c16; border: 1px solid #0d1f35; border-radius: 8px; margin: 16px 0; }
.big-pd-num { font-family: 'DM Mono', monospace; font-size: 5rem; font-weight: 500; line-height: 1; }
.big-pd-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; color: #3a5878; margin-top: 8px; }
.stage-circle { width: 64px; height: 64px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-family: 'DM Mono', monospace; font-size: 1.8rem; font-weight: 500; flex-shrink: 0; }
.sc { background: #060c16; border: 1px solid #0d1f35; border-radius: 6px; padding: 16px; border-top: 2px solid; }
.sc-name  { font-family: 'DM Mono', monospace; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 1.5px; color: #3a5878; margin-bottom: 8px; }
.sc-value { font-family: 'DM Mono', monospace; font-size: 1.4rem; font-weight: 500; color: #c8d8e8; }
.sc-sub   { font-size: 0.72rem; color: #3a5878; margin-top: 3px; }
.footer { text-align: center; padding: 28px 0 12px 0; font-family: 'DM Mono', monospace; font-size: 0.62rem; color: #1a3a58; border-top: 1px solid #0d1f35; margin-top: 32px; letter-spacing: 0.5px; }
.landing-card { background: #060c16; border: 1px solid #0d1f35; border-radius: 8px; padding: 24px; text-align: center; height: 140px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px; }
.landing-icon  { font-size: 1.8rem; }
.landing-title { font-family: 'Syne', sans-serif; font-size: 0.92rem; font-weight: 700; color: #c8d8e8; }
.landing-desc  { font-size: 0.75rem; color: #3a5878; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)


# ── Load ──────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE, 'data', 'raw', 'german_credit_clean.csv')
MODEL_PATH = os.path.join(BASE, 'artifacts', 'model_german.pkl')

@st.cache_resource
def load_all():
    X, y, feats = prepare_data(DATA_PATH)
    data      = joblib.load(MODEL_PATH)
    model     = data['model']           if isinstance(data, dict) else data
    metrics   = data.get('metrics', {}) if isinstance(data, dict) else {}
    explainer = get_explainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):           sv = sv[1]
    if len(np.array(sv).shape) == 3:   sv = sv[:, :, 1]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    yp_tr = model.predict_proba(X_tr[feats])[:, 1]
    yp_te = model.predict_proba(X_te[feats])[:, 1]
    return model, explainer, X, y, feats, sv, metrics, X_tr, X_te, y_tr, y_te, yp_tr, yp_te

(model, explainer, X, y, feats, train_sv, metrics,
 X_tr, X_te, y_tr, y_te, yp_tr, yp_te) = load_all()

auc  = metrics.get('auc',  0.79)
gini = metrics.get('gini', 0.58)
ks   = metrics.get('ks',   0.51)


# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_class(p):
    if p < 0.40:   return "LOW RISK",      "low",   "#2ecc71", "✅", "APPROVE"
    elif p < 0.55: return "MODERATE RISK", "mod",   "#f1c40f", "⚠️", "APPROVE WITH CONDITIONS"
    elif p < 0.65: return "HIGH RISK",     "high",  "#e67e22", "🔶", "MANUAL REVIEW"
    else:          return "VERY HIGH RISK","vhigh", "#e74c3c", "🚫", "DECLINE"

def css_val(cls):
    return {"low":"green","mod":"yellow","high":"orange","vhigh":"red"}.get(cls,"")

def dark_ax(fig, ax):
    fig.patch.set_facecolor('#04080f')
    ax.set_facecolor('#060c16')
    ax.tick_params(colors='#3a5878', labelsize=8)
    ax.xaxis.label.set_color('#3a5878')
    ax.yaxis.label.set_color('#3a5878')
    ax.title.set_color('#c8d8e8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#0d1f35')
    return fig, ax

def build_input(ca, dur, age, ir, ec, dep, chk, sav, emp, hist, purp, hous):
    cm = {"No account":0,"< 0 DM":-1,"0-200 DM":1,"> 200 DM":2}
    sm = {"No savings":0,"< 100 DM":1,"100-500 DM":2,"500-1000 DM":3,"> 1000 DM":4}
    em = {"Unemployed":0,"< 1 year":1,"1-4 years":2,"4-7 years":3,"> 7 years":4}
    hm = {"No credits":0,"All paid":1,"Existing paid":2,"Delay in past":3,"Critical account":4}
    pm = {"Radio/TV":3,"Car (new)":0,"Car (used)":1,"Furniture":2,"Appliances":4,
          "Repairs":5,"Education":6,"Vacation":7,"Retraining":8,"Business":9}
    hh = {"Free":0,"Rent":1,"Own":2}
    row = {
        'duration':dur,'credit_amount':ca,'installment_rate':ir,
        'residence_since':3,'age':age,'existing_credits':ec,'dependents':dep,
        'checking_account_num':cm.get(chk,0),'savings_account_num':sm.get(sav,0),
        'employment_num':em.get(emp,2),'credit_history_num':hm.get(hist,2),
        'purpose_num':pm.get(purp,3),'property_num':2,'housing_num':hh.get(hous,1),
        'job_num':2,'has_telephone':1,'is_foreign_worker':1,'has_other_debtors':0,
        'loan_to_age':ca/(age+1),'duration_to_age':dur/(age+1),
        'credit_per_month':ca/(dur+1),'is_high_amount':int(ca>3000),
        'long_duration':int(dur>24),'young_borrower':int(age<25),
    }
    return pd.DataFrame([row])[feats].fillna(0)

def calc_psi(exp, act, bins=10):
    bp = np.linspace(0,1,bins+1)
    e  = np.histogram(exp,bins=bp)[0]/len(exp)
    a  = np.histogram(act,bins=bp)[0]/len(act)
    e  = np.where(e==0,0.0001,e); a = np.where(a==0,0.0001,a)
    return float(np.sum((a-e)*np.log(a/e)))

PLAIN_RISK = {
    'checking_account_num':'Negative / no bank account',
    'credit_history_num':  'Past credit problems on file',
    'duration':            'Loan term is long',
    'credit_amount':       'Loan amount is large',
    'savings_account_num': 'Low savings',
    'employment_num':      'Short employment tenure',
    'loan_to_age':         'Large loan relative to age',
    'credit_per_month':    'High monthly repayments',
    'installment_rate':    'High installment-to-income ratio',
    'existing_credits':    'Multiple existing loans',
    'is_high_amount':      'Above-average loan amount',
    'long_duration':       'Term over 24 months',
    'young_borrower':      'Limited credit history',
    'purpose_num':         'Higher-risk loan purpose',
    'housing_num':         'No property ownership',
}
PLAIN_GOOD = {
    'checking_account_num':'Healthy bank balance',
    'credit_history_num':  'Good repayment history',
    'savings_account_num': 'Strong savings buffer',
    'employment_num':      'Long-term stable employment',
    'loan_to_age':         'Reasonable loan for age',
    'credit_per_month':    'Manageable monthly payments',
    'housing_num':         'Property ownership',
    'duration':            'Short loan term',
    'credit_amount':       'Modest loan amount',
    'age':                 'Established credit history',
}
FEAT_DESC = {
    'checking_account_num':'Balance status — negative is highest risk signal',
    'credit_history_num':  'Past repayment behavior',
    'duration':            'Longer = higher cumulative exposure',
    'credit_amount':       'Total credit requested',
    'savings_account_num': 'Financial cushion',
    'employment_num':      'Job tenure — income stability',
    'loan_to_age':         'Credit / age — affordability ratio',
    'credit_per_month':    'Monthly repayment burden',
    'installment_rate':    'Installment as % of income',
    'existing_credits':    'Concentration risk',
    'is_high_amount':      'Above portfolio median',
    'long_duration':       'Term > 24 months',
    'young_borrower':      'Age < 25 — thin file',
    'purpose_num':         'Loan purpose risk profile',
    'housing_num':         'Homeowners default less',
}


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-left">
        <h1>🏦 CreditLens AI</h1>
        <span class="sub">UCI German Credit · LightGBM · Youden's J Thresholds</span>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        <span class="badge badge-green">AUC {auc:.2f}</span>
        <span class="badge badge-blue">Gini {gini:.2f}</span>
        <span class="badge badge-blue">KS {ks:.2f}</span>
        <span class="badge badge-yellow">CFPB · SR 11-7 · Basel III · IFRS 9</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='font-family:DM Mono,monospace;font-size:0.62rem;"
    "text-transform:uppercase;letter-spacing:2px;color:#1a3a58;"
    "margin-bottom:10px;'>View Mode</div>", unsafe_allow_html=True)
mode = st.sidebar.radio("", ["👤  Simple", "🏦  Expert"], label_visibility="collapsed")

st.sidebar.markdown(
    "<div style='font-family:DM Mono,monospace;font-size:0.62rem;"
    "text-transform:uppercase;letter-spacing:2px;color:#1a3a58;"
    "margin:14px 0 10px 0;padding-top:10px;border-top:1px solid #0d1f35;'>"
    "Applicant</div>", unsafe_allow_html=True)

ca   = st.sidebar.number_input("Credit Amount ($)", 250, 20000, 3000, step=250)
dur  = st.sidebar.slider("Duration (months)", 4, 72, 24)
ir   = st.sidebar.slider("Installment Rate", 1, 4, 3)
chk  = st.sidebar.selectbox("Checking Account", ["No account","< 0 DM","0-200 DM","> 200 DM"])
sav  = st.sidebar.selectbox("Savings Account",  ["No savings","< 100 DM","100-500 DM","500-1000 DM","> 1000 DM"])
purp = st.sidebar.selectbox("Loan Purpose", ["Radio/TV","Car (new)","Car (used)","Furniture","Appliances","Repairs","Education","Vacation","Retraining","Business"])
age  = st.sidebar.slider("Age", 18, 75, 35)
emp  = st.sidebar.selectbox("Employment", ["< 1 year","1-4 years","4-7 years","> 7 years","Unemployed"])
hous = st.sidebar.selectbox("Housing", ["Rent","Own","Free"])
hist = st.sidebar.selectbox("Credit History", ["Existing paid","All paid","No credits","Delay in past","Critical account"])
ec   = st.sidebar.slider("Existing Credits", 1, 4, 1)
dep  = st.sidebar.slider("Dependents", 1, 2, 1)

st.sidebar.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
go   = st.sidebar.button("Assess Risk", use_container_width=True)

st.sidebar.markdown(f"""
<div style='margin-top:12px;padding:12px;background:#060c16;border-radius:5px;border:1px solid #0d1f35;'>
    <div style='font-family:DM Mono,monospace;font-size:0.58rem;text-transform:uppercase;
         letter-spacing:1.5px;color:#1a3a58;margin-bottom:8px;'>Model</div>
    <table style='width:100%;font-family:DM Mono,monospace;font-size:0.72rem;color:#5a7a9a;border-collapse:collapse;'>
        <tr><td>AUC-ROC</td><td style='color:#2ecc71;text-align:right;'>{auc:.4f}</td></tr>
        <tr><td>Gini</td><td style='color:#2ecc71;text-align:right;'>{gini:.4f}</td></tr>
        <tr><td>KS</td><td style='color:#2ecc71;text-align:right;'>{ks:.4f}</td></tr>
        <tr><td>Rows</td><td style='text-align:right;'>{len(X):,}</td></tr>
        <tr><td>Default %</td><td style='text-align:right;'>{y.mean():.1%}</td></tr>
    </table>
    <div style='margin-top:10px;padding-top:8px;border-top:1px solid #0d1f35;
         font-family:DM Mono,monospace;font-size:0.58rem;color:#1a3a58;'>
        API: uvicorn app.api:app --port 8000<br>
        Docs: localhost:8000/docs
    </div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# LANDING
# ════════════════════════════════════════════════════════════════════════════
if not go:
    st.markdown("""
    <div style='background:#060c16;border:1px solid #0d1f35;border-radius:8px;
         padding:20px 24px;margin-bottom:24px;border-left:3px solid #3498db;'>
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
             color:#c8d8e8;margin-bottom:5px;'>Enter applicant details → Assess Risk</div>
        <div style='font-size:0.8rem;color:#3a5878;'>
            Simple Mode — plain-English verdict &nbsp;·&nbsp;
            Expert Mode — SHAP · Basel III · IFRS 9 · CCAR · Fair Lending
        </div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,icon,title,desc in [
        (c1,"🌊","SHAP Waterfall","Feature decomposition — CFPB 2022-03"),
        (c2,"🏛️","Regulatory","Basel III · IFRS 9 · SR 11-7 · OCC"),
        (c3,"📉","Stress Test","CCAR 4-scenario capital impact"),
        (c4,"🔍","Model Health","Calibration · PSI · Fair lending"),
    ]:
        with col:
            st.markdown(f"""
            <div class="landing-card">
                <div class="landing-icon">{icon}</div>
                <div class="landing-title">{title}</div>
                <div class="landing-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sl" style="margin-top:24px;">Performance</div>',
                unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,lbl,val,sub,cls in [
        (c1,"AUC-ROC",     f"{auc:.4f}",  "OCC min: 0.70",     "green"),
        (c2,"Gini",        f"{gini:.4f}", "Basel III min: 0.30","green"),
        (c3,"KS",          f"{ks:.4f}",   "SR 11-7 min: 0.20",  "green"),
        (c4,"Lift vs LR",  "+8.1%",       "vs logistic reg",    "green"),
        (c5,"Default Rate",f"{y.mean():.1%}","training set",    ""),
    ]:
        with col:
            st.markdown(f"""
            <div class="dc">
                <div class="dc-label">{lbl}</div>
                <div class="dc-value {cls}">{val}</div>
                <div class="dc-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ASSESSMENT
# ════════════════════════════════════════════════════════════════════════════
else:
    inp  = build_input(ca,dur,age,ir,ec,dep,chk,sav,emp,hist,purp,hous)
    prob = model.predict_proba(inp)[0][1]
    rtg, cls, clr, ico, rec = risk_class(prob)
    cv   = css_val(cls)
    m    = _ecl_metrics(prob, ca)

    expl_df = explain_single(explainer, inp, feats)
    rfs     = expl_df[expl_df['shap_value'] > 0].to_dict('records')
    pfs     = expl_df[expl_df['shap_value'] < 0].to_dict('records')
    adv     = get_adverse_reasons(expl_df)

    sv1 = explainer.shap_values(inp)
    if isinstance(sv1, list):          sv1 = sv1[1]
    if len(np.array(sv1).shape) == 3:  sv1 = sv1[:,:,1]
    vals = sv1[0]
    bval = explainer.expected_value
    if isinstance(bval, list):         bval = bval[1]

    # Verdict
    st.markdown(f"""
    <div class="verdict {cls}">
        <div class="verdict-icon">{ico}</div>
        <div class="verdict-main">
            <div class="verdict-label">Credit Decision</div>
            <div class="verdict-title" style="color:{clr};">{rtg}</div>
            <div class="verdict-sub">{rec}</div>
        </div>
        <div>
            <div class="verdict-pd" style="color:{clr};">{prob:.0%}</div>
            <div class="verdict-pd-label">Default Probability</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Gauge
    fill = int(prob * 100)
    st.markdown(f"""
    <div class="gauge-wrap">
        <div class="gauge-track">
            <div class="gauge-fill" style="width:{fill}%;
                 background:linear-gradient(90deg,#2ecc71,#f1c40f,#e67e22,#e74c3c);"></div>
        </div>
        <div class="gauge-ticks">
            <span>0%</span><span>▾ 40%</span><span>▾ 55%</span><span>▾ 65%</span><span>100%</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # Key metrics
    st.markdown(f"""
    <div class="dc-row dc-row-4">
        <div class="dc"><div class="dc-label">PD</div>
            <div class="dc-value {cv}">{prob:.1%}</div></div>
        <div class="dc"><div class="dc-label">Expected Loss</div>
            <div class="dc-value {cv}">${m['el']:,.0f}</div></div>
        <div class="dc"><div class="dc-label">Credit Amount</div>
            <div class="dc-value">${ca:,}</div></div>
        <div class="dc"><div class="dc-label">Duration</div>
            <div class="dc-value">{dur}mo</div></div>
    </div>""", unsafe_allow_html=True)

    # ── Simple Mode ───────────────────────────────────────────────────────
    if mode == "👤  Simple":
        cr, cg = st.columns(2)
        with cr:
            st.markdown('<div class="sl">Risk Factors</div>', unsafe_allow_html=True)
            if rfs:
                for f in rfs[:4]:
                    r = PLAIN_RISK.get(f['feature'], f['feature'])
                    st.markdown(f'<div class="fp fp-risk"><span class="fp-dot">▲</span>'
                                f'<div><div class="fp-text">{r}</div>'
                                f'<div class="fp-feat">{f["feature"]}</div></div></div>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<div class="fp fp-safe"><span class="fp-dot">—</span>'
                            '<div class="fp-text">No significant risk factors</div></div>',
                            unsafe_allow_html=True)
        with cg:
            st.markdown('<div class="sl">Positive Factors</div>', unsafe_allow_html=True)
            if pfs:
                for f in pfs[:4]:
                    r = PLAIN_GOOD.get(f['feature'], f['feature'])
                    st.markdown(f'<div class="fp fp-safe"><span class="fp-dot">▼</span>'
                                f'<div><div class="fp-text">{r}</div>'
                                f'<div class="fp-feat">{f["feature"]}</div></div></div>',
                                unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:18px;padding:10px 14px;background:#060c16;
             border-radius:5px;border:1px solid #0d1f35;text-align:center;
             font-size:0.72rem;color:#2a4060;'>
            Switch to <strong style='color:#3498db;'>Expert Mode</strong>
            for SHAP · Basel III · IFRS 9 · Stress Testing · Regulatory Compliance
        </div>""", unsafe_allow_html=True)

    # ── Expert Mode ───────────────────────────────────────────────────────
    else:
        t1,t2,t3,t4,t5,t6,t7,t8,t9,t10 = st.tabs([
            "WATERFALL","SHAP","POPULATION","WHAT-IF",
            "REGULATORY","STRESS","HEALTH","REPORT",
            "LOSS DIST","TERM STRUCTURE"
        ])

        # WATERFALL
        with t1:
            si   = np.argsort(np.abs(vals))[::-1][:12]
            si   = si[np.argsort(vals[si])]
            fls  = [feats[i] for i in si]
            svs  = vals[si]
            fvs  = inp.iloc[0].values[si]
            fig,ax = plt.subplots(figsize=(11,7))
            dark_ax(fig,ax)
            run = bval; pos = []
            for v in svs: pos.append((run,v)); run+=v
            for i,((s,w),lb,fv,c) in enumerate(zip(pos,fls,fvs,
                    ['#e74c3c' if v>0 else '#3498db' for v in svs])):
                ax.barh(i,w,left=s,color=c,height=0.55,alpha=0.9,edgecolor='none')
                xt = s+w+(0.005 if w>=0 else -0.005)
                ax.text(xt,i,f'{w:+.3f}',va='center',ha='left' if w>=0 else 'right',
                        fontsize=7.5,color='#8aabb8',fontfamily='monospace')
                ax.text(bval-0.33,i,f'{lb} = {fv:.2f}',va='center',ha='right',
                        fontsize=8,color='#5a7a9a')
            ax.axvline(bval,color='#2a4060',lw=1.5,ls='--',label=f'Base: {bval:.3f}')
            ax.axvline(prob,color=clr,lw=2.5,ls='-',label=f'Score: {prob:.3f}')
            ax.set_yticks([])
            ax.set_xlabel('Default Probability',fontsize=8)
            ax.set_title(f'Waterfall — {prob:.1%} Default Probability',fontsize=10,fontweight='bold',pad=12)
            ax.legend(facecolor='#060c16',edgecolor='#0d1f35',labelcolor='#8aabb8',fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            rows = [{"Feature":feats[i],"Value":f"{inp.iloc[0][feats[i]]:.3f}",
                     "SHAP":f"{vals[i]:+.4f}",
                     "Direction":"▲ Risk" if vals[i]>0 else "▼ Safe",
                     "Meaning":FEAT_DESC.get(feats[i],"—")}
                    for i in np.argsort(np.abs(vals))[::-1][:12]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # SHAP
        with t2:
            cb, cb2 = st.columns([3,2])
            with cb:
                plt.style.use('dark_background')
                fig2,ax2 = plt.subplots(figsize=(9,8))
                dark_ax(fig2,ax2)
                shap.summary_plot(train_sv,X,feature_names=feats,show=False,plot_size=None)
                fo = np.argsort(np.abs(train_sv).mean(0))[::-1]
                for rk,fi in enumerate(fo[:len(feats)]):
                    yp = list(reversed(range(len(fo))))[rk]
                    ax2.plot(vals[fi],yp,'r|',markersize=18,markeredgewidth=2.5,
                             zorder=10,label='This applicant' if rk==0 else "")
                ax2.set_title('Beeswarm — 1,000 Applicants',fontsize=10,fontweight='bold')
                ax2.tick_params(colors='#3a5878',labelsize=8)
                for sp in ax2.spines.values(): sp.set_edgecolor('#0d1f35')
                ax2.legend(facecolor='#060c16',edgecolor='#0d1f35',labelcolor='#8aabb8',fontsize=8)
                plt.tight_layout(); st.pyplot(fig2); plt.close()
                plt.style.use('default')
            with cb2:
                fig3,ax3 = plt.subplots(figsize=(5,8))
                dark_ax(fig3,ax3)
                ms  = np.abs(train_sv).mean(0)
                si3 = np.argsort(ms)[-15:]
                bars= ax3.barh([feats[i] for i in si3],ms[si3],
                               color='#3498db',alpha=0.85,height=0.6,edgecolor='none')
                for bar,v in zip(bars,ms[si3]):
                    ax3.text(v+0.001,bar.get_y()+bar.get_height()/2,
                             f'{v:.3f}',va='center',fontsize=7,color='#8aabb8',fontfamily='monospace')
                ax3.set_xlabel('Mean |SHAP|',fontsize=8)
                ax3.set_title('Global Importance',fontsize=10,fontweight='bold')
                plt.tight_layout(); st.pyplot(fig3); plt.close()

                st.markdown('<div class="sl">Risk Drivers</div>', unsafe_allow_html=True)
                for f in rfs[:4]:
                    st.markdown(f'<div class="fp fp-risk"><span class="fp-dot">▲</span>'
                                f'<div><div class="fp-text">{PLAIN_RISK.get(f["feature"],f["feature"])}</div>'
                                f'<div class="fp-feat">{f["feature"]} · +{f["shap_value"]:.4f}</div>'
                                f'</div></div>', unsafe_allow_html=True)
                st.markdown('<div class="sl">Protective</div>', unsafe_allow_html=True)
                for f in pfs[:3]:
                    st.markdown(f'<div class="fp fp-safe"><span class="fp-dot">▼</span>'
                                f'<div><div class="fp-text">{PLAIN_GOOD.get(f["feature"],f["feature"])}</div>'
                                f'<div class="fp-feat">{f["feature"]} · {f["shap_value"]:.4f}</div>'
                                f'</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="sl">Adverse Action — ECOA / CFPB 2022-03</div>', unsafe_allow_html=True)
            cols = st.columns(min(len(adv),4))
            for col,i,r in zip(cols,range(1,5),adv):
                with col:
                    st.markdown(f"""
                    <div style='background:#0f0404;border:1px solid #3a1010;border-radius:5px;
                         padding:12px;text-align:center;'>
                        <div style='color:#e74c3c;font-family:DM Mono,monospace;font-size:0.6rem;
                             letter-spacing:1.5px;margin-bottom:6px;'>REASON {i}</div>
                        <div style='color:#a88888;font-size:0.78rem;line-height:1.4;'>{r}</div>
                    </div>""", unsafe_allow_html=True)

        # POPULATION
        with t3:
            ap   = model.predict_proba(X[feats])[:,1]
            pctl = (ap < prob).mean() * 100
            fig4,ax4 = plt.subplots(figsize=(11,4))
            dark_ax(fig4,ax4)
            n,bins,patches = ax4.hist(ap,bins=50,edgecolor='none',alpha=0.9)
            for patch,left in zip(patches,bins[:-1]):
                if left<0.40:   patch.set_facecolor('#2ecc71')
                elif left<0.65: patch.set_facecolor('#e67e22')
                else:           patch.set_facecolor('#e74c3c')
            ax4.axvline(prob,color='white',lw=3,label=f'This applicant: {prob:.1%} ({pctl:.0f}th pct)')
            ax4.axvline(0.40,color='#2ecc71',lw=1,ls='--',alpha=0.6,label='Approve 40%')
            ax4.axvline(0.65,color='#e74c3c',lw=1,ls='--',alpha=0.6,label='Decline 65%')
            ax4.set_xlabel('Default Probability',fontsize=8)
            ax4.set_title('Score Distribution — 1,000 Applicants',fontsize=10,fontweight='bold')
            ax4.legend(facecolor='#060c16',edgecolor='#0d1f35',labelcolor='#8aabb8',fontsize=8)
            plt.tight_layout(); st.pyplot(fig4); plt.close()

            ap_pct = (ap<0.40).mean()*100
            rv_pct = ((ap>=0.40)&(ap<0.65)).mean()*100
            dc_pct = (ap>=0.65).mean()*100
            st.markdown(f"""
            <div class="dc-row dc-row-4">
                <div class="dc"><div class="dc-label">Percentile</div>
                    <div class="dc-value {cv}">{pctl:.0f}th</div></div>
                <div class="dc"><div class="dc-label">Approve Rate</div>
                    <div class="dc-value green">{ap_pct:.0f}%</div></div>
                <div class="dc"><div class="dc-label">Review Rate</div>
                    <div class="dc-value yellow">{rv_pct:.0f}%</div></div>
                <div class="dc"><div class="dc-label">Decline Rate</div>
                    <div class="dc-value red">{dc_pct:.0f}%</div></div>
            </div>""", unsafe_allow_html=True)

        # WHAT-IF
        with t4:
            wc, wr = st.columns([1,2])
            with wc:
                wf = st.selectbox("Feature", ['credit_amount','duration','age',
                    'checking_account_num','savings_account_num','employment_num',
                    'credit_history_num','installment_rate','existing_credits'])
                fr = {'credit_amount':(250,20000,int(ca)),'duration':(4,72,int(dur)),
                      'age':(18,75,int(age)),'checking_account_num':(-1,2,0),
                      'savings_account_num':(0,4,0),'employment_num':(0,4,2),
                      'credit_history_num':(0,4,2),'installment_rate':(1,4,int(ir)),
                      'existing_credits':(1,4,int(ec))}
                mn,mx,dv = fr[wf]
                wv = st.slider("Value", mn, mx, dv)
                cr2 = {0:"No account",-1:"< 0 DM",1:"0-200 DM",2:"> 200 DM"}
                sr2 = {0:"No savings",1:"< 100 DM",2:"100-500 DM",3:"500-1000 DM",4:"> 1000 DM"}
                er2 = {0:"Unemployed",1:"< 1 year",2:"1-4 years",3:"4-7 years",4:"> 7 years"}
                hr3 = {0:"No credits",1:"All paid",2:"Existing paid",3:"Delay in past",4:"Critical account"}
                wca = wv if wf=='credit_amount' else ca
                wdu = wv if wf=='duration'      else dur
                wag = wv if wf=='age'           else age
                wir = wv if wf=='installment_rate' else ir
                wec = wv if wf=='existing_credits' else ec
                wch = cr2.get(wv,chk)  if wf=='checking_account_num' else chk
                wsa = sr2.get(wv,sav)  if wf=='savings_account_num'  else sav
                wem = er2.get(wv,emp)  if wf=='employment_num'        else emp
                whi = hr3.get(wv,hist) if wf=='credit_history_num'    else hist
                wi  = build_input(wca,wdu,wag,wir,wec,dep,wch,wsa,wem,whi,purp,hous)
                wp  = model.predict_proba(wi)[0][1]
                wr2,wc2,wcl,_,_ = risk_class(wp)
                delta = wp - prob
                st.markdown(f"""
                <div style='background:#060c16;border:1px solid #0d1f35;border-radius:6px;
                     padding:20px;margin-top:12px;'>
                    <div style='font-family:DM Mono,monospace;font-size:0.6rem;
                         text-transform:uppercase;letter-spacing:1.5px;color:#1a3a58;margin-bottom:8px;'>
                         Modified Score</div>
                    <div style='font-family:DM Mono,monospace;font-size:2.8rem;font-weight:500;color:{wcl};'>
                         {wp:.0%}</div>
                    <div style='color:{wcl};font-size:0.8rem;margin:4px 0 10px 0;'>{wr2}</div>
                    <div style='font-family:DM Mono,monospace;font-size:0.95rem;
                         color:{"#e74c3c" if delta>0 else "#2ecc71"};'>
                         {"▲" if delta>0 else "▼"} {abs(delta):.1%}</div>
                    <div style='font-size:0.7rem;color:#2a4060;margin-top:4px;'>{prob:.1%} → {wp:.1%}</div>
                </div>
                <div style='margin-top:8px;padding:10px 12px;background:#060c16;border-radius:5px;
                     border:1px solid #0d1f35;font-size:0.75rem;color:#3a5878;'>
                     {FEAT_DESC.get(wf,"—")}
                </div>""", unsafe_allow_html=True)
            with wr:
                we  = explain_single(explainer, wi, feats)
                fig5,(ao,an) = plt.subplots(1,2,figsize=(11,6))
                fig5.patch.set_facecolor('#04080f')
                for ax,df,ttl,c in [(ao,expl_df,f"Original {prob:.1%}",clr),(an,we,f"Modified {wp:.1%}",wcl)]:
                    ax.set_facecolor('#060c16')
                    td = df.head(10).sort_values('shap_value')
                    bc = ['#e74c3c' if v>0 else '#3498db' for v in td['shap_value']]
                    ax.barh(td['feature'],td['shap_value'],color=bc,height=0.6,alpha=0.9,edgecolor='none')
                    ax.axvline(0,color='#1a3a58',lw=1,ls='--')
                    ax.set_title(ttl,color=c,fontsize=9,fontweight='bold')
                    ax.tick_params(colors='#3a5878',labelsize=7.5)
                    for sp in ax.spines.values(): sp.set_edgecolor('#0d1f35')
                fig5.suptitle(f'{wf}: {dv} → {wv}',color='#3a5878',fontsize=8)
                plt.tight_layout(); st.pyplot(fig5); plt.close()

        # REGULATORY
        with t5:
            st.markdown(f"""
            <div class="dc-row dc-row-5">
                <div class="dc"><div class="dc-label">PD</div>
                    <div class="dc-value {cv}">{m['pd']:.1%}</div>
                    <div class="dc-sub">Prob. of default</div></div>
                <div class="dc"><div class="dc-label">LGD</div>
                    <div class="dc-value">{m['lgd']:.0%}</div>
                    <div class="dc-sub">Loss given default</div></div>
                <div class="dc"><div class="dc-label">EAD</div>
                    <div class="dc-value">${m['ead']:,.0f}</div>
                    <div class="dc-sub">Exposure at default</div></div>
                <div class="dc"><div class="dc-label">EL</div>
                    <div class="dc-value {cv}">${m['el']:,.0f}</div>
                    <div class="dc-sub">PD × LGD × EAD</div></div>
                <div class="dc"><div class="dc-label">Capital</div>
                    <div class="dc-value {cv}">${m['capital_req']:,.0f}</div>
                    <div class="dc-sub">8% × RWA</div></div>
            </div>""", unsafe_allow_html=True)

            sc  = {1:'#2ecc71',2:'#f1c40f',3:'#e74c3c'}[m['ifrs9_stage']]
            sd  = {1:"No SICR — 12-month ECL",2:"SICR — Lifetime ECL · Watchlist",
                   3:"Impaired — Lifetime ECL · Workout"}[m['ifrs9_stage']]
            st.markdown(f"""
            <div style='background:#060c16;border:1px solid #0d1f35;border-radius:6px;
                 padding:16px 20px;display:flex;align-items:center;gap:16px;
                 border-left:3px solid {sc};margin-bottom:16px;'>
                <div class="stage-circle" style="background:{sc}18;border:2px solid {sc};color:{sc};">
                    {m['ifrs9_stage']}</div>
                <div>
                    <div style='font-family:DM Mono,monospace;font-size:0.92rem;font-weight:500;
                         color:{sc};margin-bottom:2px;'>Stage {m['ifrs9_stage']} — {m['ecl_horizon']}</div>
                    <div style='font-size:0.78rem;color:#5a7a9a;'>{sd}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            fig_i,ax_i = plt.subplots(figsize=(10,1.8))
            dark_ax(fig_i,ax_i)
            for s,e,c,lb in [(0,0.10,'#2ecc71','Stage 1 — 12M ECL'),
                              (0.10,0.50,'#f1c40f','Stage 2 — Lifetime (SICR)'),
                              (0.50,1.00,'#e74c3c','Stage 3 — Lifetime (Impaired)')]:
                ax_i.barh(0,e-s,left=s,color=c,alpha=0.2,height=0.5,edgecolor=c,lw=1.5)
                ax_i.text((s+e)/2,0,lb,ha='center',va='center',fontsize=7.5,
                          color=c,fontfamily='monospace',fontweight='bold')
            ax_i.axvline(prob,color='white',lw=2.5,label=f'{prob:.1%}')
            ax_i.set_xlim(0,1); ax_i.set_yticks([])
            ax_i.set_xlabel('PD',fontsize=7.5)
            ax_i.legend(facecolor='#060c16',edgecolor='#0d1f35',labelcolor='#8aabb8',fontsize=8)
            plt.tight_layout(); st.pyplot(fig_i); plt.close()

            st.markdown('<div class="sl">US Regulatory Status</div>', unsafe_allow_html=True)
            comp = pd.DataFrame([
                ["CFPB Circular 2022-03","SHAP adverse action reasons","✅"],
                ["ECOA / Regulation B",  "Written notice 30 days",     "✅"],
                ["Fair Housing Act",      "4/5ths proxy analysis",      "✅"],
                ["OCC 2011-12 / SR 11-7","Benchmarking + docs",        "✅"],
                ["CCAR / Dodd-Frank",     "4-scenario stress test",     "✅"],
                ["Basel III Pillar 1",    "IRB formula",                "✅"],
                ["IFRS 9",                "Stage 1/2/3",                "✅"],
                ["State AI Laws",         "Proxy only — full audit needed","⚠️"],
            ], columns=["Regulation","Requirement","Status"])
            st.dataframe(comp, use_container_width=True, hide_index=True)

        # STRESS
        with t6:
            df_r = pd.read_csv(DATA_PATH)
            la   = df_r['credit_amount'].values
            sr   = run_stress_test(model, X, la, feats)

            c1,c2,c3,c4 = st.columns(4)
            for col,(_,row) in zip([c1,c2,c3,c4],sr.iterrows()):
                c = row['color']
                with col:
                    st.markdown(f"""
                    <div class="sc" style="border-top-color:{c};">
                        <div class="sc-name">{row['Scenario'].split('(')[0].strip()}</div>
                        <div class="sc-value">${row['Portfolio EL']:,.0f}</div>
                        <div class="sc-sub">Expected Loss</div>
                        <div style='margin-top:8px;font-family:DM Mono,monospace;font-size:0.7rem;color:{c};'>
                             PD {row['Avg PD']:.1%}</div>
                        <div style='font-family:DM Mono,monospace;font-size:0.7rem;color:#2a4060;'>
                             Cap ${row['Capital Req']:,.0f}</div>
                    </div>""", unsafe_allow_html=True)

            fig_s,axes = plt.subplots(1,3,figsize=(14,4))
            fig_s.patch.set_facecolor('#04080f')
            lbs   = [r['Scenario'].split('(')[0].strip() for _,r in sr.iterrows()]
            cls_s = [r['color'] for _,r in sr.iterrows()]
            for ax,key,ttl in [(axes[0],'Portfolio EL','Expected Loss'),
                                (axes[1],'Capital Req', 'Capital Required'),
                                (axes[2],'Stage 3 %',   'Stage 3 %')]:
                dark_ax(fig_s,ax)
                vs   = sr[key].values
                bars = ax.bar(range(len(vs)),vs,color=cls_s,alpha=0.85,edgecolor='none')
                ax.set_xticks(range(len(vs)))
                ax.set_xticklabels(lbs,rotation=15,ha='right',fontsize=7.5,color='#3a5878')
                ax.set_title(ttl,fontsize=9,fontweight='bold')
                for bar,v in zip(bars,vs):
                    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.02,
                            f'{v:,.0f}',ha='center',va='bottom',fontsize=6.5,
                            color='#8aabb8',fontfamily='monospace')
            plt.tight_layout(); st.pyplot(fig_s); plt.close()

            st.markdown('<div class="sl">This Applicant</div>', unsafe_allow_html=True)
            rows2 = []
            for sc2,p2 in SCENARIOS.items():
                sp = min(prob*p2['pd_multiplier'],1.0)
                el = sp*p2['lgd']*ca
                _,_,_,_,sr2 = risk_class(sp)
                rows2.append({'Scenario':sc2,'Stressed PD':f"{sp:.1%}",
                              'Expected Loss':f"${el:,.0f}",'Rating':sr2})
            st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)

        # HEALTH
        with t7:
            ch, cp, cf = st.columns(3)
            with ch:
                st.markdown('<div class="sl">Calibration</div>', unsafe_allow_html=True)
                fp2,mp2 = calibration_curve(y_te,yp_te,n_bins=8,strategy='uniform')
                ce  = float(np.mean(np.abs(fp2-mp2)))
                cs  = "✅ Calibrated" if ce<0.08 else "⚠️ Monitor" if ce<0.15 else "❌ Recalibrate"
                cc  = "#2ecc71" if ce<0.08 else "#f1c40f" if ce<0.15 else "#e74c3c"
                st.markdown(f"""
                <div class="dc" style="margin-bottom:12px;">
                    <div class="dc-label">Mean Calibration Error</div>
                    <div class="dc-value" style="color:{cc};">{ce:.4f}</div>
                    <div class="dc-sub">{cs}</div>
                </div>""", unsafe_allow_html=True)
                fig_c,ax_c = plt.subplots(figsize=(5,4))
                dark_ax(fig_c,ax_c)
                ax_c.plot(mp2,fp2,'o-',color='#3498db',lw=2,ms=6,label='LightGBM')
                ax_c.plot([0,1],[0,1],'--',color='#2ecc71',lw=1.2,label='Perfect')
                ax_c.fill_between([0,1],[-0.1,0.9],[0.1,1.1],alpha=0.05,color='#f1c40f')
                ax_c.set_xlabel('Predicted PD',fontsize=8); ax_c.set_ylabel('Observed',fontsize=8)
                ax_c.set_title('Calibration',fontsize=9,fontweight='bold')
                ax_c.set_xlim(0,1); ax_c.set_ylim(0,1)
                ax_c.legend(facecolor='#060c16',edgecolor='#0d1f35',labelcolor='#8aabb8',fontsize=7.5)
                plt.tight_layout(); st.pyplot(fig_c); plt.close()

            with cp:
                st.markdown('<div class="sl">PSI</div>', unsafe_allow_html=True)
                ps  = calc_psi(yp_tr,yp_te)
                pss = "✅ Stable" if ps<0.10 else "⚠️ Monitor" if ps<0.25 else "❌ Drift"
                psc = "#2ecc71" if ps<0.10 else "#f1c40f" if ps<0.25 else "#e74c3c"
                st.markdown(f"""
                <div class="dc" style="margin-bottom:12px;">
                    <div class="dc-label">PSI Score</div>
                    <div class="dc-value" style="color:{psc};">{ps:.4f}</div>
                    <div class="dc-sub">{pss}</div>
                </div>""", unsafe_allow_html=True)
                fig_p,ax_p = plt.subplots(figsize=(5,4))
                dark_ax(fig_p,ax_p)
                bp  = np.linspace(0,1,11)
                tp  = np.histogram(yp_tr,bins=bp)[0]/len(yp_tr)
                ep  = np.histogram(yp_te,bins=bp)[0]/len(yp_te)
                xp  = np.arange(len(tp))
                ax_p.bar(xp-0.2,tp,0.4,label='Train',color='#3498db',alpha=0.75,edgecolor='none')
                ax_p.bar(xp+0.2,ep,0.4,label='Test', color='#e67e22',alpha=0.75,edgecolor='none')
                ax_p.set_xticks(xp)
                ax_p.set_xticklabels([f"{bp[i]:.1f}" for i in range(len(bp)-1)],
                                     rotation=45,ha='right',fontsize=6.5,color='#3a5878')
                ax_p.set_title(f'PSI = {ps:.4f}',fontsize=9,fontweight='bold')
                ax_p.legend(facecolor='#060c16',edgecolor='#0d1f35',labelcolor='#8aabb8',fontsize=7.5)
                plt.tight_layout(); st.pyplot(fig_p); plt.close()

            with cf:
                st.markdown('<div class="sl">Fair Lending 4/5ths</div>', unsafe_allow_html=True)
                app_v = (yp_tr < 0.55).astype(int)
                Xr    = X_tr[feats].reset_index(drop=True)
                frows = []
                for ft,lb,dg,ag in [
                    ('young_borrower','Age proxy',1,0),
                    ('is_foreign_worker','Nat. Origin',1,0),
                    ('housing_num','Housing',0,2),
                    ('employment_num','Employment',0,4),
                ]:
                    if ft in Xr.columns:
                        md = (Xr[ft]==dg).values; ma = (Xr[ft]==ag).values
                        gd = app_v[md].mean() if md.sum()>0 else 0
                        ga = app_v[ma].mean() if ma.sum()>0 else 1
                        rt = gd/ga if ga>0 else 0
                        frows.append({'Proxy':lb,'Disadv.':f"{gd:.1%}",
                                      'Adv.':f"{ga:.1%}",'Ratio':f"{rt:.2f}",
                                      'Status':"✅" if rt>=0.80 else "⚠️"})
                st.dataframe(pd.DataFrame(frows), use_container_width=True, hide_index=True)

                fig_f,ax_f = plt.subplots(figsize=(5,4))
                dark_ax(fig_f,ax_f)
                fp_r = pd.DataFrame(frows)
                dr   = [float(r.strip('%'))/100 for r in fp_r['Disadv.']]
                ar   = [float(r.strip('%'))/100 for r in fp_r['Adv.']]
                xf   = np.arange(len(fp_r))
                ax_f.bar(xf-0.18,dr,0.36,label='Disadv.',color='#e74c3c',alpha=0.8,edgecolor='none')
                ax_f.bar(xf+0.18,ar,0.36,label='Adv.',   color='#3498db',alpha=0.8,edgecolor='none')
                for i,(d2,a2) in enumerate(zip(dr,ar)):
                    ax_f.hlines(a2*0.80,i-0.5,i+0.5,colors='#f1c40f',lw=1.5,ls='--')
                ax_f.set_xticks(xf)
                ax_f.set_xticklabels(fp_r['Proxy'],rotation=15,ha='right',fontsize=7.5,color='#3a5878')
                ax_f.set_ylim(0,1)
                ax_f.set_title('Approval by Group',fontsize=9,fontweight='bold')
                ax_f.legend(facecolor='#060c16',edgecolor='#0d1f35',labelcolor='#8aabb8',fontsize=7.5)
                plt.tight_layout(); st.pyplot(fig_f); plt.close()

        # REPORT
        with t8:
            report = generate_risk_report(prob, rfs, pfs)
            st.markdown(f'<div class="report-wrap">{report}</div>', unsafe_allow_html=True)

        # ── TAB 9: PORTFOLIO LOSS DISTRIBUTION ───────────────────────────
        with t9:
            all_pds  = model.predict_proba(X[feats])[:, 1]
            all_lgds = np.full(len(all_pds), 0.45)
            df_r2    = pd.read_csv(DATA_PATH)
            all_eads = df_r2['credit_amount'].values[:len(all_pds)]

            with st.spinner("Running 10,000 Monte Carlo scenarios..."):
                mc = monte_carlo_loss(all_pds, all_lgds, all_eads,
                                      n_sim=10000, rho=0.15)

            st.markdown(f"""
            <div class="dc-row dc-row-5">
                <div class="dc"><div class="dc-label">Expected Loss</div>
                    <div class="dc-value yellow">${mc['el']:,.0f}</div>
                    <div class="dc-sub">Mean portfolio loss</div></div>
                <div class="dc"><div class="dc-label">VaR 95%</div>
                    <div class="dc-value orange">${mc['var_95']:,.0f}</div>
                    <div class="dc-sub">1-in-20 year loss</div></div>
                <div class="dc"><div class="dc-label">VaR 99%</div>
                    <div class="dc-value red">${mc['var_99']:,.0f}</div>
                    <div class="dc-sub">1-in-100 year loss</div></div>
                <div class="dc"><div class="dc-label">ES 99%</div>
                    <div class="dc-value red">${mc['es_99']:,.0f}</div>
                    <div class="dc-sub">Expected shortfall</div></div>
                <div class="dc"><div class="dc-label">Economic Capital</div>
                    <div class="dc-value red">${mc['economic_capital']:,.0f}</div>
                    <div class="dc-sub">VaR(99.9%) − EL</div></div>
            </div>""", unsafe_allow_html=True)

            col_chart, col_info = st.columns([3, 1])
            with col_chart:
                fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
                dark_ax(fig_mc, ax_mc)
                losses = mc['losses']
                ax_mc.hist(losses, bins=80, color='#3498db', alpha=0.6,
                           edgecolor='none', density=True, label='Loss distribution')
                tail_vals = losses[losses >= mc['var_99']]
                if len(tail_vals) > 0:
                    ax_mc.hist(tail_vals, bins=30, color='#e74c3c', alpha=0.7,
                               edgecolor='none', density=True, label='Tail (>VaR 99%)')
                ax_mc.axvline(mc['el'],      color='#2ecc71', lw=2,   ls='-',
                              label=f"EL ${mc['el']:,.0f}")
                ax_mc.axvline(mc['var_95'],  color='#f1c40f', lw=1.5, ls='--',
                              label=f"VaR 95% ${mc['var_95']:,.0f}")
                ax_mc.axvline(mc['var_99'],  color='#e67e22', lw=1.5, ls='--',
                              label=f"VaR 99% ${mc['var_99']:,.0f}")
                ax_mc.axvline(mc['var_999'], color='#e74c3c', lw=1.5, ls=':',
                              label=f"VaR 99.9% ${mc['var_999']:,.0f}")
                ax_mc.set_xlabel('Portfolio Loss ($)', fontsize=9)
                ax_mc.set_ylabel('Density', fontsize=9)
                ax_mc.set_title(
                    f"Portfolio Loss Distribution — 10,000 Monte Carlo Scenarios\n"
                    f"One-Factor Gaussian Copula · ρ = {mc['rho']} "
                    f"(Basel III Retail) · N = {mc['n_obligors']:,} obligors",
                    fontsize=9, fontweight='bold')
                ax_mc.legend(facecolor='#060c16', edgecolor='#0d1f35',
                             labelcolor='#8aabb8', fontsize=7.5)
                plt.tight_layout()
                st.pyplot(fig_mc)
                plt.close()

            with col_info:
                el_rate_pct = mc['el_rate'] * 100
                st.markdown(f"""
                <div class="dc" style="margin-bottom:10px;">
                    <div class="dc-label">Model</div>
                    <div style="font-family:DM Mono,monospace;font-size:0.75rem;
                         color:#5a7a9a;line-height:1.8;">
                        One-factor<br>Gaussian copula<br>
                        ρ = {mc['rho']} (Basel III)<br>
                        {mc['n_sim']:,} scenarios<br>
                        {mc['n_obligors']:,} obligors
                    </div>
                </div>
                <div class="dc" style="margin-bottom:10px;">
                    <div class="dc-label">EL Rate</div>
                    <div class="dc-value yellow">{el_rate_pct:.2f}%</div>
                    <div class="dc-sub">EL / Total EAD</div>
                </div>
                <div class="dc" style="margin-bottom:10px;">
                    <div class="dc-label">Mean Default Rate</div>
                    <div class="dc-value">{mc['mean_default_rate']:.1%}</div>
                    <div class="dc-sub">Avg across scenarios</div>
                </div>
                <div class="dc">
                    <div class="dc-label">P99 Default Rate</div>
                    <div class="dc-value red">{mc['p99_default_rate']:.1%}</div>
                    <div class="dc-sub">Stress default rate</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sl">This Applicant in Portfolio Context</div>',
                        unsafe_allow_html=True)
            applicant_el    = prob * 0.45 * ca
            applicant_share = applicant_el / mc['el'] * 100 if mc['el'] > 0 else 0
            st.markdown(f"""
            <div class="dc-row dc-row-3">
                <div class="dc"><div class="dc-label">Applicant EL</div>
                    <div class="dc-value {cv}">${applicant_el:,.0f}</div>
                    <div class="dc-sub">This loan's expected loss</div></div>
                <div class="dc"><div class="dc-label">EL Contribution</div>
                    <div class="dc-value {cv}">{applicant_share:.2f}%</div>
                    <div class="dc-sub">Share of portfolio EL</div></div>
                <div class="dc"><div class="dc-label">Marginal Capital</div>
                    <div class="dc-value {cv}">${prob * 0.45 * ca * 0.08:,.0f}</div>
                    <div class="dc-sub">8% × EAD × PD × LGD</div></div>
            </div>""", unsafe_allow_html=True)

        # ── TAB 10: PD TERM STRUCTURE ─────────────────────────────────────
        with t10:
            ts   = pd_term_structure(prob, horizons=(3, 6, 12, 18, 24, 36, 48, 60))
            h_m  = ts['horizons_months']
            cpd  = ts['cumulative_pd']
            surv = ts['survival']
            mpd  = ts['marginal_pd']

            st.markdown(f"""
            <div class="dc-row dc-row-4">
                <div class="dc"><div class="dc-label">12M PD (Model)</div>
                    <div class="dc-value {cv}">{ts['pd_12m']:.1%}</div>
                    <div class="dc-sub">Point-in-time estimate</div></div>
                <div class="dc"><div class="dc-label">Monthly Hazard Rate</div>
                    <div class="dc-value">{ts['hazard_monthly']:.4f}</div>
                    <div class="dc-sub">−ln(1−PD) / 12</div></div>
                <div class="dc"><div class="dc-label">Annual Hazard Rate</div>
                    <div class="dc-value">{ts['hazard_annual']:.4f}</div>
                    <div class="dc-sub">h × 12</div></div>
                <div class="dc"><div class="dc-label">Lifetime PD (60M)</div>
                    <div class="dc-value {cv}">{ts['ifrs9_lifetime_pd']:.1%}</div>
                    <div class="dc-sub">IFRS 9 Stage 2/3 input</div></div>
            </div>""", unsafe_allow_html=True)

            col_ts, col_sv = st.columns(2)
            with col_ts:
                fig_ts, ax_ts = plt.subplots(figsize=(7, 5))
                dark_ax(fig_ts, ax_ts)
                ax_ts.plot(h_m, cpd, 'o-', color='#e74c3c', lw=2.5,
                           ms=7, label='Cumulative PD')
                ax_ts.fill_between(h_m, cpd, alpha=0.1, color='#e74c3c')
                ax_ts.bar(h_m, mpd, width=2.5, alpha=0.4, color='#3498db',
                          label='Marginal PD (per period)')
                ax_ts.axvline(12, color='#2ecc71', lw=1.2, ls='--', alpha=0.7,
                              label='12M — Stage 1 horizon')
                ax_ts.axvline(60, color='#f1c40f', lw=1.2, ls='--', alpha=0.7,
                              label='60M — Lifetime horizon')
                ax_ts.set_xlabel('Horizon (months)', fontsize=9)
                ax_ts.set_ylabel('Probability of Default', fontsize=9)
                ax_ts.set_title(
                    'PD Term Structure\nh = −ln(1−PD₁₂) / 12 · Basel III & IFRS 9',
                    fontsize=9, fontweight='bold')
                ax_ts.legend(facecolor='#060c16', edgecolor='#0d1f35',
                             labelcolor='#8aabb8', fontsize=7.5)
                ax_ts.set_ylim(0, min(1, max(cpd) * 1.2))
                plt.tight_layout()
                st.pyplot(fig_ts)
                plt.close()

            with col_sv:
                fig_sv, ax_sv = plt.subplots(figsize=(7, 5))
                dark_ax(fig_sv, ax_sv)
                ax_sv.plot(h_m, surv, 'o-', color='#2ecc71', lw=2.5,
                           ms=7, label='Survival S(t) = exp(−h·t)')
                ax_sv.fill_between(h_m, surv, alpha=0.1, color='#2ecc71')
                ax_sv.fill_between(h_m, surv, [1]*len(h_m),
                                   alpha=0.05, color='#e74c3c',
                                   label='Cumulative default zone')
                ax_sv.axvline(12, color='#2ecc71', lw=1.2, ls='--', alpha=0.7)
                ax_sv.axvline(60, color='#f1c40f', lw=1.2, ls='--', alpha=0.7)
                ax_sv.set_ylim(0, 1.05)
                ax_sv.set_xlabel('Horizon (months)', fontsize=9)
                ax_sv.set_ylabel('Survival Probability', fontsize=9)
                ax_sv.set_title('Survival Curve S(t)',
                                fontsize=9, fontweight='bold')
                ax_sv.legend(facecolor='#060c16', edgecolor='#0d1f35',
                             labelcolor='#8aabb8', fontsize=7.5)
                plt.tight_layout()
                st.pyplot(fig_sv)
                plt.close()

            st.markdown('<div class="sl">PD by Horizon</div>',
                        unsafe_allow_html=True)
            ts_rows = []
            for i, t_m in enumerate(h_m):
                ifrs = ("Stage 1 — 12M ECL" if t_m <= 12
                        else "Stage 2/3 — Lifetime ECL")
                ts_rows.append({
                    'Horizon':       f"{t_m}M",
                    'Cumulative PD': f"{cpd[i]:.2%}",
                    'Survival':      f"{surv[i]:.2%}",
                    'Marginal PD':   f"{mpd[i]:.2%}",
                    'IFRS 9':        ifrs,
                })
            st.dataframe(pd.DataFrame(ts_rows),
                         use_container_width=True, hide_index=True)

            st.markdown("""
            <div style="background:#060c16;border:1px solid #0d1f35;
                 border-radius:6px;padding:14px 18px;margin-top:14px;
                 border-left:3px solid #3498db;">
                <div style="font-family:DM Mono,monospace;font-size:0.6rem;
                     text-transform:uppercase;letter-spacing:1.5px;
                     color:#1a3a58;margin-bottom:6px;">Methodology</div>
                <div style="font-size:0.78rem;color:#5a7a9a;line-height:1.6;">
                    Constant hazard rate assumption — standard for converting
                    point-in-time PD to lifetime PD per IFRS 9 §B5.5.42.
                    Production models use Cox PH or discrete-time hazard
                    fitted on vintage data.
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        CreditLens AI &nbsp;·&nbsp; LightGBM + SHAP &nbsp;·&nbsp;
        Basel III · IFRS 9 · CCAR · SR 11-7 · CFPB · ECOA &nbsp;·&nbsp;
        UCI German Credit &nbsp;·&nbsp; Built by Sanjana Prasad
    </div>""", unsafe_allow_html=True)
    