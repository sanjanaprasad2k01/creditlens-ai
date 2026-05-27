"""CreditLens AI — Streamlit Application."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import joblib

# Add src folders to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'data_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'explainability'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'genai'))

from preprocessing import prepare_data, engineer_features, FEATURE_COLS
from shap_engine import get_explainer, explain_single, get_adverse_reasons
from narrative_engine import generate_risk_report
from trainer import train_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditLens AI",
    page_icon="🏦",
    layout="wide"
)

# ── Load model and data (cached so it only runs once) ─────────────────────────
@st.cache_resource
def load_all():
    """Load or train model, create SHAP explainer."""
    X, y, feats = prepare_data(
        r"C:\Users\sanja\Desktop\creditlens-ai\data\raw\credit_data.csv"
    )
    model_path = r"C:\Users\sanja\Desktop\creditlens-ai\artifacts\model.pkl"

    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model, _, _, _ = train_model(X, y, save_path=model_path)

    explainer = get_explainer(model)
    return model, explainer, X, y, feats

model, explainer, X, y, feats = load_all()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏦 CreditLens AI")
st.caption("ML-powered credit risk assessment with SHAP explainability")
st.divider()

# ── Sidebar — Applicant Inputs ────────────────────────────────────────────────
st.sidebar.header("📋 Applicant Details")

loan_amnt      = st.sidebar.number_input("Loan Amount ($)",        1000,  40000,  10000, step=500)
annual_inc     = st.sidebar.number_input("Annual Income ($)",     20000, 500000,  65000, step=1000)
dti            = st.sidebar.slider("Debt-to-Income Ratio (%)",        0,     50,     15)
revol_util     = st.sidebar.slider("Credit Utilization (%)",          0,    100,     30)
revol_bal      = st.sidebar.number_input("Revolving Balance ($)",      0, 100000,  10000, step=500)
total_acc      = st.sidebar.slider("Total Accounts",                   1,     60,     12)
delinq_2yrs    = st.sidebar.slider("Delinquencies (past 2 yrs)",       0,     10,      0)
pub_rec        = st.sidebar.slider("Public Records",                   0,      5,      0)
inq_last_6mths = st.sidebar.slider("Credit Inquiries (6 months)",      0,     10,      1)
emp_length     = st.sidebar.selectbox("Employment Length", [
    "< 1 year", "1 year", "2 years", "3 years", "4 years",
    "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
], index=4)

assess_btn = st.sidebar.button("🔍 Assess Credit Risk", use_container_width=True)

# ── Main content ──────────────────────────────────────────────────────────────
if not assess_btn:
    st.info("👈 Fill in the applicant details on the left and click **Assess Credit Risk**.")

    st.subheader("📊 Training Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Applicants", f"{len(X):,}")
    col2.metric("Features Used", len(feats))
    col3.metric("Default Rate", f"{y.mean():.1%}")
    col4.metric("Model", "LightGBM")

else:
    # ── Build input row ───────────────────────────────────────────────────────
    emp_map = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
        '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
        '8 years': 8, '9 years': 9, '10+ years': 10
    }
    emp_length_num     = emp_map[emp_length]
    dti_ratio          = dti / 100
    credit_utilization = revol_util / 100

    raw = pd.DataFrame([{
        'loan_amnt':      loan_amnt,
        'annual_inc':     annual_inc,
        'dti':            dti,
        'revol_util':     revol_util,
        'revol_bal':      revol_bal,
        'total_acc':      total_acc,
        'delinq_2yrs':    delinq_2yrs,
        'pub_rec':        pub_rec,
        'emp_length':     emp_length,
        'inq_last_6mths': inq_last_6mths,
    }])

    # Feature engineering
    raw['emp_length_num']     = emp_length_num
    raw['loan_to_income']     = loan_amnt / (annual_inc + 1)
    raw['dti_ratio']          = dti_ratio
    raw['credit_utilization'] = credit_utilization
    raw['has_delinquency']    = int(delinq_2yrs > 0)
    raw['has_public_record']  = int(pub_rec > 0)
    raw['high_inquiry']       = int(inq_last_6mths > 2)
    raw['dti_x_utilization']  = dti_ratio * credit_utilization
    raw['repayment_stress']   = raw['loan_to_income'].values[0] * dti_ratio
    raw['account_diversity']  = np.log1p(total_acc)

    inp = raw[feats].fillna(0)

    # ── Prediction ────────────────────────────────────────────────────────────
    prob = model.predict_proba(inp)[0][1]

    # ── Risk rating ───────────────────────────────────────────────────────────
    if prob < 0.35:
        rating, color, rec = "LOW RISK", "🟢", "APPROVE — standard terms"
    elif prob < 0.50:
        rating, color, rec = "MODERATE RISK", "🟡", "APPROVE WITH CONDITIONS"
    elif prob < 0.65:
        rating, color, rec = "HIGH RISK", "🟠", "MANUAL REVIEW REQUIRED"
    else:
        rating, color, rec = "VERY HIGH RISK", "🔴", "DECLINE — exceeds threshold"

    # ── Top metrics row ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Default Probability", f"{prob:.1%}")
    col2.metric("Risk Rating", f"{color} {rating}")
    col3.metric("Recommendation", rec)
    col4.metric("Loan Amount", f"${loan_amnt:,}")

    st.divider()

    # ── SHAP explanation ──────────────────────────────────────────────────────
    expl_df            = explain_single(explainer, inp, feats)
    risk_factors       = expl_df[expl_df['shap_value'] > 0].to_dict('records')
    protective_factors = expl_df[expl_df['shap_value'] < 0].to_dict('records')
    adverse_reasons    = get_adverse_reasons(expl_df)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📊 Feature Contributions (SHAP)")
        fig, ax = plt.subplots(figsize=(8, 6))
        top = expl_df.head(12).sort_values('shap_value')
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in top['shap_value']]
        ax.barh(top['feature'], top['shap_value'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on default risk)")
        ax.set_title("Red = increases risk | Blue = reduces risk")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("⚠️ Adverse Action Reasons (ECOA)")
        for i, reason in enumerate(adverse_reasons, 1):
            st.error(f"{i}. {reason}")

        st.subheader("✅ Protective Factors")
        for f in protective_factors[:3]:
            v = f.get('value', 0)
            st.success(f"**{f['feature']}** = {v:.3f} → reduces risk by {abs(f['shap_value']):.4f}")

    st.divider()

    # ── Risk Report ───────────────────────────────────────────────────────────
    st.subheader("📄 Risk Assessment Report")
    report = generate_risk_report(prob, risk_factors, protective_factors)
    st.markdown(report)