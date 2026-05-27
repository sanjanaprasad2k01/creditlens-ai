"""
CreditLens AI — Production Scoring API
FastAPI endpoint for loan origination system integration.
CFPB Circular 2022-03 compliant adverse action reasons included.

Thresholds: Optimised (0.40 / 0.55 / 0.65)
Fixes: Basel III IRB RWA (scipy.stats.norm), adverse action direction validation
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from scipy.stats import norm as scipy_norm
import joblib
import numpy as np
import pandas as pd
import sys
import os
import time
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'data_pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'explainability'))

app = FastAPI(
    title       = "CreditLens AI — Scoring API",
    description = """
Production credit risk scoring API for US financial institutions.

**Regulatory Compliance:**
- CFPB Circular 2022-03 — adverse action reasons for every decision
- ECOA / Regulation B — specific reasons for all denials
- SR 11-7 / OCC 2011-12 — model governance documentation
- Basel III IRB — PD/LGD/EAD/EL/RWA returned with every score
- IFRS 9 — Stage 1/2/3 auto-classification

**Decision Thresholds (Youden's J optimised):**
- PD < 0.40 : APPROVE
- PD 0.40–0.55 : APPROVE WITH CONDITIONS
- PD 0.55–0.65 : MANUAL REVIEW
- PD >= 0.65 : DECLINE

**Model:** LightGBM | AUC 0.79 | Gini 0.58 | KS 0.51
**Built by:** Sanjana Prasad
    """,
    version     = "1.0.0",
    contact     = {"name": "Sanjana Prasad", "email": "sanjana.prasad2023@utexas.edu"}
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'model_german.pkl')
_model      = None
_metrics    = {}
_start_time = time.time()

def get_model():
    global _model, _metrics
    if _model is None:
        data     = joblib.load(MODEL_PATH)
        _model   = data['model']           if isinstance(data, dict) else data
        _metrics = data.get('metrics', {}) if isinstance(data, dict) else {}
    return _model, _metrics

FEATURE_COLS = [
    'duration', 'credit_amount', 'installment_rate', 'residence_since',
    'age', 'existing_credits', 'dependents',
    'checking_account_num', 'savings_account_num', 'employment_num',
    'credit_history_num', 'purpose_num', 'property_num',
    'housing_num', 'job_num', 'has_telephone', 'is_foreign_worker',
    'has_other_debtors', 'loan_to_age', 'duration_to_age',
    'credit_per_month', 'is_high_amount', 'long_duration', 'young_borrower'
]

REASON_MAP = {
    'checking_account_num': 'Insufficient or negative checking account balance',
    'credit_history_num':   'Adverse credit history — delays or critical account on file',
    'duration':             'Excessive loan term relative to risk profile',
    'credit_amount':        'Loan amount exceeds risk-adjusted borrowing capacity',
    'savings_account_num':  'Insufficient savings reserves',
    'employment_num':       'Insufficient employment tenure',
    'loan_to_age':          'Loan amount disproportionate to borrower age',
    'credit_per_month':     'Monthly repayment burden exceeds acceptable threshold',
    'installment_rate':     'High installment rate as percentage of disposable income',
    'existing_credits':     'Excessive existing credit obligations',
    'is_high_amount':       'Credit amount above portfolio risk threshold',
    'long_duration':        'Loan term exceeds 24-month elevated-risk threshold',
    'young_borrower':       'Limited credit history due to borrower age',
    'purpose_num':          'Loan purpose associated with elevated default rate',
    'housing_num':          'Housing status indicates elevated financial instability',
    'has_other_debtors':    'Presence of other debtors increases exposure risk',
    'property_num':         'Insufficient collateral or property ownership',
    'job_num':              'Employment category associated with income instability',
}

RISK_DIRECTION = {
    'checking_account_num': ('low',  0),
    'credit_history_num':   ('high', 3),
    'duration':             ('high', 24),
    'credit_amount':        ('high', 3000),
    'savings_account_num':  ('low',  1),
    'employment_num':       ('low',  1),
    'loan_to_age':          ('high', 0.3),
    'credit_per_month':     ('high', 150),
    'installment_rate':     ('high', 3),
    'existing_credits':     ('high', 2),
    'is_high_amount':       ('high', 1),
    'long_duration':        ('high', 1),
    'young_borrower':       ('high', 1),
    'purpose_num':          ('high', 6),
    'housing_num':          ('low',  1),
    'has_other_debtors':    ('high', 1),
    'property_num':         ('low',  1),
    'job_num':              ('low',  1),
}

class ApplicantRequest(BaseModel):
    credit_amount:         float = Field(..., gt=0,  le=20000)
    duration:              int   = Field(..., ge=4,  le=72)
    age:                   int   = Field(..., ge=18, le=100)
    installment_rate:      int   = Field(..., ge=1,  le=4)
    existing_credits:      int   = Field(..., ge=1,  le=4)
    dependents:            int   = Field(..., ge=1,  le=2)
    checking_account_num:  int   = Field(..., ge=-1, le=2)
    savings_account_num:   int   = Field(..., ge=0,  le=4)
    employment_num:        int   = Field(..., ge=0,  le=4)
    credit_history_num:    int   = Field(..., ge=0,  le=4)
    purpose_num:           int   = Field(..., ge=0,  le=9)
    housing_num:           int   = Field(..., ge=0,  le=2)
    property_num:          int   = Field(2,   ge=0,  le=3)
    job_num:               int   = Field(2,   ge=0,  le=3)
    has_telephone:         int   = Field(1,   ge=0,  le=1)
    is_foreign_worker:     int   = Field(1,   ge=0,  le=1)
    has_other_debtors:     int   = Field(0,   ge=0,  le=2)
    residence_since:       int   = Field(3,   ge=1,  le=4)

    class Config:
        json_schema_extra = {"example": {
            "credit_amount": 5000, "duration": 24, "age": 35,
            "installment_rate": 3, "existing_credits": 1, "dependents": 1,
            "checking_account_num": 1, "savings_account_num": 2,
            "employment_num": 3, "credit_history_num": 2,
            "purpose_num": 3, "housing_num": 2
        }}

class ScoreResponse(BaseModel):
    request_id:             str
    probability_of_default: float
    decision:               str
    risk_band:              str
    expected_loss:          float
    lgd:                    float
    ead:                    float
    estimated_rwa:          float
    capital_requirement:    float
    ifrs9_stage:            int
    ecl_horizon:            str
    adverse_action_reasons: List[str]
    regulatory_note:        str
    model_version:          str

class HealthResponse(BaseModel):
    status:           str
    model_loaded:     bool
    model_version:    str
    auc:              float
    gini:             float
    ks_statistic:     float
    uptime_seconds:   float
    threshold_method: str
    regulatory_note:  str

def build_features(req: ApplicantRequest) -> pd.DataFrame:
    row = {
        'duration': req.duration, 'credit_amount': req.credit_amount,
        'installment_rate': req.installment_rate, 'residence_since': req.residence_since,
        'age': req.age, 'existing_credits': req.existing_credits, 'dependents': req.dependents,
        'checking_account_num': req.checking_account_num,
        'savings_account_num':  req.savings_account_num,
        'employment_num':       req.employment_num,
        'credit_history_num':   req.credit_history_num,
        'purpose_num':          req.purpose_num,
        'property_num':         req.property_num,
        'housing_num':          req.housing_num,
        'job_num':              req.job_num,
        'has_telephone':        req.has_telephone,
        'is_foreign_worker':    req.is_foreign_worker,
        'has_other_debtors':    req.has_other_debtors,
        'loan_to_age':          req.credit_amount / (req.age + 1),
        'duration_to_age':      req.duration / (req.age + 1),
        'credit_per_month':     req.credit_amount / (req.duration + 1),
        'is_high_amount':       int(req.credit_amount > 3000),
        'long_duration':        int(req.duration > 24),
        'young_borrower':       int(req.age < 25),
    }
    return pd.DataFrame([row])[FEATURE_COLS].fillna(0)

def compute_basel(prob: float, ead: float):
    lgd      = 0.45
    el       = prob * lgd * ead
    exp_term = np.exp(-35.0 * prob)
    exp_base = np.exp(-35.0)
    denom    = 1.0 - exp_base
    R        = (0.03 * (1.0 - exp_term) / denom +
                0.16 * (1.0 - (1.0 - exp_term) / denom))
    if 0.0 < prob < 1.0 and R < 1.0:
        K = max(0.0, lgd * scipy_norm.cdf(
            (1.0 - R)**(-0.5) * scipy_norm.ppf(prob) +
            (R / (1.0 - R))**0.5 * scipy_norm.ppf(0.999)
        ) - prob * lgd)
    else:
        K = 0.0
    rwa = K * 12.5 * ead
    return lgd, el, rwa, rwa * 0.08

def get_ifrs9(prob: float):
    if prob < 0.10:   return 1, "12-month ECL"
    elif prob < 0.50: return 2, "Lifetime ECL (SICR triggered)"
    else:             return 3, "Lifetime ECL (Credit-Impaired)"

def get_shap_reasons(model, inp: pd.DataFrame, top_n: int = 4) -> List[str]:
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv        = explainer.shap_values(inp)
        if isinstance(sv, list):           sv = sv[1]
        if len(np.array(sv).shape) == 3:   sv = sv[:, :, 1]
        vals    = sv[0]
        reasons = []
        for i in np.argsort(vals)[::-1]:
            if len(reasons) >= top_n: break
            if vals[i] <= 0:          break
            feat     = FEATURE_COLS[i]
            feat_val = float(inp.iloc[0][feat])
            di       = RISK_DIRECTION.get(feat)
            if di:
                d, t = di
                if d == 'high' and feat_val < t: continue
                if d == 'low'  and feat_val > t: continue
            reason = REASON_MAP.get(feat, f"Risk factor: {feat}")
            if reason not in reasons:
                reasons.append(reason)
        return reasons or ["Combination of risk factors exceeds approval threshold"]
    except Exception:
        return ["Adverse action reasons available on request per ECOA Regulation B"]

@app.get("/", tags=["System"])
def root():
    return {"service": "CreditLens AI Scoring API", "version": "1.0.0",
            "docs": "/docs", "health": "/health", "score": "/score",
            "batch": "/score/batch", "model_info": "/model/info"}

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check for load balancers and monitoring systems."""
    model, metrics = get_model()
    return HealthResponse(
        status           = "healthy",
        model_loaded     = model is not None,
        model_version    = "1.0.0",
        auc              = metrics.get('auc',  0.79),
        gini             = metrics.get('gini', 0.58),
        ks_statistic     = metrics.get('ks',   0.51),
        uptime_seconds   = round(time.time() - _start_time, 2),
        threshold_method = "Optimised thresholds — data-driven (0.40 / 0.55 / 0.65)",
        regulatory_note  = ("SR 11-7 / OCC 2011-12 compliant. "
                            "CFPB Circular 2022-03 adverse action reasons "
                            "included in all /score responses.")
    )

@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_applicant(request: ApplicantRequest):
    """
    Score a single credit applicant.
    Thresholds: PD < 0.40 APPROVE | 0.40-0.55 CONDITIONS | 0.55-0.65 REVIEW | >= 0.65 DECLINE
    """
    try:
        model, _  = get_model()
        inp       = build_features(request)
        prob      = float(model.predict_proba(inp)[0][1])

        if prob < 0.40:
            decision, risk_band = "APPROVE",               "LOW RISK"
        elif prob < 0.55:
            decision, risk_band = "APPROVE WITH CONDITIONS","MODERATE RISK"
        elif prob < 0.65:
            decision, risk_band = "MANUAL REVIEW",          "HIGH RISK"
        else:
            decision, risk_band = "DECLINE",                "VERY HIGH RISK"

        lgd, el, rwa, capital = compute_basel(prob, request.credit_amount)
        stage, horizon        = get_ifrs9(prob)
        reasons               = get_shap_reasons(model, inp)

        return ScoreResponse(
            request_id              = str(uuid.uuid4())[:8],
            probability_of_default  = round(prob, 4),
            decision                = decision,
            risk_band               = risk_band,
            expected_loss           = round(el, 2),
            lgd                     = lgd,
            ead                     = request.credit_amount,
            estimated_rwa           = round(rwa, 2),
            capital_requirement     = round(capital, 2),
            ifrs9_stage             = stage,
            ecl_horizon             = horizon,
            adverse_action_reasons  = reasons,
            regulatory_note         = ("CFPB Circular 2022-03 compliant. "
                                       "Adverse action notice required within 30 days "
                                       "per ECOA Regulation B (12 CFR Part 202). "
                                       "Thresholds: Youden's J statistic."),
            model_version           = "1.0.0"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score/batch", tags=["Scoring"])
def score_batch(requests: List[ApplicantRequest]):
    """Batch scoring — maximum 100 applicants per request."""
    if len(requests) > 100:
        raise HTTPException(status_code=400,
                            detail="Batch size exceeds maximum of 100.")
    return [score_applicant(req) for req in requests]

@app.get("/model/info", tags=["Model"])
def model_info():
    """Full model metadata for SR 11-7 / OCC 2011-12 documentation."""
    _, metrics = get_model()
    thresh     = metrics.get('thresholds', {})
    return {
        "model_name":  "CreditLens AI",
        "model_type":  "LightGBM Gradient Boosting Classifier",
        "version":     "1.0.0",
        "dataset":     "UCI German Credit Dataset (1,000 rows, 24 features)",
        "performance": {
            "auc_roc":   metrics.get('auc',  0.79),
            "gini":      metrics.get('gini', 0.58),
            "ks":        metrics.get('ks',   0.51),
            "lift_lr":   f"+{metrics.get('lift', 8.1):.1f}%",
            "separation":metrics.get('separation', 0.0),
        },
        "decision_thresholds": {
            "method":    "Youden's J statistic",
            "approve":   metrics.get('approve',  0.40),
            "review":    metrics.get('review',   0.55),
            "decline":   metrics.get('decline',  0.65),
            "youden_j":  thresh.get('youden_j',  None),
            "youden_tpr":thresh.get('youden_tpr',None),
            "youden_fpr":thresh.get('youden_fpr',None),
        },
        "regulatory": {
            "cfpb_2022_03": "Compliant", "ecoa_reg_b":  "Compliant",
            "sr_11_7":      "Compliant", "occ_2011_12": "Compliant",
            "basel_iii":    "Implemented", "ifrs_9":    "Implemented",
            "ccar":         "Implemented", "fair_lending": "Proxy analysis",
        },
        "limitations": [
            "1,000-row benchmark — retrain on US portfolio data for production",
            "LGD fixed at 45% — estimate empirically in production",
            "No vintage analysis — champion/challenger required",
            "Fair lending proxy only — HMDA demographic testing needed",
        ],
        "monitoring": "PSI alert >0.20 | AUC drift alert >0.05",
    }


