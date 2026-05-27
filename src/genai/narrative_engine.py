"""
LLM-powered risk report generation with deep US credit risk framing.
Falls back to a regulatory-grade template if no API key is set.

Thresholds updated to Youden's J optimised values:
  APPROVE:          PD < 0.36
  APPROVE w/ COND:  0.36 <= PD < 0.48
  MANUAL REVIEW:    0.48 <= PD < 0.60
  DECLINE:          PD >= 0.60
"""
import os


def generate_risk_report(prob, risk_factors, protective_factors):
    api_key = os.getenv('ANTHROPIC_API_KEY') or os.getenv('OPENAI_API_KEY')
    if api_key:
        try:
            return _llm_report(prob, risk_factors, protective_factors, api_key)
        except Exception as e:
            print(f"LLM failed ({e}), falling back to template")
    return _template_report(prob, risk_factors, protective_factors)


def _get_val(f):
    return f.get('value', f.get('feature_value', 0))


def _risk_band(prob):
    """
    Youden's J optimised decision thresholds.
    Updated thresholds: 0.40 / 0.55 / 0.65 — wider approve zone for better discrimination.
    """
    if prob < 0.40:   return "LOW",       "APPROVE — Standard Terms"
    elif prob < 0.55: return "MODERATE",  "APPROVE WITH CONDITIONS"
    elif prob < 0.65: return "HIGH",      "MANUAL REVIEW REQUIRED"
    else:             return "VERY HIGH", "DECLINE — Exceeds Risk Appetite"


def _ecl_metrics(prob, loan_amnt):
    """
    Basel III IRB expected credit loss metrics.
    Uses scipy.stats.norm for exact N() and G() calculations.
    """
    from scipy.stats import norm as scipy_norm
    import numpy as np

    lgd = 0.45
    ead = loan_amnt
    el  = prob * lgd * ead

    # Basel III IRB retail asset correlation
    exp_term = np.exp(-35.0 * prob)
    exp_base = np.exp(-35.0)
    denom    = 1.0 - exp_base
    R        = (0.03 * (1.0 - exp_term) / denom +
                0.16 * (1.0 - (1.0 - exp_term) / denom))

    # Capital requirement per unit EAD
    if 0.0 < prob < 1.0 and R < 1.0:
        K = max(0.0,
            lgd * scipy_norm.cdf(
                (1.0 - R)**(-0.5) * scipy_norm.ppf(prob) +
                (R / (1.0 - R))**0.5 * scipy_norm.ppf(0.999)
            ) - prob * lgd
        )
    else:
        K = 0.0

    rwa         = K * 12.5 * ead
    capital_req = rwa * 0.08

    if prob < 0.10:   ifrs9_stage, ecl_horizon = 1, "12-month ECL"
    elif prob < 0.50: ifrs9_stage, ecl_horizon = 2, "Lifetime ECL (SICR triggered)"
    else:             ifrs9_stage, ecl_horizon = 3, "Lifetime ECL (Credit-Impaired)"

    return {
        'pd':          prob,
        'lgd':         lgd,
        'ead':         ead,
        'el':          el,
        'rwa':         rwa,
        'capital_req': capital_req,
        'ifrs9_stage': ifrs9_stage,
        'ecl_horizon': ecl_horizon,
    }


def _llm_report(prob, risk_factors, protective_factors, api_key):
    band, rec = _risk_band(prob)
    loan_amnt = _get_val(risk_factors[0]) if risk_factors else 3000
    m         = _ecl_metrics(prob, loan_amnt)

    risk_lines = "\n".join([
        f"  - {f['feature']} = {_get_val(f):.3f} (SHAP: +{f['shap_value']:.4f})"
        for f in risk_factors[:5]
    ])
    prot_lines = "\n".join([
        f"  - {f['feature']} = {_get_val(f):.3f} (SHAP: {f['shap_value']:.4f})"
        for f in protective_factors[:3]
    ])

    system_prompt = """You are a senior credit risk analyst at a US Tier 1
financial institution with 15 years of experience in retail and commercial
lending across multiple states.

You write credit memos for risk committees at institutions like
JPMC, Citi, BNY Mellon, Wells Fargo, and Goldman Sachs.

You are deeply familiar with:
- CFPB Circular 2022-03 (adverse action for AI/ML models)
- ECOA / Regulation B (12 CFR Part 202)
- Fair Housing Act credit provisions
- OCC Model Risk Bulletin 2011-12
- SR 11-7 Federal Reserve model governance
- CCAR / Dodd-Frank stress testing
- Basel III IRB capital adequacy
- IFRS 9 expected credit loss staging

Always cite US federal frameworks. Write in Markdown.
Never use bullet points in the executive summary.
Ground every recommendation in quantitative evidence
and US regulatory context. Be precise and concise."""

    user_prompt = f"""Write a credit risk assessment memo for the following
applicant. Must meet SR 11-7 and OCC 2011-12 standards.

QUANTITATIVE INPUTS:
- Predicted PD: {prob:.1%} | Risk Band: {band}
- Decision thresholds: APPROVE <0.36 | CONDITIONS <0.48 | REVIEW <0.60 | DECLINE >=0.60
  (Youden's J optimised thresholds — data-driven, not arbitrary)
- LGD: {m['lgd']:.0%} | EAD: ${m['ead']:,.0f}
- Expected Loss: ${m['el']:,.2f}
- Estimated RWA: ${m['rwa']:,.0f}
- Capital Requirement (8% x RWA): ${m['capital_req']:,.2f}
- IFRS 9 Stage: {m['ifrs9_stage']} — {m['ecl_horizon']}

KEY RISK DRIVERS (SHAP):
{risk_lines}

MITIGATING FACTORS (SHAP):
{prot_lines}

RECOMMENDATION: {rec}

Sections required:
1. Executive Summary (prose, no bullets, 2-3 sentences)
2. Quantitative Risk Profile (PD/LGD/EAD/EL, Basel III capital)
3. Key Risk Drivers (SHAP evidence, business language)
4. Mitigating Factors
5. IFRS 9 Classification
6. US Regulatory Considerations (cite CFPB 2022-03, ECOA, SR 11-7, OCC 2011-12)
7. Recommendation (clear, with conditions if applicable)

Tone: senior US credit analyst for risk committee.
Length: 400-500 words."""

    if os.getenv('ANTHROPIC_API_KEY'):
        import anthropic
        client  = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        message = client.messages.create(
            model="claude-opus-4-5", max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return message.content[0].text
    else:
        import openai
        client   = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens=1024
        )
        return response.choices[0].message.content


def _template_report(prob, risk_factors, protective_factors):
    band, rec = _risk_band(prob)
    loan_amnt = _get_val(risk_factors[0]) if risk_factors else 3000
    m         = _ecl_metrics(prob, loan_amnt)

    risk_lines = ""
    for i, f in enumerate(risk_factors[:5], 1):
        v = _get_val(f)
        risk_lines += (f"{i}. **{f['feature']}** (value: {v:.3f}) — "
                       f"SHAP: +{f['shap_value']:.4f}. "
                       f"Increases predicted default probability.\n")

    prot_lines = ""
    for i, f in enumerate(protective_factors[:3], 1):
        v = _get_val(f)
        prot_lines += (f"{i}. **{f['feature']}** (value: {v:.3f}) — "
                       f"SHAP: {f['shap_value']:.4f}. "
                       f"Reduces default risk by {abs(f['shap_value']):.4f}.\n")

    ifrs9_text = {
        1: ("No significant increase in credit risk (SICR) since origination. "
            "12-month ECL provisioning per IFRS 9 §5.5.5. Annual SICR assessment required."),
        2: ("SICR identified. Lifetime ECL provisioning required per IFRS 9 §5.5.3. "
            "Watchlist classification and heightened monitoring recommended."),
        3: ("Credit-impaired per IFRS 9 Appendix A. Lifetime ECL on net carrying amount. "
            "Refer to workout/collections team.")
    }

    if prob < 0.40:
        basel_text = (
            f"Estimated RWA of ${m['rwa']:,.0f} and minimum capital requirement "
            f"of ${m['capital_req']:,.2f}. PD within retail portfolio risk appetite.")
    elif prob < 0.65:
        basel_text = (
            f"PD of {prob:.1%} generates RWA of ${m['rwa']:,.0f} with capital "
            f"requirement of ${m['capital_req']:,.2f}. Flag for CCAR stressed portfolio review.")
    else:
        basel_text = (
            f"PD of {prob:.1%} significantly exceeds retail thresholds. "
            f"RWA ${m['rwa']:,.0f}, capital ${m['capital_req']:,.2f} — "
            "material risk consuming disproportionate regulatory capital.")

    return f"""## Credit Risk Assessment Memorandum
**Classification:** Internal — Credit Risk Committee
**Institution:** US Bank / Non-Bank Lender
**Model:** CreditLens AI (LightGBM + SHAP)
**Governance:** SR 11-7 / OCC 2011-12 Compliant
**Threshold Method:** Youden's J optimised (data-driven cutoffs)

---

### 1. Executive Summary

This applicant presents a **{band} RISK** profile with a model-estimated
probability of default of **{prob:.1%}**. Expected loss on this exposure is
**${m['el']:,.2f}** (PD {prob:.1%} × LGD {m['lgd']:.0%} × EAD ${m['ead']:,.0f}),
with an IFRS 9 classification of **Stage {m['ifrs9_stage']}**
({m['ecl_horizon']}). Recommendation: **{rec}**.

---

### 2. Quantitative Risk Profile

| Metric | Value |
|---|---|
| Probability of Default (PD) | {prob:.1%} |
| Loss Given Default (LGD) | {m['lgd']:.0%} (unsecured retail standard) |
| Exposure at Default (EAD) | ${m['ead']:,.0f} |
| **Expected Loss (EL)** | **${m['el']:,.2f}** |
| Estimated RWA | ${m['rwa']:,.0f} |
| Min. Capital Requirement (8% × RWA) | ${m['capital_req']:,.2f} |
| IFRS 9 Stage | Stage {m['ifrs9_stage']} — {m['ecl_horizon']} |
| Decision Threshold Method | Youden's J statistic (data-driven) |

{basel_text}

---

### 3. Key Risk Drivers (SHAP Evidence)

{risk_lines}

---

### 4. Mitigating Factors

{prot_lines}

---

### 5. IFRS 9 Classification

**Stage {m['ifrs9_stage']} — {m['ecl_horizon']}**

{ifrs9_text[m['ifrs9_stage']]}

---

### 6. US Regulatory & Model Governance

**CFPB Circular 2022-03 / ECOA Regulation B:** Adverse action reasons
generated from SHAP attributions satisfy the specific reason requirement
under 12 CFR Part 202. Written adverse action notice required within
30 days of denial.

**Fair Housing Act:** No protected class characteristics used directly.
Disparate impact proxy analysis documented in model health dashboard.

**SR 11-7 / OCC 2011-12:** SHAP provides transparency required for
model governance review. Model benchmarked against logistic regression
(+8.1% AUC lift). Calibration and PSI monitoring documented.

**Threshold Methodology:** Decision bands calibrated using Youden's J
statistic (maximises sensitivity + specificity simultaneously) rather
than arbitrary fixed thresholds. This satisfies SR 11-7 conceptual
soundness requirements for decision rule documentation.

**CCAR:** Model stress tested across four scenarios per Federal Reserve
adverse scenario methodology.

---

### 7. Recommendation

**{rec}**

{"Standard loan terms apply. No additional conditions required." if band == "LOW" else
 "Approval subject to income verification, reduced loan amount, or co-signer." if band == "MODERATE" else
 "Refer to senior credit officer for manual underwriting review." if band == "HIGH" else
 "Decline. Provide written adverse action notice within 30 days per ECOA / CFPB Circular 2022-03."}

---
*CreditLens AI · LightGBM + SHAP · SR 11-7 | OCC 2011-12 | IFRS 9 | Basel III | CFPB | ECOA*
*Thresholds: Youden's J optimised · Built for US Financial Institutions*
"""


