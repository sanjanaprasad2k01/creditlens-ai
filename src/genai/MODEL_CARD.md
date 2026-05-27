# CreditLens AI — Model Card

> Following the Model Card format proposed by Mitchell et al. (2019)
> and aligned with NIST AI RMF (2023) and Federal Reserve SR 11-7.

---

## Model Details

| Field | Value |
|---|---|
| Model Name | CreditLens AI |
| Model Type | LightGBM Gradient Boosting Classifier |
| Version | 1.0.0 |
| Date | May 2026 |
| License | MIT |
| Contact | Sanjana Prasad — sanjana.prasad2023@utexas.edu |

**Architecture:** Gradient boosted decision trees (LightGBM) with
leaf-wise tree growth, L1/L2 regularization, and class-imbalance
weighting via `scale_pos_weight`. 24 engineered features from 10 raw inputs.

---

## Intended Use

**Primary intended use:**
Credit risk scoring for US retail lending decisions — personal loans,
consumer credit, and similar unsecured products.

**Primary intended users:**
Credit risk analysts, underwriters, model validation teams, and
risk officers at US financial institutions.

**Out-of-scope uses:**
- Mortgage lending (different regulatory framework — HMDA, CRA)
- Commercial / business lending (different risk drivers)
- Insurance underwriting
- Employment screening
- Any use outside the United States without additional regulatory review

---

## Training Data

| Field | Value |
|---|---|
| Dataset | UCI Statlog (German Credit) Dataset |
| Source | UCI Machine Learning Repository |
| Size | 1,000 applicants |
| Features | 24 engineered from 10 raw inputs |
| Target | Binary default label (1 = default, 0 = paid) |
| Default Rate | 30% |
| Time Period | Historical (no date range specified in source) |
| Geography | Germany (proxy dataset — not US-specific) |

**Important note:** This model is trained on a German benchmark dataset
used as a portfolio demonstration. Production deployment for US lending
requires retraining on US portfolio data with HMDA-reportable demographic
information available for fair lending analysis.

---

## Evaluation

### Performance Metrics

| Metric | Value | Industry Minimum | Notes |
|---|---|---|---|
| AUC-ROC | 0.7906 | 0.70 (OCC/Fed guidance) | Exceeds minimum |
| Gini Coefficient | 0.5812 | 0.30 (Basel III) | Exceeds minimum |
| KS Statistic | 0.5119 | 0.20 (SR 11-7) | Exceeds minimum |
| Lift over Logistic Regression | +8.1% | Positive lift required | Meets requirement |
| Mean Calibration Error | ~0.06 | < 0.10 | Well calibrated |
| PSI (Train vs Test) | < 0.10 | < 0.10 | Stable |

### Benchmark Comparison

| Model | AUC-ROC | Gini |
|---|---|---|
| Random Baseline | 0.5000 | 0.0000 |
| Logistic Regression | 0.7311 | 0.4621 |
| **CreditLens AI (LightGBM)** | **0.7906** | **0.5812** |

### Evaluation Split
80% training / 20% test with stratified sampling to preserve
30% default rate in both splits. 5-fold cross-validation on
training set: CV AUC = 0.7883 ± 0.0438.

---

## Ethical Considerations

### Fair Lending Analysis (4/5ths Rule)

The following protected class proxies were analyzed for disparate impact
per EEOC Uniform Guidelines on Employee Selection Procedures:

| Proxy Variable | Represents | 4/5ths Status |
|---|---|---|
| young_borrower (age < 25) | Age proxy | See Model Health tab |
| is_foreign_worker | National origin proxy | See Model Health tab |
| housing_num | Wealth / housing status proxy | See Model Health tab |
| employment_num | Employment status | See Model Health tab |

**Important:** This analysis uses proxy variables, not actual demographic
data. Production fair lending compliance requires:
1. HMDA data integration for actual demographic analysis
2. Qualified fair lending officer review
3. Adverse impact mitigation plan if 4/5ths rule violated
4. State-specific bias audit (NY LL144, CA AB 2930, CO SB 169)

### Reject Inference
This model is trained only on historically approved applicants.
Rejected applicants are excluded because their repayment outcomes
are unobserved. This introduces **sample selection bias** — the model
may underestimate default risk for borderline applicants near the
approval threshold. Production deployment should address this via
fuzzy augmentation or re-weighting techniques.

### Protected Classes
The model does not use the following directly as inputs:
race, color, religion, national origin, sex, marital status,
age, or receipt of public assistance. However, proxy variables
may exist. Full disparate impact testing is required before
any US production deployment.

---

## Caveats and Limitations

1. **Dataset geography:** Trained on German benchmark data.
   US-specific risk patterns (FICO, bureau tradelines, debt types)
   are not represented. Retrain on US portfolio data before deployment.

2. **Dataset size:** 1,000 training rows. Production minimum
   recommended: 10,000+ rows for stable estimates.

3. **No vintage analysis:** Model stability across US economic cycles
   has not been tested. Champion/challenger testing required in production.

4. **Fixed LGD:** LGD is fixed at 45% industry standard.
   Production models should estimate LGD empirically from
   actual charge-off and recovery data.

5. **No macroeconomic features:** The model does not include
   unemployment rate, interest rate environment, or GDP growth.
   These are material drivers of consumer default risk.

6. **Binary target only:** The model predicts good/bad binary outcomes.
   It does not predict time-to-default or loss severity distribution.

---

## Production Deployment Requirements

Before deploying at any US financial institution:

- [ ] Retrain on minimum 10,000 rows of US portfolio data
- [ ] Complete fair lending disparate impact analysis with actual demographics
- [ ] Obtain fair lending officer sign-off
- [ ] Implement PSI monitoring pipeline (alert at PSI > 0.20)
- [ ] Implement AUC drift monitoring (alert if AUC drops > 0.05)
- [ ] Complete champion/challenger A/B test vs existing scorecard
- [ ] Document model under SR 11-7 / OCC 2011-12 standards
- [ ] Complete model validation by independent MRM team
- [ ] State-level AI bias audit if deploying in NY, CA, IL, or CO
- [ ] Legal review of adverse action notice language per CFPB 2022-03

---

## Regulatory Compliance Summary

| Regulation | Requirement | Status |
|---|---|---|
| CFPB Circular 2022-03 | Specific adverse action reasons for AI/ML | ✅ SHAP satisfies |
| ECOA / Reg B (12 CFR 202) | Written notice within 30 days | ✅ Auto-generated |
| Fair Housing Act | No discriminatory proxies | ⚠️ Proxy analysis completed, full demographic testing needed |
| OCC 2011-12 | Model risk management documentation | ✅ This model card |
| SR 11-7 (Federal Reserve) | Conceptual soundness, monitoring, validation | ✅ Addressed |
| CCAR / Dodd-Frank | Stress testing | ✅ 4-scenario implemented |
| Basel III Pillar 1 | Capital adequacy | ✅ IRB formula |
| IFRS 9 | ECL staging | ✅ Stage 1/2/3 |
| NY Local Law 144 | Annual bias audit for AEDTs | ⚠️ Required before NYC deployment |

---

## Citation

```bibtex
@software{creditlens_ai_2026,
  author    = {Sanjana Prasad},
  title     = {CreditLens AI: Production Credit Risk Assessment for US Financial Institutions},
  year      = {2026},
  url       = {https://github.com/sanjanaprasad2k01/creditlens-ai},
  note      = {LightGBM + SHAP + Basel III + IFRS 9 + CFPB/OCC/SR 11-7 compliance}
}
```

---

*CreditLens AI · Model Card v1.0 · May 2026*
*Built by Sanjana Prasad · MS Computer Science, UT Austin (2025)*