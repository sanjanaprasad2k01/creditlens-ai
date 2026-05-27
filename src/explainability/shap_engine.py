"""SHAP-based model explainability."""
import shap
import pandas as pd
import numpy as np
import sys
import os


def get_explainer(model):
    """Create a SHAP explainer for the model.
    TreeExplainer is optimized for tree-based models (XGBoost, LightGBM).
    It's fast because it uses the tree structure directly."""
    return shap.TreeExplainer(model)


def explain_single(explainer, X_row, feature_names):
    """Explain a single prediction.
    X_row: one row of features (one applicant)
    Returns: DataFrame with each feature, its value, and its SHAP contribution
    """
    sv = explainer.shap_values(X_row)

    # Handle list output [class_0_shap, class_1_shap]
    if isinstance(sv, list):
        sv = sv[1]

    # Handle 3D array from LightGBM (samples, features, classes)
    if len(np.array(sv).shape) == 3:
        sv = sv[:, :, 1]

    vals = sv[0] if len(np.array(sv).shape) > 1 else sv

    df = pd.DataFrame({
        'feature': feature_names,
        'value': X_row.iloc[0].values,
        'shap_value': vals
    }).sort_values('shap_value', key=abs, ascending=False)

    return df


REASON_MAP = {
    'credit_utilization':  'High proportion of revolving credit in use',
    'dti_ratio':           'Debt obligations high relative to income',
    'loan_to_income':      'Loan amount high relative to income',
    'has_delinquency':     'Recent delinquency on credit history',
    'has_public_record':   'Public records (bankruptcy/liens) found',
    'dti_x_utilization':   'Combined high debt and credit usage',
    'repayment_stress':    'Indicators of repayment difficulty',
    'inq_last_6mths':      'Multiple recent credit inquiries',
    'high_inquiry':        'Elevated credit-seeking behavior',
    'delinq_2yrs':         'Delinquencies in past 2 years',
    'revol_bal':           'High revolving credit balance',
    'emp_length_num':      'Limited employment history',
    'loan_amnt':           'Large loan amount requested',
    'annual_inc':          'Income level relative to obligations',
    'total_acc':           'Number of credit accounts',
    'pub_rec':             'Public records on file',
    'account_diversity':   'Limited credit account diversity',
    'revol_to_income':     'High revolving balance relative to income',
    'inc_per_account':     'Low income relative to number of accounts',
    'risk_score':          'Combined delinquency, public records, and inquiry risk',
    'loan_x_dti':          'Large loan combined with high debt burden',
    'util_sq':             'Severely elevated credit utilization',
}


def get_adverse_reasons(explanation_df, top_n=4):
    """Get top N adverse action reasons (ECOA-compliant).
    Filters to features with POSITIVE SHAP values (increasing default risk)
    and maps them to human-readable reasons."""
    risk = explanation_df[explanation_df['shap_value'] > 0].head(top_n)
    return [REASON_MAP.get(r['feature'], f"Risk: {r['feature']}")
            for _, r in risk.iterrows()]


def visualize_shap(model, X, feature_names, save_dir='artifacts'):
    """Generate and save 3 SHAP visualizations."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving files
    os.makedirs(save_dir, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Handle LightGBM list output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if len(np.array(shap_values).shape) == 3:
        shap_values = shap_values[:, :, 1]

    # ── PLOT 1: Summary Dot Plot ──────────────────────────────────────────
    # Shows distribution of SHAP values for ALL applicants
    # Each dot = one applicant, color = feature value (red=high, blue=low)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title("SHAP Summary: Feature Impact on Default Risk",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: artifacts/shap_summary.png")

    # ── PLOT 2: Bar Plot (Mean Absolute SHAP) ────────────────────────────
    # Shows average importance of each feature across all applicants
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_type='bar', show=False)
    plt.title("Mean Feature Importance (SHAP)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: artifacts/shap_importance.png")

    # ── PLOT 3: Waterfall Plot for Single Applicant ──────────────────────
    # Shows exactly HOW the model reached its decision for applicant 0
    # Each bar = one feature pushing the score up or down
    plt.figure(figsize=(10, 8))
    shap_exp = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value[1] if isinstance(
            explainer.expected_value, list) else explainer.expected_value,
        data=X.iloc[0].values,
        feature_names=feature_names
    )
    shap.plots.waterfall(shap_exp, show=False)
    plt.title("Applicant #1: Why This Decision?", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_waterfall.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Saved: artifacts/shap_waterfall.png")


if __name__ == "__main__":
    import joblib

    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline'))
    from preprocessing import prepare_data

    model = joblib.load(r"C:\Users\sanja\Desktop\creditlens-ai\artifacts\model.pkl")
    X, y, feature_names = prepare_data(
        r"C:\Users\sanja\Desktop\creditlens-ai\data\raw\credit_data.csv")

    # Single applicant explanation (text)
    X_row = X.iloc[[0]]
    explainer = get_explainer(model)
    explanation = explain_single(explainer, X_row, feature_names)

    print("\n=== SHAP EXPLANATION FOR APPLICANT ===")
    print(explanation.to_string(index=False))

    print("\n=== ADVERSE ACTION REASONS ===")
    reasons = get_adverse_reasons(explanation)
    for i, reason in enumerate(reasons, 1):
        print(f"{i}. {reason}")

    # Generate visual plots
    print("\n=== GENERATING SHAP VISUALIZATIONS ===")
    visualize_shap(model, X, feature_names)