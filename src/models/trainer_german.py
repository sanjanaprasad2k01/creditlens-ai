"""
Trainer for German Credit dataset.
Includes optimal threshold selection, full discrimination metrics,
and confusion matrix analysis at each threshold.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    precision_recall_curve, confusion_matrix,
    average_precision_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline'))


def find_optimal_threshold(y_true, y_prob):
    """
    Find optimal classification threshold using three methods:

    1. Youden's J statistic  = Sensitivity + Specificity - 1
       Industry standard for credit scorecard cutoff selection.
       Maximises the sum of TPR and TNR simultaneously.

    2. F1 Score maximisation
       Balances precision and recall — good for imbalanced datasets.

    3. Precision-Recall breakeven
       Where precision = recall — useful for business cost symmetry.

    Returns all three thresholds so the user can choose based on
    their specific business objective (minimise false approvals vs
    minimise false denials).
    """
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)

    # ── Method 1: Youden's J ─────────────────────────────────────────────
    j_scores    = tpr - fpr
    j_idx       = np.argmax(j_scores)
    youden_thresh = float(thresholds_roc[j_idx])
    youden_tpr    = float(tpr[j_idx])
    youden_fpr    = float(fpr[j_idx])
    youden_j      = float(j_scores[j_idx])

    # ── Method 2: F1 maximisation ────────────────────────────────────────
    prec, rec, thresholds_pr = precision_recall_curve(y_true, y_prob)
    f1_scores  = 2 * prec * rec / (prec + rec + 1e-10)
    f1_idx     = np.argmax(f1_scores[:-1])
    f1_thresh  = float(thresholds_pr[f1_idx])
    f1_best    = float(f1_scores[f1_idx])

    # ── Method 3: Precision-Recall breakeven ────────────────────────────
    diff       = np.abs(prec - rec)
    pr_idx     = np.argmin(diff[:-1])
    pr_thresh  = float(thresholds_pr[pr_idx])

    return {
        'youden':          youden_thresh,
        'youden_tpr':      youden_tpr,
        'youden_fpr':      youden_fpr,
        'youden_j':        youden_j,
        'f1':              f1_thresh,
        'f1_score':        f1_best,
        'pr_breakeven':    pr_thresh,
    }


def evaluate_at_threshold(y_true, y_prob, threshold, label=""):
    """Full evaluation at a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / TPR
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    precision    = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1           = f1_score(y_true, y_pred)
    accuracy     = (tp + tn) / len(y_true)
    fpr          = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"\n  Threshold = {threshold:.4f} {label}")
    print(f"  {'─'*50}")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Sensitivity (TPR): {sensitivity:.4f}  ← catches {sensitivity:.0%} of defaults")
    print(f"  Specificity (TNR): {specificity:.4f}  ← approves {specificity:.0%} of good payers")
    print(f"  Precision:         {precision:.4f}")
    print(f"  F1 Score:          {f1:.4f}")
    print(f"  False Positive Rate: {fpr:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>15} Predicted: No Default  Predicted: Default")
    print(f"  Actual: No Default    TN={tn:>5}               FP={fp:>5}")
    print(f"  Actual: Default       FN={fn:>5}               TP={tp:>5}")

    return {
        'threshold':   threshold,
        'accuracy':    accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision':   precision,
        'f1':          f1,
        'fpr':         fpr,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def plot_discrimination_charts(y_true, y_prob, thresholds_dict,
                                save_dir='artifacts'):
    """Generate ROC, PR, and threshold analysis charts."""
    os.makedirs(save_dir, exist_ok=True)

    fpr_arr, tpr_arr, thresh_arr = roc_curve(y_true, y_prob)
    prec_arr, rec_arr, pr_thresh = precision_recall_curve(y_true, y_prob)
    auc_val  = roc_auc_score(y_true, y_prob)
    ap_val   = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0a1628')
    colors = {'youden': '#00d4ff', 'f1': '#ffeb3b', 'pr_breakeven': '#ff9800'}

    # ── Chart 1: ROC Curve ───────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#0f1f3d')
    ax.plot(fpr_arr, tpr_arr, color='#00e676', linewidth=2.5,
            label=f'LightGBM (AUC = {auc_val:.4f})')
    ax.plot([0,1], [0,1], '--', color='#4a6080', linewidth=1.2, label='Random (AUC = 0.50)')

    # Plot threshold points
    for method, thresh in [
        ('youden', thresholds_dict['youden']),
        ('f1',     thresholds_dict['f1']),
    ]:
        idx = np.argmin(np.abs(thresh_arr - thresh))
        ax.scatter(fpr_arr[idx], tpr_arr[idx], s=120,
                   color=colors[method], zorder=5,
                   label=f'{method.title()} threshold ({thresh:.3f})')

    ax.set_xlabel('False Positive Rate (1 - Specificity)',
                  color='#7fa8d4', fontsize=9)
    ax.set_ylabel('True Positive Rate (Sensitivity)', color='#7fa8d4', fontsize=9)
    ax.set_title('ROC Curve', color='#ffffff', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#a0b4c8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax.legend(facecolor='#0a1628', edgecolor='#1e3a5f',
              labelcolor='#e2e8f0', fontsize=8)

    # ── Chart 2: Precision-Recall Curve ─────────────────────────────────
    ax = axes[1]
    ax.set_facecolor('#0f1f3d')
    ax.plot(rec_arr, prec_arr, color='#00d4ff', linewidth=2.5,
            label=f'LightGBM (AP = {ap_val:.4f})')
    baseline = y_true.mean()
    ax.axhline(baseline, color='#4a6080', linewidth=1.2, linestyle='--',
               label=f'Baseline (default rate = {baseline:.2f})')

    for method, thresh in [
        ('f1', thresholds_dict['f1']),
        ('pr_breakeven', thresholds_dict['pr_breakeven']),
    ]:
        idx = np.argmin(np.abs(pr_thresh - thresh))
        ax.scatter(rec_arr[idx], prec_arr[idx], s=120,
                   color=colors[method], zorder=5,
                   label=f'{method.replace("_"," ").title()} ({thresh:.3f})')

    ax.set_xlabel('Recall (Sensitivity)', color='#7fa8d4', fontsize=9)
    ax.set_ylabel('Precision', color='#7fa8d4', fontsize=9)
    ax.set_title('Precision-Recall Curve', color='#ffffff',
                 fontsize=11, fontweight='bold')
    ax.tick_params(colors='#a0b4c8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax.legend(facecolor='#0a1628', edgecolor='#1e3a5f',
              labelcolor='#e2e8f0', fontsize=8)

    # ── Chart 3: Threshold Analysis ──────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor('#0f1f3d')

    thresh_range = np.linspace(0.1, 0.9, 100)
    tpr_list, tnr_list, f1_list = [], [], []

    for t in thresh_range:
        y_pred = (y_prob >= t).astype(int)
        cm     = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        tnr_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        f1_list.append(f1_score(y_true, y_pred))

    ax.plot(thresh_range, tpr_list, color='#f44336', linewidth=2,
            label='Sensitivity (catches defaults)')
    ax.plot(thresh_range, tnr_list, color='#00e676', linewidth=2,
            label='Specificity (approves good payers)')
    ax.plot(thresh_range, f1_list,  color='#ffeb3b', linewidth=2,
            label='F1 Score')

    # Vertical lines at key thresholds
    for method, thresh, c in [
        ('Youden',  thresholds_dict['youden'], '#00d4ff'),
        ('F1 Max',  thresholds_dict['f1'],     '#ff9800'),
    ]:
        ax.axvline(thresh, color=c, linewidth=1.5, linestyle='--',
                   label=f'{method}: {thresh:.3f}')

    ax.set_xlabel('Classification Threshold', color='#7fa8d4', fontsize=9)
    ax.set_ylabel('Score', color='#7fa8d4', fontsize=9)
    ax.set_title('Sensitivity vs Specificity vs F1 by Threshold',
                 color='#ffffff', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#a0b4c8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
    ax.legend(facecolor='#0a1628', edgecolor='#1e3a5f',
              labelcolor='#e2e8f0', fontsize=7.5)
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0, 1)

    fig.suptitle('CreditLens AI — Discrimination Analysis',
                 color='#ffffff', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'discrimination_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a1628')
    plt.close()
    print(f"\n[OK] Discrimination chart saved to {path}")


def train_model(X, y, save_path='artifacts/model_german.pkl'):
    """
    Train LightGBM with full discrimination analysis and
    optimal threshold selection.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scale = len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)

    # ── Benchmarks ───────────────────────────────────────────────────────
    print("\n=== BENCHMARK COMPARISON ===")
    dummy = DummyClassifier(strategy='uniform', random_state=42)
    dummy.fit(X_train, y_train)
    dummy_auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])
    print(f"Random baseline:     AUC {dummy_auc:.4f} | Gini {2*dummy_auc-1:.4f}")

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_auc  = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
    print(f"Logistic Regression: AUC {lr_auc:.4f} | Gini {2*lr_auc-1:.4f}")

    # ── LightGBM ─────────────────────────────────────────────────────────
    model = lgb.LGBMClassifier(
        max_depth         = 5,
        learning_rate     = 0.01,
        n_estimators      = 1000,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale,
        num_leaves        = 25,
        min_child_samples = 15,
        reg_alpha         = 0.2,
        reg_lambda        = 0.2,
        random_state      = 42,
        verbose           = -1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"\nLightGBM CV AUC:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    gini   = 2 * auc - 1

    # KS Statistic
    defaulters     = y_prob[y_test == 1]
    non_defaulters = y_prob[y_test == 0]
    ks_stat, _     = ks_2samp(defaulters, non_defaulters)
    lift           = (auc - lr_auc) / lr_auc * 100

    print(f"\n=== LIGHTGBM RESULTS ===")
    print(f"AUC-ROC:             {auc:.4f}")
    print(f"Gini Coefficient:    {gini:.4f}  (industry min: 0.30)")
    print(f"KS Statistic:        {ks_stat:.4f}  (industry min: 0.20)")
    print(f"Relative lift:       +{lift:.1f}% over logistic regression")

    # ── Score distribution analysis ───────────────────────────────────────
    print(f"\n=== SCORE DISTRIBUTION ===")
    print(f"Min PD:     {y_prob.min():.4f}")
    print(f"Max PD:     {y_prob.max():.4f}")
    print(f"Mean PD:    {y_prob.mean():.4f}")
    print(f"Median PD:  {np.median(y_prob):.4f}")
    print(f"25th pct:   {np.percentile(y_prob, 25):.4f}")
    print(f"75th pct:   {np.percentile(y_prob, 75):.4f}")

    # Separation between good and bad
    mean_default = y_prob[y_test == 1].mean()
    mean_good    = y_prob[y_test == 0].mean()
    separation   = mean_default - mean_good
    print(f"\nMean PD — Defaults:     {mean_default:.4f}")
    print(f"Mean PD — Non-Defaults: {mean_good:.4f}")
    print(f"Separation:             {separation:.4f}  "
          f"{'✅ Good separation' if separation > 0.15 else '⚠️ Limited separation'}")

    # ── Optimal threshold selection ───────────────────────────────────────
    print(f"\n=== OPTIMAL THRESHOLD ANALYSIS ===")
    thresholds = find_optimal_threshold(y_test, y_prob)

    print(f"\n  Method 1 — Youden's J Statistic (industry standard):")
    print(f"    Threshold: {thresholds['youden']:.4f}")
    print(f"    TPR:       {thresholds['youden_tpr']:.4f}")
    print(f"    FPR:       {thresholds['youden_fpr']:.4f}")
    print(f"    J Score:   {thresholds['youden_j']:.4f}")

    print(f"\n  Method 2 — F1 Score Maximisation:")
    print(f"    Threshold: {thresholds['f1']:.4f}")
    print(f"    F1 Score:  {thresholds['f1_score']:.4f}")

    print(f"\n  Method 3 — Precision-Recall Breakeven:")
    print(f"    Threshold: {thresholds['pr_breakeven']:.4f}")

    # ── Evaluate at three thresholds ─────────────────────────────────────
    print(f"\n=== PERFORMANCE AT EACH THRESHOLD ===")

    results = {}
    results['youden'] = evaluate_at_threshold(
        y_test, y_prob, thresholds['youden'],
        "← Youden's J (maximises sensitivity + specificity)")
    results['f1'] = evaluate_at_threshold(
        y_test, y_prob, thresholds['f1'],
        "← F1 maximisation (balances precision + recall)")
    results['default_50'] = evaluate_at_threshold(
        y_test, y_prob, 0.50,
        "← Default 0.50 threshold (for comparison)")

    # ── Business recommendation ───────────────────────────────────────────
    print(f"\n=== BUSINESS THRESHOLD RECOMMENDATION ===")
    print(f"""
  Use Youden's J threshold ({thresholds['youden']:.4f}) when:
  → You want the best balance between catching defaults and
    approving good payers. Standard choice for retail credit.

  Use F1 threshold ({thresholds['f1']:.4f}) when:
  → Default costs and false-approval costs are roughly equal.
    Good for symmetric risk appetite.

  Use a lower threshold (e.g. 0.35) when:
  → False negatives (missed defaults) are very costly.
    Conservative risk appetite — stricter approval policy.

  Use a higher threshold (e.g. 0.65) when:
  → Maximising approvals is the priority.
    Higher risk tolerance — growth-oriented lending.

  RECOMMENDED for CreditLens AI:
  → Youden's J = {thresholds['youden']:.4f}
  → This gives {results['youden']['sensitivity']:.0%} default detection rate
    with {results['youden']['specificity']:.0%} good-payer approval rate.
    """)

    # ── Three-band threshold update ───────────────────────────────────────
    youden  = thresholds['youden']
    approve = round(youden * 0.75, 2)   # Clear approve zone
    review  = round(youden, 2)           # Youden's J as the main cutoff
    decline = round(youden * 1.25, 2)   # Clear decline zone

    print(f"\n=== UPDATED DECISION BANDS ===")
    print(f"  APPROVE:          PD < {approve:.2f}")
    print(f"  APPROVE w/ COND:  {approve:.2f} ≤ PD < {review:.2f}")
    print(f"  MANUAL REVIEW:    {review:.2f} ≤ PD < {decline:.2f}")
    print(f"  DECLINE:          PD ≥ {decline:.2f}")
    print(f"\n  (Replace your current 0.35/0.50/0.65 bands with these)")

    # Evaluate at new bands
    print(f"\n  Performance at new APPROVE/DECLINE split ({review:.2f}):")
    evaluate_at_threshold(y_test, y_prob, review, "← Youden's optimal cutoff")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_discrimination_charts(y_test, y_prob, thresholds)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({
        'model':   model,
        'scaler':  scaler,
        'metrics': {
            'auc':        auc,
            'gini':       gini,
            'ks':         ks_stat,
            'lr_auc':     lr_auc,
            'lift':       lift,
            'thresholds': thresholds,
            'approve':    approve,
            'review':     review,
            'decline':    decline,
            'separation': separation,
        }
    }, save_path)
    print(f"\n[OK] Model saved to {save_path}")
    print(f"[OK] Thresholds saved: approve={approve}, review={review}, decline={decline}")

    return model, X_test, y_test, auc, thresholds


if __name__ == "__main__":
    from preprocessing_german import prepare_data

    X, y, feats = prepare_data(
        r"C:\Users\sanja\Desktop\creditlens-ai\data\raw\german_credit_clean.csv")
    print(f"Dataset: {len(X)} rows | {len(feats)} features | Default rate: {y.mean():.1%}")

    model, X_test, y_test, auc, thresholds = train_model(X, y)

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"AUC-ROC:              {auc:.4f}")
    print(f"Gini:                 {2*auc-1:.4f}")
    print(f"Optimal threshold:    {thresholds['youden']:.4f} (Youden's J)")
    print(f"{'='*60}")
    print(f"\nNext step: Update your decision thresholds in app/main.py")
    print(f"using the UPDATED DECISION BANDS printed above.")