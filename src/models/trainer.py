""""Model training, evaluation, and benchmarking."""
import lightgbm as lgb
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp


def train_model(X, y, save_path='artifacts/model.pkl'):
    """Train LightGBM with full benchmarking and credit risk metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scale = len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1)

    # ── Benchmarks ────────────────────────────────────────────────────────
    print("\n=== BENCHMARK COMPARISON ===")

    dummy = DummyClassifier(strategy='uniform', random_state=42)
    dummy.fit(X_train, y_train)
    dummy_auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:, 1])
    print(f"Random baseline AUC:       {dummy_auc:.4f} | Gini: {2*dummy_auc-1:.4f}")

    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    lr             = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    lr_auc  = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
    lr_gini = 2 * lr_auc - 1
    print(f"Logistic regression AUC:   {lr_auc:.4f} | Gini: {lr_gini:.4f}")

    # ── LightGBM ──────────────────────────────────────────────────────────
    model = lgb.LGBMClassifier(
        max_depth=4, learning_rate=0.01, n_estimators=1000,
        subsample=0.7, colsample_bytree=0.7, scale_pos_weight=scale,
        num_leaves=15, min_child_samples=30, reg_alpha=0.5, reg_lambda=0.5,
        random_state=42, verbose=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"\nLightGBM CV AUC:           {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    gini   = 2 * auc - 1

    defaulters     = y_prob[y_test == 1]
    non_defaulters = y_prob[y_test == 0]
    ks_stat, _     = ks_2samp(defaulters, non_defaulters)
    lift           = (auc - lr_auc) / lr_auc * 100

    print(f"\n=== LIGHTGBM RESULTS ===")
    print(f"AUC-ROC:                   {auc:.4f}")
    print(f"Gini Coefficient:          {gini:.4f}  (industry min: 0.30)")
    print(f"KS Statistic:              {ks_stat:.4f}  (industry min: 0.20)")
    print(f"Relative lift over LR:     {lift:.1f}%")
    print(f"\nBenchmark Summary:")
    print(f"  Random     → AUC: {dummy_auc:.4f} | Gini: {2*dummy_auc-1:.4f}")
    print(f"  Log. Reg.  → AUC: {lr_auc:.4f} | Gini: {lr_gini:.4f}")
    print(f"  LightGBM   → AUC: {auc:.4f} | Gini: {gini:.4f}  ← your model")
    print(f"\n{classification_report(y_test, model.predict(X_test), target_names=['Non-Default','Default'])}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({'model': model, 'scaler': scaler,
                 'metrics': {'auc': auc, 'gini': gini, 'ks': ks_stat,
                             'lr_auc': lr_auc, 'lift': lift}},
                save_path)
    print(f"[OK] Model saved to {save_path}")

    return model, X_test, y_test, auc


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline'))
    from preprocessing import prepare_data
    X, y, feats = prepare_data(
        r"C:\Users\sanja\Desktop\creditlens-ai\data\raw\credit_data.csv")
    print(f"Features: {len(feats)}, Rows: {len(X)}")
    model, X_test, y_test, auc = train_model(X, y)