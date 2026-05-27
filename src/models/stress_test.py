"""
Basel III / CCAR stress testing for CreditLens AI.
Four scenarios: Base, Mild Recession, Severe Recession, GFC-equivalent.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_pipeline'))

SCENARIOS = {
    'Base Case':             {'pd_multiplier': 1.0, 'lgd': 0.45, 'color': '#00e676'},
    'Mild Recession (1.5x)': {'pd_multiplier': 1.5, 'lgd': 0.50, 'color': '#ffeb3b'},
    'Severe Recession (2x)': {'pd_multiplier': 2.0, 'lgd': 0.55, 'color': '#ff9800'},
    'GFC Equivalent (2.5x)': {'pd_multiplier': 2.5, 'lgd': 0.60, 'color': '#f44336'},
}


def run_stress_test(model, X, loan_amounts, feature_names, save_dir='artifacts'):
    """Run full stress test suite and generate report + charts."""
    os.makedirs(save_dir, exist_ok=True)

    base_probs = model.predict_proba(X[feature_names])[:, 1]
    ead_total  = loan_amounts.sum()
    results    = []

    print("\n=== STRESS TEST RESULTS ===")
    print(f"Portfolio: {len(X):,} loans | Total EAD: DM {ead_total:,.0f}\n")
    print(f"{'Scenario':<30} {'Avg PD':>8} {'Port. EL':>14} "
          f"{'Capital Req':>14} {'Stage 3 %':>10}")
    print("-" * 80)

    for scenario, params in SCENARIOS.items():
        stressed_pd  = np.clip(base_probs * params['pd_multiplier'], 0, 1)
        lgd          = params['lgd']
        el_per_loan  = stressed_pd * lgd * loan_amounts
        portfolio_el = el_per_loan.sum()

        # Basel III IRB simplified RWA
        exp_term = np.exp(-35 * stressed_pd)
        r   = 0.03 * (1 - exp_term) / (1 - np.exp(-35) + 1e-10) + \
              0.16 * (1 - (1 - exp_term) / (1 - np.exp(-35) + 1e-10))
        k   = lgd * np.maximum(
            0, (2.326 * np.sqrt(r) - r) / (np.sqrt(1 - r) + 1e-10) \
               - stressed_pd * lgd)
        rwa         = (k * 12.5 * loan_amounts).sum()
        capital_req = rwa * 0.08
        stage3_pct  = (stressed_pd >= 0.50).mean() * 100

        results.append({
            'Scenario':     scenario,
            'PD Multiplier':params['pd_multiplier'],
            'Avg PD':       stressed_pd.mean(),
            'Portfolio EL': portfolio_el,
            'RWA':          rwa,
            'Capital Req':  capital_req,
            'Stage 3 %':    stage3_pct,
            'color':        params['color']
        })

        print(f"{scenario:<30} {stressed_pd.mean():>7.1%} "
              f"DM {portfolio_el:>12,.0f} DM {capital_req:>12,.0f} "
              f"{stage3_pct:>9.1f}%")

    # ── Charts ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0a1628')

    labels = [r['Scenario'].split('(')[0].strip() for r in results]
    colors = [r['color'] for r in results]

    for ax, key, title, prefix in [
        (axes[0], 'Portfolio EL', 'Portfolio Expected Loss',       'DM '),
        (axes[1], 'Capital Req',  'Regulatory Capital (Basel III)','DM '),
        (axes[2], 'Stage 3 %',   'IFRS 9 Stage 3 Migration (%)',  ''),
    ]:
        ax.set_facecolor('#0f1f3d')
        vals = [r[key] for r in results]
        bars = ax.bar(range(len(results)), vals,
                      color=colors, alpha=0.85, edgecolor='none')
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(labels, rotation=15, ha='right',
                           fontsize=7.5, color='#a0b4c8')
        ax.set_title(title, color='#ffffff', fontsize=9, fontweight='bold')
        ax.tick_params(colors='#a0b4c8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e3a5f')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02,
                    f'{prefix}{v:,.0f}',
                    ha='center', va='bottom', fontsize=6.5,
                    color='#e2e8f0', fontfamily='monospace')

    fig.suptitle('CreditLens AI — Basel III / CCAR Stress Test',
                 color='#ffffff', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'stress_test.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0a1628')
    plt.close()
    print(f"\n[OK] Chart saved to {path}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    from preprocessing_german import prepare_data

    data         = joblib.load(
        r"C:\Users\sanja\Desktop\creditlens-ai\artifacts\model_german.pkl")
    model        = data['model'] if isinstance(data, dict) else data
    X, y, feats  = prepare_data(
        r"C:\Users\sanja\Desktop\creditlens-ai\data\raw\german_credit_clean.csv")
    df_raw       = pd.read_csv(
        r"C:\Users\sanja\Desktop\creditlens-ai\data\raw\german_credit_clean.csv")
    loan_amounts = df_raw['credit_amount'].values

    results = run_stress_test(model, X, loan_amounts, feats)
    print("\n=== SUMMARY ===")
    print(results[['Scenario','Avg PD','Portfolio EL',
                   'Capital Req','Stage 3 %']].to_string(index=False))