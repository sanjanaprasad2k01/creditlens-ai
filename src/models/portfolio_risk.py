"""
CreditLens AI — Portfolio Loss Distribution
One-factor Gaussian copula model for correlated defaults.

Implements the Basel III/BIS approach to portfolio credit risk:
- Systematic factor M drives correlated defaults across obligors
- Asset correlation rho = 0.15 (Basel III retail revolving)
- Output: full loss distribution, VaR, ES, economic capital

Interview talking points:
- "I used a one-factor Gaussian copula — same framework as Basel IRB"
- "rho=0.15 is the Basel III supervisory correlation for retail"
- "Economic capital = VaR(99.9%) - EL, the buffer above expected loss"
- "ES is a coherent risk measure — VaR is not, it ignores tail shape"
"""
import numpy as np
from scipy.stats import norm


def monte_carlo_loss(pds, lgds, eads, n_sim=10000, rho=0.15, seed=42):
    """
    One-factor Gaussian copula portfolio loss simulation.

    Each obligor's latent asset return:
        Z_i = sqrt(rho) * M + sqrt(1-rho) * eps_i

    Obligor i defaults when Z_i < Phi^{-1}(PD_i)

    Where:
        M   = systematic (macro) factor — same shock hits all obligors
        eps = idiosyncratic factor — independent per obligor
        rho = asset correlation — how much of risk is systematic

    Parameters
    ----------
    pds  : array-like, shape (n,)  — model PDs per obligor
    lgds : array-like, shape (n,)  — LGD per obligor
    eads : array-like, shape (n,)  — EAD per obligor
    n_sim: int                     — number of Monte Carlo scenarios
    rho  : float                   — asset correlation (0.15 = Basel III retail)
    seed : int                     — random seed for reproducibility

    Returns
    -------
    dict with full loss distribution and risk metrics
    """
    np.random.seed(seed)

    pds  = np.clip(np.array(pds,  dtype=float), 1e-6, 1-1e-6)
    lgds = np.array(lgds, dtype=float)
    eads = np.array(eads, dtype=float)
    n    = len(pds)

    # Default thresholds — Phi^{-1}(PD_i)
    thresholds = norm.ppf(pds)       # shape: (n,)

    # Systematic factor — one draw per simulation
    M = np.random.standard_normal(n_sim)          # shape: (n_sim,)

    # Idiosyncratic shocks — one draw per obligor per simulation
    eps = np.random.standard_normal((n_sim, n))   # shape: (n_sim, n)

    # Asset returns per obligor per simulation
    # Broadcasting: M[:,None] is (n_sim,1), eps is (n_sim,n)
    Z = np.sqrt(rho) * M[:, None] + np.sqrt(1 - rho) * eps  # (n_sim, n)

    # Default indicator: 1 if asset return < threshold
    defaults = (Z < thresholds[None, :]).astype(float)       # (n_sim, n)

    # Loss per obligor (LGD × EAD)
    loss_per = lgds * eads                                    # (n,)

    # Portfolio loss per simulation
    losses = defaults @ loss_per                              # (n_sim,)

    # Risk metrics
    el              = float(np.mean(losses))
    var_95          = float(np.percentile(losses, 95))
    var_99          = float(np.percentile(losses, 99))
    var_999         = float(np.percentile(losses, 99.9))
    tail_99         = losses[losses >= var_99]
    es_99           = float(tail_99.mean()) if len(tail_99) > 0 else var_99
    economic_capital= max(0.0, float(var_999 - el))

    # Default rate distribution
    default_rates   = defaults.mean(axis=1)
    mean_dr         = float(default_rates.mean())
    p99_dr          = float(np.percentile(default_rates, 99))

    return {
        'losses':            losses,
        'el':                el,
        'var_95':            var_95,
        'var_99':            var_99,
        'var_999':           var_999,
        'es_99':             es_99,
        'economic_capital':  economic_capital,
        'mean_default_rate': mean_dr,
        'p99_default_rate':  p99_dr,
        'n_obligors':        n,
        'n_sim':             n_sim,
        'rho':               rho,
        'total_ead':         float(eads.sum()),
        'el_rate':           el / float(eads.sum()) if eads.sum() > 0 else 0,
    }


def pd_term_structure(pd_12m, horizons=(3, 6, 12, 18, 24, 36, 48, 60)):
    """
    Convert 12-month PD to a full term structure using constant hazard rate.

    Assumes a constant monthly hazard rate h derived from PD_12m:
        h = -ln(1 - PD_12m) / 12

    Cumulative PD at horizon t months:
        PD(t) = 1 - exp(-h * t)

    Survival probability:
        S(t)  = exp(-h * t) = 1 - PD(t)

    This is the standard Basel III approach for deriving lifetime PD
    from a 12-month PD estimate, consistent with IFRS 9 §B5.5.42
    guidance on PD estimation.

    Parameters
    ----------
    pd_12m   : float — 12-month model PD (0 to 1)
    horizons : tuple — time horizons in months

    Returns
    -------
    dict with term structure arrays
    """
    pd_12m = float(np.clip(pd_12m, 1e-6, 1-1e-6))

    # Monthly hazard rate from 12-month PD
    h = -np.log(1.0 - pd_12m) / 12.0

    horizons    = list(horizons)
    pd_curve    = [1.0 - np.exp(-h * t) for t in horizons]
    surv_curve  = [np.exp(-h * t)        for t in horizons]
    # Marginal default probability in each period (conditional on surviving to start)
    marginal    = [pd_curve[0]] + [
        (pd_curve[i] - pd_curve[i-1]) / surv_curve[i-1]
        for i in range(1, len(horizons))
    ]

    return {
        'pd_12m':             pd_12m,
        'hazard_monthly':     h,
        'hazard_annual':      h * 12,
        'horizons_months':    horizons,
        'cumulative_pd':      pd_curve,
        'survival':           surv_curve,
        'marginal_pd':        marginal,
        'ifrs9_lifetime_pd':  pd_curve[horizons.index(60)] if 60 in horizons else pd_curve[-1],
    }