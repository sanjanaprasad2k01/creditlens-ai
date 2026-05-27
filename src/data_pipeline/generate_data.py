"""Generate synthetic credit data for CreditLens AI."""
# ↑ This is a docstring — describes what the file does

import pandas as pd      # pandas: library for working with tables (DataFrames)
import numpy as np       # numpy: library for math operations and random numbers
import os                # os: library for file/folder operations

def generate():          # "def" defines a function. This function is called "generate"
    np.random.seed(42)   # Makes random numbers reproducible (same data every time you run)
    n = 5000             # We'll create 5,000 loan applications

    # Create each column of data using random number generators
    # np.random.lognormal(11, 0.5, n) = 5000 random numbers from a lognormal distribution
    # Lognormal is right-skewed — most people earn $50-80K, few earn $300K+
    annual_inc = np.random.lognormal(11, 0.5, n).round(0)

    # Uniform distribution: every value between 1000 and 40000 is equally likely
    loan_amnt = np.random.uniform(1000, 40000, n).round(0)

    # DTI (Debt-to-Income): ranges from 0% to 45%
    dti = np.random.uniform(0, 45, n).round(1)

    # Credit utilization: normally distributed around 50%, clipped to 0-100%
    # np.clip ensures no values go below 0 or above 100
    revol_util = np.clip(np.random.normal(50, 25, n), 0, 100).round(1)

    revol_bal = np.random.uniform(0, 50000, n).round(0)
    total_acc = np.random.randint(2, 40, n)  # Random integers from 2 to 39

    # np.random.choice picks randomly from a list
    # More 0s than 1s/2s/3s = most people have no delinquencies (realistic)
    delinq_2yrs = np.random.choice([0,0,0,0,0,1,1,2,3], n)
    pub_rec = np.random.choice([0,0,0,0,0,0,1,1,2], n)
    inq_last_6mths = np.random.choice([0,0,0,1,1,2,2,3,4,5], n)

    emp_length = np.random.choice(
        ['< 1 year','1 year','2 years','3 years','4 years',
         '5 years','6 years','7 years','8 years','9 years','10+ years'], n)

    # ============================================================
    # THIS IS THE CLEVER PART: Generate realistic default labels
    # Instead of random, we make defaults CORRELATED with risk factors
    # ============================================================
    risk_score = (
        (dti / 45) * 0.25 +                          # Higher DTI = more risk
        (revol_util / 100) * 0.20 +                   # Higher utilization = more risk
        (loan_amnt / annual_inc) * 0.20 +             # Bigger loan vs income = more risk
        (delinq_2yrs > 0).astype(float) * 0.15 +     # Any delinquency = risk
        (pub_rec > 0).astype(float) * 0.10 +          # Any public records = risk
        (inq_last_6mths / 5) * 0.10                   # More inquiries = risk
    )
    # Sigmoid function converts risk score to probability between 0 and 1
    # This is the same function used in Logistic Regression
    default_prob = 1 / (1 + np.exp(-5 * (risk_score - 0.45)))

    # np.random.binomial(1, p) = coin flip with probability p
    # If default_prob = 0.7, there's a 70% chance is_default = 1
    is_default = np.random.binomial(1, default_prob)

    # Assign loan status based on default flag
    loan_status = np.where(is_default,
        np.random.choice(['Charged Off', 'Default', 'Late (31-120 days)'], n),
        'Fully Paid')

    # pd.DataFrame creates a table (like an Excel spreadsheet in memory)
    df = pd.DataFrame({
        'loan_amnt': loan_amnt, 'annual_inc': annual_inc, 'dti': dti,
        'revol_util': revol_util, 'revol_bal': revol_bal, 'total_acc': total_acc,
        'delinq_2yrs': delinq_2yrs, 'pub_rec': pub_rec, 'emp_length': emp_length,
        'inq_last_6mths': inq_last_6mths, 'loan_status': loan_status})

    # os.makedirs creates the folder if it doesn't exist
    # exist_ok=True means "don't error if folder already exists"
    os.makedirs('data/raw', exist_ok=True)

    # .to_csv saves the DataFrame as a CSV file
    # index=False means "don't add a row number column"
    df.to_csv('data/raw/credit_data.csv', index=False)

    print(f"[OK] Generated {n} records -> data/raw/credit_data.csv")
    print(f"[OK] Default rate: {is_default.mean():.1%}")
    return df

# This block runs ONLY when you execute the file directly
# (not when another file imports it)
if __name__ == '__main__':
    generate()