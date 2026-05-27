"""Data loading, cleaning, and feature engineering."""
import pandas as pd
import numpy as np

# List of all 17 features the model will use
# This is defined at the top so every file can import it
FEATURE_COLS = [
    'loan_amnt', 'annual_inc', 'dti_ratio', 'credit_utilization',
    'revol_bal', 'total_acc', 'delinq_2yrs', 'pub_rec',
    'emp_length_num', 'inq_last_6mths', 'loan_to_income',
    'has_delinquency', 'has_public_record', 'high_inquiry',
    'dti_x_utilization', 'repayment_stress', 'account_diversity'
]

def load_and_clean(filepath):
    """Load CSV and perform basic cleaning."""
    df = pd.read_csv(r"C:\Users\sanja\Desktop\creditlens-ai\data\raw\credit_data.csv")         # Read the CSV into a DataFrame

    # Create the TARGET variable (what we're predicting)
    # isin() checks if loan_status is in the list of default statuses
    # .astype(int) converts True/False to 1/0
    default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)']
    df['target'] = df['loan_status'].isin(default_statuses).astype(int)
    #  ↑ target=1 means DEFAULT, target=0 means FULLY PAID

    # Convert text employment length to a number
    # str.extract(r'(\d+)') pulls the first number from the text
    # "5 years" → 5, "< 1 year" → 1, "10+ years" → 10
    df['emp_length_num'] = df['emp_length'].str.extract(r'(\d+)').astype(float).fillna(0)


    # Fill missing values with the median (middle value)
    # Why median not mean? Median is resistant to outliers
    # If most people make $60K but one makes $10M, mean=$200K (misleading), median=$60K (accurate)
    df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())
    df['annual_inc'] = df['annual_inc'].fillna(df['annual_inc'].median())

    return df


def engineer_features(df):
    """Create risk-relevant features with business justification."""
    df = df.copy()  # Don't modify the original DataFrame

    # FEATURE: Loan-to-Income Ratio
    # Business logic: Borrowing $20K on $50K income (0.40) is riskier
    #                 than borrowing $20K on $200K income (0.10)
    # +1 in denominator prevents division by zero
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)


  # FEATURE: DTI as decimal (0-1 range instead of 0-100)
    # Models work better with consistent scales
    df['dti_ratio'] = df['dti'] / 100


  # FEATURE: Credit utilization as decimal
    df['credit_utilization'] = df['revol_util'] / 100

    # FEATURE: Binary flags (1 or 0)
    # Simpler for the model to learn "has ANY delinquency" vs "has 0,1,2,3..."
    df['has_delinquency'] = (df['delinq_2yrs'] > 0).astype(int)
    df['has_public_record'] = (df['pub_rec'] > 0).astype(int)
    df['high_inquiry'] = (df['inq_last_6mths'] > 2).astype(int)

    # FEATURE: Interaction terms
    # High DTI ALONE is risky. High utilization ALONE is risky.
    # Both together? MUCH riskier. The interaction captures this compounding effect.
    df['dti_x_utilization'] = df['dti_ratio'] * df['credit_utilization']
    df['repayment_stress'] = df['loan_to_income'] * df['dti_ratio']

    # FEATURE: Account diversity
    # np.log1p = log(1+x), compresses large numbers
    # Having 5 accounts vs 30 accounts matters, but 30 vs 35 doesn't much
    df['account_diversity'] = np.log1p(df['total_acc'])

    return df


def prepare_data(filepath):
    """Full pipeline: load → clean → engineer → return X, y, feature names."""
    df = load_and_clean(r"C:\Users\sanja\Desktop\creditlens-ai\data\raw\credit_data.csv")
    df = engineer_features(df)

    # X = features (inputs to the model) — a table with 17 columns
    # y = target (what we're predicting) — a single column of 0s and 1s
    # .fillna(0) replaces any remaining missing values with 0
    X = df[FEATURE_COLS].fillna(0)
    y = df['target']

    return X, y, FEATURE_COLS
