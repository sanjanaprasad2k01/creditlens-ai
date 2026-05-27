"""Preprocessing pipeline for UCI German Credit dataset."""
import pandas as pd
import numpy as np

FEATURE_COLS = [
    'duration', 'credit_amount', 'installment_rate', 'residence_since',
    'age', 'existing_credits', 'dependents',
    'checking_account_num', 'savings_account_num', 'employment_num',
    'credit_history_num', 'purpose_num', 'property_num',
    'housing_num', 'job_num',
    'has_telephone', 'is_foreign_worker', 'has_other_debtors',
    'loan_to_age', 'duration_to_age', 'credit_per_month',
    'is_high_amount', 'long_duration', 'young_borrower'
]

# ── Encoding maps ─────────────────────────────────────────────────────────────
CHECKING_MAP = {'A11': -1, 'A12': 1, 'A13': 2, 'A14': 0}
SAVINGS_MAP  = {'A61': 1, 'A62': 2, 'A63': 3, 'A64': 4, 'A65': 0}
EMPLOY_MAP   = {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}
HISTORY_MAP  = {'A30': 0, 'A31': 1, 'A32': 2, 'A33': 3, 'A34': 4}
PURPOSE_MAP  = {
    'A40': 0, 'A41': 1, 'A42': 2, 'A43': 3, 'A44': 4,
    'A45': 5, 'A46': 6, 'A47': 7, 'A48': 8, 'A49': 9, 'A410': 10
}
PROPERTY_MAP    = {'A121': 3, 'A122': 2, 'A123': 1, 'A124': 0}
HOUSING_MAP     = {'A151': 0, 'A152': 1, 'A153': 2}
JOB_MAP         = {'A171': 0, 'A172': 1, 'A173': 2, 'A174': 3}
OTHER_DEB_MAP   = {'A101': 0, 'A102': 1, 'A103': 2}
INSTALL_MAP     = {'A141': 1, 'A142': 2, 'A143': 0}


def load_and_clean(filepath):
    df = pd.read_csv(r"data/raw/german_credit_clean.csv")

    # Encode categorical columns to numeric
    df['checking_account_num'] = df['checking_account'].map(CHECKING_MAP).fillna(0)
    df['savings_account_num']  = df['savings_account'].map(SAVINGS_MAP).fillna(0)
    df['employment_num']       = df['employment'].map(EMPLOY_MAP).fillna(0)
    df['credit_history_num']   = df['credit_history'].map(HISTORY_MAP).fillna(2)
    df['purpose_num']          = df['purpose'].map(PURPOSE_MAP).fillna(0)
    df['property_num']         = df['property'].map(PROPERTY_MAP).fillna(0)
    df['housing_num']          = df['housing'].map(HOUSING_MAP).fillna(0)
    df['job_num']              = df['job'].map(JOB_MAP).fillna(1)
    df['has_other_debtors']    = df['other_debtors'].map(OTHER_DEB_MAP).fillna(0)
    df['other_installments']   = df['other_installments'].map(INSTALL_MAP).fillna(0)

    # Binary flags
    df['has_telephone']      = (df['telephone'] == 'A192').astype(int)
    df['is_foreign_worker']  = (df['foreign_worker'] == 'A201').astype(int)

    return df


def engineer_features(df):
    df = df.copy()

    # Ratio features
    df['loan_to_age']      = df['credit_amount'] / (df['age'] + 1)
    df['duration_to_age']  = df['duration'] / (df['age'] + 1)
    df['credit_per_month'] = df['credit_amount'] / (df['duration'] + 1)

    # Binary risk flags
    df['is_high_amount']  = (df['credit_amount'] > df['credit_amount'].median()).astype(int)
    df['long_duration']   = (df['duration'] > 24).astype(int)
    df['young_borrower']  = (df['age'] < 25).astype(int)

    return df


def prepare_data(filepath):
    df = load_and_clean(r"data/raw/german_credit_clean.csv")
    df = engineer_features(df)

    X = df[FEATURE_COLS].fillna(0)
    y = df['target']

    print(f"\nDataset: {len(X)} rows | {len(FEATURE_COLS)} features")
    print(f"Default rate: {y.mean():.1%}")

    return X, y, FEATURE_COLS