from enum import Enum

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder


def load_and_preprocess_credit(path: str = "./datasets/credit") -> pd.DataFrame:
    gender_map = {"'male single'": "male", "'female div/dep/mar'": "female", "'male mar/wid'": "male",
                  "'male div/sep'": "male"}
    status_map = {"'male single'": "single", "'female div/dep/mar'": "married/wid/sep",
                  "'male mar/wid'": "married/wid/sep", "'male div/sep'": "married/wid/sep"}

    dt = pd.read_csv(path, header=0, skipinitialspace=True, na_values='?')
    dt.dropna(inplace=True)
    dt.reset_index(drop=True, inplace=True)
    categorical_cols = ['checking_status', 'credit_history', 'purpose',
                        'savings_status', 'employment', 'personal_status',
                        'other_parties', 'property_magnitude', 'other_payment_plans',
                        'housing', 'job', 'own_telephone', 'foreign_worker']
    numerical_cols = ['duration', 'credit_amount', 'installment_commitment',
                    'residence_since', 'age', 'existing_credits', 'num_dependents']

    dt['class'] = dt['class'].map({'bad': 0, 'good': 1})

    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(dt[categorical_cols])
    dt_encoded = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

    dt[numerical_cols] = dt[numerical_cols].astype(int)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dt[numerical_cols])
    dt_scaled = pd.DataFrame(scaled, columns=numerical_cols)

    dt_preprocessed = pd.concat([dt_encoded, dt_scaled, dt["class"]], axis=1, ignore_index=True)
    return dt_preprocessed
