import pandas as pd

from preprocess.preprocessing_adult import Mode
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_bank(mode: Mode = Mode.TRAIN, path: str = "./datasets/bank") -> pd.DataFrame:

    column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
     'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'class']
    if mode == Mode.TRAIN:
        path = path + "_data.csv"
        # Load train data
        dt = pd.read_csv(path, header=None, names=column_names,
                         skipinitialspace=True, na_values='?')
    elif mode == Mode.TEST:
        path = path + "_test.csv"
        # Load test data
        dt = pd.read_csv(path, header=0, names=column_names,
                         skipinitialspace=True, na_values='?')
    else:
        raise ValueError('mode must be either train or test')

    # Convert categorical features to numerical labels
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in cat_cols:
        le = LabelEncoder()
        dt[col] = le.fit_transform(dt[col].astype(str))

    # Convert 'y' column to binary 0/1
    dt['y'] = dt['y'].map({'no': 0, 'yes': 1})

    # Create dummy variables for categorical features
    dummy_cols = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    dt = pd.get_dummies(dt, columns=dummy_cols, drop_first=True)

    # Normalize numeric features
    num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    scaler = StandardScaler()
    dt[num_cols] = scaler.fit_transform(dt[num_cols])

    return dt
