import pandas as pd

from preprocess.preprocessing_adult import Mode
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


def load_and_preprocess_bank(mode: Mode = Mode.TRAIN, path: str = "./datasets/bank") -> pd.DataFrame:
    # column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
    #                 'month',
    #                 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'class']
    #
    dt = pd.read_csv(path, header=0, skipinitialspace=True, na_values='?')

    dt.dropna(inplace=True)
    dt.reset_index(drop=True, inplace=True)

    # Drop 'duration' column
    dt.drop('duration', axis=1, inplace=True)

    # rename 'deposit column to 'class'
    dt.rename(columns={'deposit': 'class'}, inplace=True)

    # Convert categorical features to numerical labels
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(dt[categorical_cols])
    dt_encoded = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

    # Convert 'y' column to binary 0/1
    dt['class'] = dt['class'].map({'no': 0, 'yes': 1})

    # Normalize numeric features
    numerical_cols = ['balance', 'day', 'age', 'campaign', 'pdays', 'previous']
    dt[numerical_cols] = dt[numerical_cols].astype(int)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dt[numerical_cols])
    dt_scaled = pd.DataFrame(scaled, columns=numerical_cols)

    dt_preprocessed = pd.concat([dt_encoded, dt_scaled, dt["class"]], axis=1, ignore_index=True)


    return dt_preprocessed
