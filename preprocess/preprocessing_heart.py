import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from preprocess.preprocessing_adult import Mode


def load_and_preprocess_heart(mode: Mode = Mode.TRAIN, path: str = "./datasets/heart") -> pd.DataFrame:
    dt = pd.read_csv(path, header=0, skipinitialspace=True, na_values='?')
    dt.dropna(inplace=True)
    dt.reset_index(drop=True, inplace=True)

    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

    # rename 'target' column to 'class'
    dt.rename(columns={'target': 'class'}, inplace=True)

    dt[numerical_cols] = dt[numerical_cols].astype(int)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dt[numerical_cols])
    dt_scaled = pd.DataFrame(scaled, columns=numerical_cols)

    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(dt[categorical_cols])
    dt_encoded = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))

    return dt
