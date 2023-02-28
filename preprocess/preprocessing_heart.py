import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from preprocess.preprocessing_adult import Mode
from preprocessing_utils import KBinsDiscretizer_continuos, quantizePrior, quantizeLOS, get_decile_score_class


def load_and_preprocess_heart(mode: Mode = Mode.TRAIN, path: str = "./datasets/heart") -> pd.DataFrame:
    column_names = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
                    'rest_ecg', 'max_heart_rate_achieved',
                    'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'class']

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
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    for col in cat_cols:
        le = LabelEncoder()
        dt[col] = le.fit_transform(dt[col].astype(str))

    # Normalize numeric features
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    scaler = StandardScaler()
    dt[num_cols] = scaler.fit_transform(dt[num_cols])

    return dt
