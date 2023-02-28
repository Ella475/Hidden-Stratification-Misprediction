import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from enum import Enum
from pathlib import Path
import os
from preprocessing_utils import KBinsDiscretizer_continuos


class Mode(Enum):
    TRAIN = 'train'
    TEST = 'test'


def load_and_preprocess_credit(mode: Mode = Mode.TRAIN, path: str = "./datasets/credit") -> pd.DataFrame:
    gender_map = {"'male single'": "male", "'female div/dep/mar'": "female", "'male mar/wid'": "male",
                  "'male div/sep'": "male"}
    status_map = {"'male single'": "single", "'female div/dep/mar'": "married/wid/sep",
                  "'male mar/wid'": "married/wid/sep", "'male div/sep'": "married/wid/sep"}
    column_names = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration',
                    'Purpose', 'Risk']

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

    dt['sex'] = dt['personal_status'].replace(gender_map)
    dt['civil_status'] = dt['personal_status'].replace(status_map)
    dt.drop(columns=["personal_status"], inplace=True)
    dt.rename(columns={"credit": "class"}, inplace=True)
    dt['class'] = dt['class'].replace({"good": "P", "bad": "N"})
    # Convert categorical variables to numerical labels
    le = LabelEncoder()
    dt['Sex'] = le.fit_transform(dt['Sex'])
    dt['Housing'] = le.fit_transform(dt['Housing'])
    dt['Saving accounts'] = le.fit_transform(dt['Saving accounts'].astype(str))
    dt['Checking account'] = le.fit_transform(dt['Checking account'].astype(str))
    dt['Purpose'] = le.fit_transform(dt['Purpose'])

    # Convert numerical labels to one-hot encoding
    ohe = OneHotEncoder(sparse=False)
    housing_ohe = ohe.fit_transform(dt[['Housing']])
    dt[['Housing_0', 'Housing_1', 'Housing_2']] = pd.DataFrame(housing_ohe, index=dt.index)

    # Drop the original categorical variables
    dt.drop(['Housing'], axis=1, inplace=True)

    # Impute missing values in 'Saving accounts' and 'Checking account'
    dt['Saving accounts'].fillna(value='unknown', inplace=True)
    dt['Checking account'].fillna(value='unknown', inplace=True)

    # Scale the numerical variables
    scaler = StandardScaler()
    dt[['Age', 'Job', 'Credit amount', 'Duration']] = scaler.fit_transform(
        dt[['Age', 'Job', 'Credit amount', 'Duration']])

    return dt
