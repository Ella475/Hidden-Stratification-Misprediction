import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from enum import Enum
from pathlib import Path
import os
from preprocessing_utils import KBinsDiscretizer_continuos


class Mode(Enum):
    TRAIN = 'train'
    TEST = 'test'


def load_and_preprocess_adult(mode: Mode = Mode.TRAIN, path: str = "./datasets/adult") -> pd.DataFrame:
    education_map = {
        '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
        'Some-college': 'High School grad', 'Masters': 'Masters',
        'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
        'Assoc-voc': 'Associates',
    }
    occupation_map = {
        "Adm-clerical": "Admin", "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar",
    }
    married_map = {
        'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
        'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
    }

    country_map = {
        'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
        'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
        'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
        'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
        'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
        'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
        'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
        'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
        'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
        'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
        'United-States': 'United-States', 'Vietnam': 'SE-Asia'
    }
    # as given by adult.names
    column_names = ['age', 'workclass', 'fnlwgt', 'education',
                    'education-num', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income-per-year']
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

    dt["education"] = dt["education"].replace(education_map)
    dt.drop(columns=["education-num", "fnlwgt"], inplace=True)
    dt["occupation"] = dt["occupation"].replace(occupation_map)
    dt["marital-status"] = dt["marital-status"].replace(married_map)
    dt["native-country"] = dt["native-country"].replace(country_map)

    dt.rename(columns={"income-per-year": "class"}, inplace=True)
    dt["class"] = dt["class"].astype('str').replace({">50K.": ">50K", "<=50K.": "<=50K"})
    dt["class"] = dt["class"].map({"<=50K": 0, ">50K": 1})
    dt.dropna(inplace=True)
    dt.reset_index(drop=True, inplace=True)
    # dt = KBinsDiscretizer_continuos(dt, bins=3)
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                        'native-country']
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(dt[categorical_cols])
    dt_encoded = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names(categorical_cols))

    numerical_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dt[numerical_cols])
    dt_scaled = pd.DataFrame(scaled, columns=numerical_cols)

    dt_preprocessed = pd.concat([dt_encoded, dt_scaled, dt["class"]], axis=1)

    dt.drop(columns=["native-country"], inplace=True)

    return dt_preprocessed

