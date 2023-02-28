import pandas as pd
from preprocess import preprocessing_adult, preprocessing_credit, preprocessing_heart, preprocessing_bank


def choose_preprocess_func(dataset_name: str = "adult", loaded_df: pd.DataFrame = None):

    if loaded_df is not None:
        return loaded_df
    if dataset_name == "adult":
        return preprocessing_adult.load_and_preprocess_adult
    elif dataset_name == "credit":
        return preprocessing_credit.load_and_preprocess_credit
    elif dataset_name == "heart":
        return preprocessing_heart.load_and_preprocess_heart
    elif dataset_name == "bank":
        return preprocessing_bank.load_and_preprocess_bank
    else:
        raise ValueError("Dataset not supported")


