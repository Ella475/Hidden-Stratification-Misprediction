from preprocess import preprocessing_adult, preprocessing_credit, preprocessing_heart, preprocessing_bank
from configs.expriment_config import DatasetNames

preprocess_func_dict = {DatasetNames.ADULT: preprocessing_adult.load_and_preprocess_adult,
                        DatasetNames.CREDIT: preprocessing_credit.load_and_preprocess_credit,
                        DatasetNames.HEART: preprocessing_heart.load_and_preprocess_heart,
                        DatasetNames.BANK: preprocessing_bank.load_and_preprocess_bank}


def choose_preprocess_func(dataset_name: DatasetNames):
    return preprocess_func_dict[dataset_name]
