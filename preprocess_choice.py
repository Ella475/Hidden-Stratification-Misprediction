import preprocessing_adult
import preprocessing_credit


def choose_preprocess_func(dataset_name: str):
    # add "preprocessing_" to the dataset name
    preprocessing_module_name = "preprocessing_" + dataset_name

    # add "load_and_preprocess_" to the dataset name
    load_and_preprocess_func_name = "load_and_preprocess_" + dataset_name

    # get the preprocessing function from the preprocessing module
    return getattr(preprocessing_module_name, load_and_preprocess_func_name)

