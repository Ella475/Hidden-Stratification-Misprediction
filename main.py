from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from clustering_calc import clustering
from clustering_utils import get_clustering_data
from datasets import customDataset
from preprocess.preprocessing_adult import load_and_preprocess_adult, Mode
from training.model_managment import create_model, load_model
from cluster_distrebution_modification import get_num_of_rows_to_remove, get_changed_datasets


def main():
    checkpoint_dir = Path('.training/checkpoints/adult')
    df_train = load_and_preprocess_adult()
    df_test = load_and_preprocess_adult(mode=Mode.TEST)

    # print missing columns in test set (if any)
    missing_cols = set(df_train.columns) - set(df_test.columns)


    # load adult model
    model = create_model(df_train.shape[1] - 1, 1)
    # load latest checkpoint
    model = load_model(checkpoint_dir, model)
    # create dataset
    train_dataset = customDataset(df_train)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_dataset = customDataset(df_test)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # get the inputs, outputs, labels, and features
    train_inputs, train_outputs, train_labels, train_features = get_clustering_data(model, train_dataloader)
    test_inputs, test_outputs, test_labels, test_features = get_clustering_data(model, test_dataloader)

    train_labels = train_labels.reshape(-1, 1).round()
    train_outputs = train_outputs.reshape(-1, 1).round()

    # create a dataframe with the features and labels (train and test)
    train_features_df = pd.DataFrame(np.concatenate((train_features, train_labels, train_outputs), axis=1))
    train_inputs_df = pd.DataFrame(np.concatenate((train_inputs, train_labels, train_outputs), axis=1))
    test_inputs_df = pd.DataFrame(np.concatenate((test_inputs, test_labels, test_outputs), axis=1))
    # test_features_df = pd.DataFrame(np.concatenate((test_features, test_labels, test_outputs), axis=1))

    # call clustering algorithm on the features and labels (train and test)
    train_inputs_div_results, train_clustering_result = clustering(train_inputs_df, clustering_parameters={})
    # train_features_div_results, clustering_result = clustering(train_features_df, clustering_parameters={})
    test_inputs_div_results, test_clustering_result = clustering(test_inputs_df, clustering_parameters={})
    # test_features_div_results, clustering_result = clustering(test_features_df, clustering_parameters={})

    print(train_inputs_div_results)

    num_of_rows_to_remove, largest_division_indices =\
        get_num_of_rows_to_remove(train_inputs_div_results, train_clustering_result)

    for rows in num_of_rows_to_remove:
        get_changed_datasets(rows, largest_division_indices, train_clustering_result, test_clustering_result)
        # call training function with the changed dataset



if __name__ == '__main__':
    main()
