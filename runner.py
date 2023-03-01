import json
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm

from clustering.cluster_manager import ClusterManager
from clustering.clustering_calc import clustering
from clustering.clustering_utils import get_clustering_data
from configs.expriment_config import Config, InputMode, DatasetNames, ClusteringMethods, EvaluationMethods
from utils.datasets import customDataset
from training.model_managment import create_model, load_model
from training.train_model import train
from training.test_model import test

from utils.utils import json_save, assert_data_is_finite_and_not_nan, choose_max_div_cluster

import warnings

warnings.filterwarnings('ignore')


def separate_classes(df):
    labels = df.iloc[:, -1]
    df_0 = df[labels == 0]
    df_1 = df[labels == 1]
    df_0.reset_index(drop=True, inplace=True)
    df_1.reset_index(drop=True, inplace=True)
    return df_0, df_1


def experiment_cluster_balance(config: Config, df_0: pd.DataFrame, df_1: pd.DataFrame, cluster_number: int = 0):
    manager = ClusterManager(data=df_0, cluster_number=cluster_number, train_test_percentage=0.2)

    test_loss_list = []
    test_acc_list = []
    test_loss_cluster_list = []
    test_acc_cluster_list = []

    train_loss_list = []
    train_acc_list = []
    train_loss_cluster_list = []
    train_acc_cluster_list = []

    cluster_percentage_list = [0, 0.25, 0.5, 0.75, 1]
    for cluster_percentage in tqdm(cluster_percentage_list):
        train_data_0 = manager.get_train_data(cluster_percentage=cluster_percentage)
        test_data_0 = manager.get_test_data()
        train_data_1, test_data_1 = train_test_split(df_1, test_size=0.2, random_state=42)

        # train on the train set
        train_data = pd.concat([train_data_0, train_data_1], axis=0, ignore_index=True)
        checkpoints_dir = str(config.cluster_checkpoint_dir / str(cluster_percentage))
        train(checkpoint_dir=checkpoints_dir, dataset_name=config.dataset_name.get_value(),
              df_preprocessed=train_data, verbose=False)

        # test on the complete test set
        test_loss, test_acc = test(checkpoint_dir=checkpoints_dir,
                                   test_data=test_data_0.iloc[:, :-1])
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)

        # test on the cluster test set
        test_data_cluster = test_data_0[test_data_0.iloc[:, -1] == cluster_number]
        test_loss_cluster, test_acc_cluster = test(checkpoint_dir=checkpoints_dir,
                                                   test_data=test_data_cluster.iloc[:, :-1])
        test_loss_cluster_list.append(test_loss_cluster)
        test_acc_cluster_list.append(test_acc_cluster)

        # test on the complete train set
        train_data_complete = manager.get_train_data_complete()
        train_loss, train_acc = test(checkpoint_dir=checkpoints_dir,
                                     test_data=train_data_complete.iloc[:, :-1])
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # test on the cluster train set
        train_data_cluster = train_data_complete[train_data_complete.iloc[:, -1] == cluster_number]
        train_loss_cluster, train_acc_cluster = test(checkpoint_dir=checkpoints_dir,
                                                     test_data=train_data_cluster.iloc[:, :-1])
        train_loss_cluster_list.append(train_loss_cluster)
        train_acc_cluster_list.append(train_acc_cluster)

    # create dict for all the results
    results_dict = {"test_loss": test_loss_list, "test_acc": test_acc_list,
                    "test_loss_cluster": test_loss_cluster_list, "test_acc_cluster": test_acc_cluster_list,
                    "train_loss": train_loss_list, "train_acc": train_acc_list,
                    "train_loss_cluster": train_loss_cluster_list, "train_acc_cluster": train_acc_cluster_list,
                    "cluster_percentage": cluster_percentage_list}

    # change result dict to values to scalars from tensor
    for key in results_dict.keys():
        results_dict[key] = [float(value) for value in results_dict[key]]

    json_save(results_dict, str(config.results_dir / 'experiment_cluster_balance.json'))


def experiment_on_class(config: Config, df_0: pd.DataFrame, df_1: pd.DataFrame):
    # call clustering algorithm on the features and labels (train and test)
    div_results, clustering_result_df = clustering(df=df_0, clustering_parameters={},
                                                          clustering_alg_name=config.clustering_method,
                                                          evaluation_alg_name=config.eval_method)

    # save clustering results to pickle file
    results_path = config.results_dir
    # cast inputs_div_results keys to str to be able to save to json
    div_results = {str(key): value for key, value in div_results.items()}
    json_save(div_results, str(results_path / 'div_results.json'))
    clustering_result_df.to_csv(str(results_path / 'df.csv'))

    # stop here if only clustering results are needed
    if config.stop_after_clustering:
        return

    # remove model output (last column) from df_1
    df_1 = df_1.iloc[:, :-1]

    cluster_num = choose_max_div_cluster(div_results)
    experiment_cluster_balance(config, clustering_result_df, df_1, cluster_number=cluster_num)


def run_exp(config: Config):
    df = config.preprocess_func(path=config.dataset_path)

    # load adult model
    model = create_model(df.shape[1] - 1, 1)
    # load latest checkpoint
    checkpoint_dir = config.trained_model_checkpoint_dir
    model = load_model(checkpoint_dir, model)
    # create dataset
    dataset = customDataset(df)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    # get the inputs, outputs, labels, and features
    inputs, outputs, labels, features = get_clustering_data(model, dataloader)

    assert_data_is_finite_and_not_nan(inputs)
    assert_data_is_finite_and_not_nan(outputs)
    assert_data_is_finite_and_not_nan(labels)
    assert_data_is_finite_and_not_nan(features)

    labels = labels.reshape(-1, 1)
    outputs = outputs.reshape(-1, 1).round()

    # create a dataframe with the features and labels (train and test)
    if config.input_mode == InputMode.FEATURES:
        input_df = pd.DataFrame(np.concatenate((features, labels, outputs), axis=1))
    elif config.input_mode == InputMode.INPUTS:
        input_df = pd.DataFrame(np.concatenate((inputs, labels, outputs), axis=1))
    else:
        raise ValueError('input_mode is not valid')

    df_0, df_1 = separate_classes(input_df)
    config.set_main_class_num(0)
    experiment_on_class(config, df_0=df_0, df_1=df_1)
    config.set_main_class_num(1)
    experiment_on_class(config, df_0=df_1, df_1=df_0)


if __name__ == '__main__':
    input_mode = InputMode.FEATURES
    dataset_name = DatasetNames.ADULT
    clustering_method = ClusteringMethods.DBSCAN
    eval_method = EvaluationMethods.PRECISION
    stop_after_clustering = False

    experiment_name = f'{dataset_name.get_value()}_{input_mode.get_value()}_{clustering_method.get_value()}_{eval_method.get_value()}{"_only_cluster" if stop_after_clustering else ""}'
    print(f'Running experiment: {experiment_name}')
    config = Config(experiment_name=experiment_name,
                    input_mode=input_mode,
                    dataset_name=dataset_name,
                    clustering_method=clustering_method,
                    eval_method=eval_method,
                    stop_after_clustering=stop_after_clustering)
    run_exp(config)
