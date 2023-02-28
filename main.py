from enum import Enum
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cluster_manager import ClusterManager
from clustering_calc import clustering
from clustering_utils import get_clustering_data
from datasets import customDataset
from preprocess.preprocessing_adult import load_and_preprocess_adult, Mode
from training.model_managment import create_model, load_model
from training.train_model import train, test


class InputMode(Enum):
    FEATURES = 1
    INPUTS = 2


def assert_data_is_finite_and_not_nan(data):
    assert np.all(np.isfinite(data)), "Data contains NaN or infinite values"
    assert not np.any(np.isnan(data)), "Data contains NaN or infinite values"
    return True


def separate_classes(df):
    labels = df.iloc[:, -1]
    df_0 = df[labels == 0]
    df_1 = df[labels == 1]
    df_0.reset_index(drop=True, inplace=True)
    df_1.reset_index(drop=True, inplace=True)
    return df_0, df_1


def draw_experiment_graphs(name, loss_list, acc_list, loss_cluster_list, acc_cluster_list, cluster_percentage_list):
    cluster_percentage_list = [x * 100 for x in cluster_percentage_list]
    # draw 4 subplots in one figure
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Experiment {name}")
    axs[0, 0].plot(cluster_percentage_list, loss_list)
    axs[0, 0].set_title("Loss")
    axs[0, 1].plot(cluster_percentage_list, acc_list)
    axs[0, 1].set_title("Accuracy")
    axs[1, 0].plot(cluster_percentage_list, loss_cluster_list)
    axs[1, 0].set_title("Loss cluster")
    axs[1, 1].plot(cluster_percentage_list, acc_cluster_list)
    axs[1, 1].set_title("Accuracy cluster")

    axs[0, 0].set(xlabel="Cluster percentage", ylabel="Loss")
    axs[0, 1].set(xlabel="Cluster percentage", ylabel="Accuracy")
    axs[1, 0].set(xlabel="Cluster percentage", ylabel="Loss cluster")
    axs[1, 1].set(xlabel="Cluster percentage", ylabel="Accuracy cluster")

    plt.savefig(f"experiment_graphs_{name}.png")
    plt.show()


def experiment_cluster_balance(name, df_0, df_1):
    cluster_number = 1
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
    for cluster_percentage in cluster_percentage_list:
        train_data_0 = manager.get_train_data(cluster_percentage=cluster_percentage)
        test_data_0 = manager.get_test_data()
        train_data_1, test_data_1 = train_test_split(df_1, test_size=0.2, random_state=42)

        train_data = pd.concat([train_data_0, train_data_1], axis=0, ignore_index=True)
        train(checkpoint_dir=f"./checkpoints/adult_{cluster_percentage}", dataset_name="adult",
              df_preprocessed=train_data)

        # test on the complete test set
        test_loss, test_acc = test(checkpoint_dir=f"./checkpoints/adult_{cluster_percentage}",
                                   test_data=test_data_0.iloc[:, :-1])
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        print(f"cluster_percentage: {cluster_percentage}, test_loss: {test_loss}, test_acc: {test_acc}")

        test_data_cluster = test_data_0[test_data_0.iloc[:, -1] == cluster_number]
        test_loss_cluster, test_acc_cluster = test(checkpoint_dir=f"./checkpoints/adult_{cluster_percentage}",
                                                   test_data=test_data_cluster.iloc[:, :-1])
        test_loss_cluster_list.append(test_loss_cluster)
        test_acc_cluster_list.append(test_acc_cluster)
        print(f"cluster_percentage: {cluster_percentage}, test_loss_cluster: {test_loss_cluster}, "
              f"test_acc_cluster: {test_acc_cluster}")

        # test on the complete train set
        train_data_complete = manager.get_train_data_complete()
        train_loss, train_acc = test(checkpoint_dir=f"./checkpoints/adult_{cluster_percentage}",
                                     test_data=train_data_complete.iloc[:, :-1])
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        print(f"cluster_percentage: {cluster_percentage}, train_loss: {train_loss}, train_acc: {train_acc}")

        train_data_cluster = train_data_complete[train_data_complete.iloc[:, -1] == cluster_number]
        train_loss_cluster, train_acc_cluster = test(checkpoint_dir=f"./checkpoints/adult_{cluster_percentage}",
                                                     test_data=train_data_cluster.iloc[:, :-1])
        train_loss_cluster_list.append(train_loss_cluster)
        train_acc_cluster_list.append(train_acc_cluster)
        print(f"cluster_percentage: {cluster_percentage}, train_loss_cluster: {train_loss_cluster}, "
              f"train_acc_cluster: {train_acc_cluster}")

    draw_experiment_graphs(name + '_test', test_loss_list, test_acc_list, test_loss_cluster_list, test_acc_cluster_list,
                           cluster_percentage_list)

    draw_experiment_graphs(name + '_train', train_loss_list, train_acc_list, train_loss_cluster_list, train_acc_cluster_list,
                           cluster_percentage_list)


def main(name, input_mode=None):
    df = load_and_preprocess_adult(path='datasets/adult.data')

    # load adult model
    model = create_model(df.shape[1] - 1, 1)
    # load latest checkpoint
    checkpoint_dir = Path('training/checkpoints/adult')
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
    if input_mode == InputMode.FEATURES:
        input_df = pd.DataFrame(np.concatenate((features, labels, outputs), axis=1))
    elif input_mode == InputMode.INPUTS:
        input_df = pd.DataFrame(np.concatenate((inputs, labels, outputs), axis=1))
    else:
        raise ValueError('input_mode is not valid')

    df_0, df_1 = separate_classes(input_df)

    # call clustering algorithm on the features and labels (train and test)
    inputs_div_results, clustering_result_df = clustering(df_0, clustering_parameters={})
    print("inputs_div_results: ", inputs_div_results)

    # remove model output (last column) from df_1
    df_1 = df_1.iloc[:, :-1]

    experiment_cluster_balance(name, clustering_result_df, df_1)


if __name__ == '__main__':
    main(name="adult", input_mode=InputMode.INPUTS)
