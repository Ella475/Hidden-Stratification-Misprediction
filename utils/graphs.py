import matplotlib.pyplot as plt
import matplotlib as mpl
from configs.expriment_config import Config
import os
import json

mpl.use('TkAgg')


def draw_experiment_graphs(config: Config, results_dict: dict):
    cluster_percentage = [int(x * 100) for x in results_dict["cluster_percentage"]]

    # plot in 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Experiment: {str(config.exp_name)}")

    # test loss
    axs[0, 0].plot(cluster_percentage, results_dict["test_loss"], label="test_loss")
    axs[0, 0].plot(cluster_percentage, results_dict["test_loss_cluster"], label="test_loss_cluster")
    axs[0, 0].set_title("Test Loss")
    axs[0, 0].set_xlabel("Cluster Percentage")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    # test acc
    axs[0, 1].plot(cluster_percentage, results_dict["test_acc"], label="test_acc")
    axs[0, 1].plot(cluster_percentage, results_dict["test_acc_cluster"], label="test_acc_cluster")
    axs[0, 1].set_title("Test Accuracy")
    axs[0, 1].set_xlabel("Cluster Percentage")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].legend()

    # train loss
    axs[1, 0].plot(cluster_percentage, results_dict["train_loss"], label="train_loss")
    axs[1, 0].plot(cluster_percentage, results_dict["train_loss_cluster"], label="train_loss_cluster")
    axs[1, 0].set_title("Train Loss")
    axs[1, 0].set_xlabel("Cluster Percentage")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].legend()

    # train acc
    axs[1, 1].plot(cluster_percentage, results_dict["train_acc"], label="train_acc")
    axs[1, 1].plot(cluster_percentage, results_dict["train_acc_cluster"], label="train_acc_cluster")
    axs[1, 1].set_title("Train Accuracy")
    axs[1, 1].set_xlabel("Cluster Percentage")
    axs[1, 1].set_ylabel("Accuracy")
    axs[1, 1].legend()

    plt.savefig(str(config.results_dir / 'graphs.png'))
    plt.show()

