import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import json

mpl.use('TkAgg')


def draw_experiment_graphs(name, results_dict):
    cluster_percentage = [int(x * 100) for x in results_dict["cluster_percentage"]]

    # plot in 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f"Experiment: {name}")

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

    plt.savefig(f"images/{name}.png")


if __name__ == "__main__":
    # loop over all json files in current directory
    for file in os.listdir():
        if file.endswith(".json"):
            with open(file, "r") as f:
                results_dict = json.load(f)
                name = file.split(".")[0]
                draw_experiment_graphs(name, results_dict)
