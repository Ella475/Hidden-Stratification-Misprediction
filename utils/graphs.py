import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from utils.utils import assert_data_is_finite_and_not_nan
from utils.dimention_reduction import reduce_dim
from configs.expriment_config import Config


import warnings

warnings.filterwarnings('ignore')

mpl.use('TkAgg')


def plot_cluster_tsne(df: pd.DataFrame, config: Config):
    clusters = df.iloc[:, -1].to_numpy()
    features = df.iloc[:, 1:-2].to_numpy()

    # Perform pca to reduce dimensionality to 25 components and then perform t-SNE to reduce to 2 components
    lower_dim_features = reduce_dim(features, n_components=25)

    tsne_x = lower_dim_features[:, 0]
    tsne_y = lower_dim_features[:, 1]

    # Create scatter plot of t-SNE representation with color-coded clusters and labels
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_x, tsne_y, c=clusters, cmap='rainbow')

    plt.title('t-SNE Visualization of clustering space of experiment' + str(config.exp_name))

    class_num = str(config.results_dir)[-1]
    plt.savefig(f'results/tsne/{config.exp_name}_{class_num}.png')

    if config.show_plots:
        plt.show()


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

    class_num = str(config.results_dir)[-1]
    plt.savefig(f'results/imbalance_plot/{config.exp_name}_{class_num}.png')
    if config.show_plots:
        plt.show()


def plot_cluster_sizes_and_divergence(df, divergence_scores, config: Config):
    cluster_sizes = df.iloc[:, -1].value_counts()
    clusters = cluster_sizes.index

    fig, ax = plt.subplots()

    for cluster in clusters:
        size = cluster_sizes[cluster]
        divergence = divergence_scores[str(int(cluster))]
        ax.scatter(size, divergence, color='blue')
        ax.annotate(str(int(cluster)), (size, divergence))

    all_size = df.iloc[:, -1].value_counts().sum()
    divergence = divergence_scores['all']
    ax.scatter(all_size, divergence, color='red')
    ax.annotate('all', (all_size, divergence))
    ax.set_xlabel('Cluster Size')
    ax.set_ylabel('Divergence Score')

    class_num = str(config.results_dir)[-1]
    plt.savefig(f'results/divergence_plot/{config.exp_name}_{class_num}.png')

    if config.show_plots:
        plt.show()



