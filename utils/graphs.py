import matplotlib.pyplot as plt
import matplotlib as mpl
from configs.expriment_config import Config
import pandas as pd
from sklearn.manifold import TSNE

import warnings

warnings.filterwarnings('ignore')

mpl.use('TkAgg')


def plot_cluster_tsne(df: pd.DataFrame, config: Config):
    # Remove last two columns from DataFrame and convert to numpy array
    features = df.iloc[:, :-2].to_numpy()

    df['clusters'] = df.iloc[:, -1]

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_features = tsne.fit_transform(features)

    # Add t-SNE coordinates to DataFrame
    df['tsne_x'] = tsne_features[:, 0]
    df['tsne_y'] = tsne_features[:, 1]

    # Create scatter plot of t-SNE representation with color-coded clusters and labels
    plt.figure(figsize=(10, 8))
    plt.scatter(df['tsne_x'], df['tsne_y'], c=df['clusters'], cmap='rainbow')

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



