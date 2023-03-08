import pandas as pd

from configs.expriment_config import Config
from utils.utils import json_load
from plots.graphs import draw_experiment_graphs, plot_cluster_tsne, plot_cluster_sizes_and_divergence


def plot_cluster_results(config: Config):
    results_dir = config.results_dir
    results_csv = results_dir / 'df.csv'
    results_df = pd.read_csv(results_csv, index_col=0, dtype=float)

    # plot the results
    plot_cluster_tsne(results_df, config)

    div_dict = json_load(results_dir / 'div_results.json')
    plot_cluster_sizes_and_divergence(results_df, div_dict, config)


def plot_exp_results(config: Config):
    results_dir = config.results_dir
    results_json = results_dir / 'experiment_cluster_balance.json'
    results_dict = json_load(results_json)

    # plot the results
    draw_experiment_graphs(config, results_dict)


def plot_classes_exp_results(config: Config, ext: str = ''):
    config.set_main_class_num(0, exp_plot=True)
    if not config.stop_after_clustering:
        plot_exp_results(config)
    plot_cluster_results(config)
    config.set_main_class_num(1, exp_plot=True)
    if not config.stop_after_clustering:
        plot_exp_results(config)
    plot_cluster_results(config)
