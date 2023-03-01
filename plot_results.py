import pandas as pd

from configs.expriment_config import Config, InputMode, DatasetNames, ClusteringMethods, EvaluationMethods
from utils.utils import json_load
from utils.graphs import draw_experiment_graphs, plot_cluster_tsne, plot_cluster_sizes_and_divergence


def plot_cluster_results(config: Config):
    results_dir = config.results_dir
    results_csv = results_dir / 'df.csv'
    results_df = pd.read_csv(results_csv, index_col=0, dtype=float)

    # plot the results
    # plot_cluster_tsne(results_df, config)

    div_dict = json_load(results_dir / 'div_results.json')
    plot_cluster_sizes_and_divergence(results_df, div_dict, config)


def plot_exp_results(config: Config):
    results_dir = config.results_dir
    results_json = results_dir / 'experiment_cluster_balance.json'
    results_dict = json_load(results_json)

    # plot the results
    draw_experiment_graphs(config, results_dict)


def plot_classes_exp_results(config: Config):
    config.set_main_class_num(0, exp_plot=True)
    plot_exp_results(config)
    plot_cluster_results(config)
    config.set_main_class_num(1, exp_plot=True)
    plot_exp_results(config)
    plot_cluster_results(config)


if __name__ == '__main__':
    input_mode = InputMode.FEATURES
    dataset_name = DatasetNames.ADULT
    clustering_method = ClusteringMethods.KMEANS
    eval_method = EvaluationMethods.PRECISION

    experiment_name = f'{dataset_name.value[0]}_{input_mode.value}_{clustering_method.value[0]}_{eval_method.value[0]}'

    print(f'Running plots for experiment: {experiment_name}')
    config = Config(experiment_name=experiment_name,
                    input_mode=input_mode,
                    dataset_name=dataset_name,
                    clustering_method=clustering_method,
                    eval_method=eval_method)

    plot_classes_exp_results(config)
