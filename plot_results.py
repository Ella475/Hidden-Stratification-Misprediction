from configs.expriment_config import Config
from utils.utils import json_load
from utils.graphs import draw_experiment_graphs


def plot_cluster_results(config: Config):
    results_dir = config.results_dir
    results_json = results_dir / 'experiment_cluster_balance.json'
    results_dict = json_load(results_json)

    # plot the results
    draw_experiment_graphs(config, results_dict)


def plot_classes_results(config: Config):
    config.set_main_class_num(0)
    plot_cluster_results(config)
    config.set_main_class_num(1)
    plot_cluster_results(config)

