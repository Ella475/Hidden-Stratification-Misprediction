import numpy as np

from configs.expriment_config import InputMode, DatasetNames, ClusteringMethods, Config
from utils.utils import json_load


def results_dict_std_and_max_div(results_dict: dict) -> (float, float):
    values = np.array(list(results_dict.values()))

    def std(x: np.array) -> float:
        return np.sqrt(np.sum(x ** 2) / len(x))

    def max_div(x: np.array) -> float:
        return np.max(np.abs(x))

    return std(values), max_div(values)


def parse_experiment_results(config: Config) -> (str, str):
    config.set_main_class_num(0, exp_plot=True)
    results_dir = config.results_dir
    results_dict = json_load(results_dir / 'div_results.json')
    std_0, max_div_0 = results_dict_std_and_max_div(results_dict)

    config.set_main_class_num(1, exp_plot=True)
    results_dir = config.results_dir
    results_dict = json_load(results_dir / 'div_results.json')
    std_1, max_div_1 = results_dict_std_and_max_div(results_dict)

    return f'{std_0:.2f}, {max_div_0:.2f}', f'{std_1:.2f}, {max_div_1:.2f}'


if __name__ == '__main__':
    for dataset_name in DatasetNames:
        lst_0 = []
        lst_1 = []
        for input_mode in InputMode:
            class_method = ClusteringMethods.KMEANS
            stop_after_clustering = False

            experiment_name = f'{dataset_name.get_value()}_{input_mode.get_value()}_{class_method.get_value()}{"_only_cluster" if stop_after_clustering else ""}'
            config = Config(experiment_name=experiment_name,
                            input_mode=input_mode,
                            dataset_name=dataset_name,
                            clustering_method=class_method,
                            stop_after_clustering=stop_after_clustering)

            results_0, results_1 = parse_experiment_results(config)
            lst_0.append(f'({results_0} )')
            lst_1.append(f'({results_1} )')
            print(input_mode.get_value())
        print(dataset_name.get_value())
        print(*lst_0, sep=' & ')
        print(*lst_1, sep=' & ')