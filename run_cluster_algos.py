from configs.expriment_config import Config, InputMode, DatasetNames, ClusteringMethods, EvaluationMethods
from runner import run_exp

if __name__ == '__main__':
    for clustering_method in ClusteringMethods:
        input_mode = InputMode.FEATURES
        dataset_name = DatasetNames.ADULT
        eval_method = EvaluationMethods.PRECISION
        stop_after_clustering = True

        experiment_name = f'{dataset_name.get_value()}_{input_mode.get_value()}_{clustering_method.get_value()}_{eval_method.get_value()}{"_only_cluster" if stop_after_clustering else ""}'
        print(f'Running experiment: {experiment_name}')
        config = Config(experiment_name=experiment_name,
                        input_mode=input_mode,
                        dataset_name=dataset_name,
                        clustering_method=clustering_method,
                        eval_method=eval_method,
                        stop_after_clustering=stop_after_clustering)
        run_exp(config)
