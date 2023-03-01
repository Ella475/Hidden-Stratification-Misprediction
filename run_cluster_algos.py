from configs.expriment_config import Config, InputMode, DatasetNames, ClusteringMethods, EvaluationMethods
from runner import run_exp

if __name__ == '__main__':
    for clustering_method in ClusteringMethods:
        if clustering_method == ClusteringMethods.DBSCAN:
            continue
        if clustering_method == ClusteringMethods.KMEANS:
            continue
        if clustering_method == ClusteringMethods.OPTICS:
            continue
        input_mode = InputMode.FEATURES
        dataset_name = DatasetNames.ADULT
        eval_method = EvaluationMethods.PRECISION
        stop_after_clustering = True

        experiment_name = f'{dataset_name.value[0]}_{input_mode.value}_{clustering_method.value[0]}_{eval_method.value[0]}{"_only_cluster" if stop_after_clustering else ""}'
        print(f'Running experiment: {experiment_name}')
        config = Config(experiment_name=experiment_name,
                        input_mode=input_mode,
                        dataset_name=dataset_name,
                        clustering_method=clustering_method,
                        eval_method=eval_method,
                        stop_after_clustering=stop_after_clustering)
        run_exp(config)
