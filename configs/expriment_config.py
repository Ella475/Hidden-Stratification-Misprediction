from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from preprocess.preprocessing_adult import load_and_preprocess_adult
from preprocess.preprocessing_credit import load_and_preprocess_credit
from preprocess.preprocessing_heart import load_and_preprocess_heart
from preprocess.preprocessing_bank import load_and_preprocess_bank


class InputMode(Enum):
    FEATURES = 'features'
    INPUTS = 'inputs'


class DatasetNames(Enum):
    ADULT = 'adult',
    BANK = 'bank',
    CREDIT = 'credit',
    HEART = 'heart'


dataset_path_dict = {DatasetNames.ADULT: 'datasets/adult.csv',
                     DatasetNames.BANK: 'datasets/bank.csv',
                     DatasetNames.CREDIT: 'datasets/credit.csv',
                     DatasetNames.HEART: 'datasets/heart.csv'}


class ClusteringMethods(Enum):
    KMEANS = 'kmeans',
    DBSCAN = 'dbscan',
    OPTICS = 'optics',
    MEANSHIFT = 'mean_shift',
    AFFINITYPROPAGATION = 'affinity_propagation',
    BIRCH = 'birch',
    GAUSSIANMIXTURE = 'gaussian_mixture',
    AGGLOMERATIVECLUSTERING = 'agglomerative_clustering'


class EvaluationMethods(Enum):
    PRECISION = 'precision',
    RECALL = 'recall',
    F1 = 'f1',
    ACCURACY = 'accuracy'


preprocess_func_dict = {DatasetNames.ADULT: load_and_preprocess_adult,
                        DatasetNames.CREDIT: load_and_preprocess_credit,
                        DatasetNames.HEART: load_and_preprocess_heart,
                        DatasetNames.BANK: load_and_preprocess_bank}


def choose_preprocess_func(dataset_name: DatasetNames):
    return preprocess_func_dict[dataset_name]


@dataclass
class Config:
    input_mode: InputMode
    dataset_name: DatasetNames
    clustering_method: str

    def __init__(self, experiment_name: str, input_mode: InputMode, dataset_name: DatasetNames,
                 clustering_method: ClusteringMethods, eval_method: EvaluationMethods):
        self.exp_name = experiment_name
        self.input_mode = input_mode
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path_dict[dataset_name]
        self.clustering_method = clustering_method.value[0]
        self.eval_method = eval_method.value[0]

        self.trained_model_checkpoint_dir = Path(f'training/checkpoints/{self.dataset_name.value[0]}')
        self.trained_model_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.preprocess_func = choose_preprocess_func(dataset_name)

        self.cluster_checkpoint_dir = Path(f'checkpoints/{experiment_name}')
        self.cluster_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = Path(f'results/{experiment_name}')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.show_plots = False

    def set_main_class_num(self, class_num: int, exp_plot: bool = False):
        self.cluster_checkpoint_dir = Path(f'checkpoints/{self.exp_name}/{class_num}')
        self.cluster_checkpoint_dir.mkdir(parents=True, exist_ok=exp_plot)

        self.results_dir = Path(f'results/{self.exp_name}/{class_num}')
        self.results_dir.mkdir(parents=True, exist_ok=exp_plot)

    def set_show_plots(self):
        self.show_plots = True
