from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.use('TkAgg')

from configs.expriment_config import DatasetNames, InputMode, ClusteringMethods, EvaluationMethods

divergence_dir = Path('results/divergence_plot')
tsne_dir = Path('results/tsne')
imbalance_dir = Path('results/imbalance_plot')


def show_pngs(pngs_path: List[Path]) -> None:
    for png_path in pngs_path:
        plt.figure()
        img = mpimg.imread(png_path)
        plt.imshow(img)

    plt.show()


if __name__ == '__main__':
    class_num = 1

    input_mode = InputMode.FEATURES
    dataset_name = DatasetNames.ADULT
    clustering_method = ClusteringMethods.DBSCAN
    eval_method = EvaluationMethods.PRECISION
    stop_after_clustering = True

    experiment_name = f'{dataset_name.get_value()}_{input_mode.get_value()}_{clustering_method.get_value()}_{eval_method.get_value()}{"_only_cluster" if stop_after_clustering else ""}'
    print(f'Running experiment: {experiment_name}')

    tsne_path = tsne_dir / f'{experiment_name}_{class_num}.png'
    divergence_path = divergence_dir / f'{experiment_name}_{class_num}.png'
    imbalance_path = imbalance_dir / f'{experiment_name}_{class_num}.png'

    if not tsne_path.exists() or not divergence_path.exists():
        print('There are no plots for this experiment')
        exit(1)

    png_paths = [tsne_path, divergence_path]

    if imbalance_path.exists():
        png_paths.append(imbalance_path)

    show_pngs(png_paths)

