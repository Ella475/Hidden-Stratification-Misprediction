from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from configs.expriment_config import DatasetNames, InputMode, ClusteringMethods


def show_images(img_paths: List[Path]) -> None:
    for img_path in img_paths:
        plt.figure()
        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()


def get_plots_path(experiment_name: str, class_num: int) -> List[Path]:
    tsne_dir = Path('results/plots/tsne')
    divergence_dir = Path('results/plots/divergence_plot')
    imbalance_dir = Path('results/plots/imbalance_plot')

    tsne_path = tsne_dir / f'{experiment_name}_{class_num}.png'
    divergence_path = divergence_dir / f'{experiment_name}_{class_num}.png'
    imbalance_path = imbalance_dir / f'{experiment_name}_{class_num}.png'

    if not tsne_path.exists() or not divergence_path.exists():
        print('There are no plots for this experiment')
        return []

    img_paths = [tsne_path, divergence_path]

    if imbalance_path.exists():
        img_paths.append(imbalance_path)

    return img_paths


def get_available_experiments() -> List[str]:
    results_dir = Path('results/plots/tsne')
    experiment_names = [str(path.stem) for path in results_dir.iterdir() if path.suffix == '.png']
    # remove trailing _0 or _1 from experiment name
    experiment_names = [name[:-2] for name in experiment_names]
    experiment_names = list(set(experiment_names))

    return experiment_names


if __name__ == '__main__':
    class_num = 0

    input_mode = InputMode.FEATURES
    dataset_name = DatasetNames.ADULT
    clustering_method = ClusteringMethods.GAUSSIANMIXTURE
    stop_after_clustering = False

    experiment_name = f'{dataset_name.get_value()}_{input_mode.get_value()}_{clustering_method.get_value()}{"_only_cluster" if stop_after_clustering else ""}'
    print(f'Running experiment: {experiment_name}')

    png_paths = get_plots_path(experiment_name, class_num)

    show_images(png_paths)

