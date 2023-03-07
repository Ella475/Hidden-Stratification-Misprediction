from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def svg_to_png(svg_folder: Path, png_folder: Path):
    # create png folder if not exists
    png_folder.mkdir(parents=True, exist_ok=True)
    # convert svg to png
    for svg_path in svg_folder.glob('*.svg'):
        png_path = png_folder / (svg_path.stem + '.png')
        try:
            plt.imsave(png_path, mpimg.imread(svg_path, format='svg'))
        except Exception as e:
            print(f'Error: {e}')


if __name__ == '__main__':
    svg_folder = Path('results/plots/tsne')
    png_folder = Path('results/plots/tsne_png')
    svg_to_png(svg_folder, png_folder)

    svg_folder = Path('results/plots/divergence_plot')
    png_folder = Path('results/plots/divergence_plot_png')
    svg_to_png(svg_folder, png_folder)

    svg_folder = Path('results/plots/imbalance_plot')
    png_folder = Path('results/plots/imbalance_plot_png')
    svg_to_png(svg_folder, png_folder)

