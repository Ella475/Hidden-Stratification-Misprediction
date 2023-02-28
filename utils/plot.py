from sympy.physics.quantum.circuitplot import pyplot
from numpy import where
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot2Dimensions (data, clusters):

    # make the plot, the data is df with three columns, the third us the clustering result
    # and it's name is result
    g = sns.scatterplot(data=data, x=data[:, 0], y=data[:, 1], hue='result',
                        palette=sns.color_palette('hls', n_colors=len(clusters)), alpha=.5)
    plt.title('Clusters after reduced dimensions')
    plt.xlabel('')
    plt.ylabel('')
    g.get_figure().savefig()
    g.get_figure().clf()
    pyplot.show()

