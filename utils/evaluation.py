import numpy as np
from sklearn.metrics import silhouette_score, precision_score, recall_score


# evaluate the quality of the clustering
def clustering_evaluation(x: np.array, y_pred: np.array):
    return silhouette_score(x, y_pred)


def precision(y_true: np.array, y_pred: np.array):
    return precision_score(y_true, y_pred)


def recall(y_true: np.array, y_pred: np.array):
    return recall_score(y_true, y_pred)
