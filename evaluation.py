import sklearn
from sklearn.metrics import silhouette_score


# evaluate the quality of the clustering
def clusteringEvaluation(x, y_pred):
    return silhouette_score(x, y_pred)


def precision(y_true, y_pred):
    return sklearn.metrics.precision_score(y_true, y_pred)


def recall_score(y_true, y_pred):
    return sklearn.metrics.recall_score(y_true, y_pred)


# def divergence(func, set_y_true, set_y_pred, subset_y_true, subset_y_pred):
#     return func(set_y_true, set_y_pred) - func(subset_y_true, subset_y_pred)
