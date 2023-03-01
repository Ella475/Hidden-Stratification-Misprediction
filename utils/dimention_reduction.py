import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def pca(features: np.ndarray, n_components: int = 25) -> np.ndarray:
    pca_features = PCA(n_components=n_components).fit_transform(features)
    return pca_features


def tsne(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    tsne_features = TSNE(n_components=n_components, init='random').fit_transform(features)
    return tsne_features


def reduce_dim(features: np.ndarray, n_components: int = 25) -> np.ndarray:
    return tsne(pca(features, n_components=n_components), n_components=2)
