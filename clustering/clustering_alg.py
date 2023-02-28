from numpy import unique
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AffinityPropagation, Birch, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


def dbscan(data, eps=0.25, min_samples=9):
    dbscan_model = DBSCAN(eps, min_samples)
    dbscan_result = dbscan_model.predict(data)
    return dbscan_result, unique(dbscan_result)


def kmeans(data, n_clusters=5):
    k_means_model = KMeans(n_clusters)
    k_meansn_result = k_means_model.fit_predict(data)
    return k_meansn_result, unique(k_meansn_result)


def gaussian_mixture(data, n_components=2):
    gaussian_model = GaussianMixture(n_components)
    gaussian_model.fit(data)
    gaussian_result = gaussian_model.predict(data)
    return gaussian_result, unique(gaussian_result)


def birch(data, threshold=0.03, n_clusters=2):
    birch_model = Birch(threshold, n_clusters)
    birch_model.fit(data)
    birch_result = birch_model.predict(data)
    return birch_result, unique(birch_result)


def affinity_propagation(data, damping=0.7):
    affinity_propagation_model = AffinityPropagation(damping)
    affinity_propagation_model.fit(data)
    affinity_propagation_result = affinity_propagation_model.predict(data)
    return affinity_propagation_result, unique(affinity_propagation_result)


def mean_shift(data):
    mean_shift_model = MeanShift()
    mean_shift_model.fit(data)
    mean_shift_result = mean_shift_model.predict(data)
    return mean_shift_result, unique(mean_shift_result)


def optics(data, eps=0.75, min_samples=10):
    optics_model = OPTICS(eps, min_samples)
    optics_model.fit(data)
    optics_result = optics_model.predict(data)
    return optics_result, unique(optics_result)


def agglomerative_clustering(data, n_clusters=2):
    agglomerative_clustering_model = AgglomerativeClustering(n_clusters)
    agglomerative_clustering_model.fit(data)
    agglomerative_clustering_result = agglomerative_clustering_model.predict(data)
    return agglomerative_clustering_result, unique(agglomerative_clustering_result)