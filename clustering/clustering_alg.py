from numpy import unique
from sklearn.cluster import KMeans, DBSCAN, Birch, OPTICS
from sklearn.mixture import GaussianMixture


def dbscan(data, eps=0.25, min_samples=50):
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_result = dbscan_model.fit(data)
    dbscan_result = dbscan_result.labels_
    return dbscan_result, unique(dbscan_result)


def kmeans(data, n_clusters=5):
    k_means_model = KMeans(n_clusters=n_clusters)
    k_meansn_result = k_means_model.fit_predict(data)
    return k_meansn_result, unique(k_meansn_result)


def gaussianmixture(data, n_components=5):
    gaussian_model = GaussianMixture(n_components=n_components)
    gaussian_model.fit(data)
    gaussian_result = gaussian_model.predict(data)
    return gaussian_result, unique(gaussian_result)


def birch(data, threshold=0.03, n_clusters=5):
    birch_model = Birch(threshold=threshold, n_clusters=n_clusters)
    birch_model.fit(data)
    birch_result = birch_model.predict(data)
    return birch_result, unique(birch_result)


def optics(data, eps=0.75, min_samples=50):
    optics_model = OPTICS(eps=eps, min_samples=min_samples)
    optics_model.fit(data)
    optics_result = optics_model.labels_
    return optics_result, unique(optics_result)
