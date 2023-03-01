import json
import pickle
import numpy as np


def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def pickle_save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def json_load(path):
    with open(path, 'r') as f:
        return json.load(f)


def json_save(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def assert_data_is_finite_and_not_nan(data):
    assert np.all(np.isfinite(data)), "Data contains NaN or infinite values"
    assert not np.any(np.isnan(data)), "Data contains NaN or infinite values"
    return True


def choose_max_div_cluster(div_results: dict) -> int:
    max_div = -100
    max_div_cluster = 0
    for cluster, div in div_results.items():
        if div > max_div:
            max_div = div
            max_div_cluster = int(cluster)
    return max_div_cluster
