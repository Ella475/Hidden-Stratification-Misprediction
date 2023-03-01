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