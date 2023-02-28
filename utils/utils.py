import json
import pickle


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