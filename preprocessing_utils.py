import numpy as np
import os

DATASET_DIR = os.path.join(".", "datasets")

def KBinsDiscretizer_continuos(dt, attributes=None, bins=3):
    attributes = dt.columns if attributes is None else attributes
    continuous_attributes = [a for a in attributes if dt.dtypes[a] != np.object]
    X_discretize = dt[attributes].copy()

    for col in continuous_attributes:
        if len(dt[col].value_counts()) > 10:
            from sklearn.preprocessing import KBinsDiscretizer
            est = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
            est.fit(dt[[col]])
            edges = [i.round() for i in est.bin_edges_][0]
            edges = [int(i) for i in edges][1:-1]
            if len(set(edges)) != len(edges):
                edges = [edges[i] for i in range(0, len(edges)) if len(edges) - 1 == i or edges[i] != edges[i + 1]]
            for i in range(0, len(edges)):
                if i == 0:
                    data_idx = dt.loc[dt[col] <= edges[i]].index
                    X_discretize.loc[data_idx, col] = f"<={edges[i]}"
                if i == len(edges) - 1:
                    data_idx = dt.loc[dt[col] > edges[i]].index
                    X_discretize.loc[data_idx, col] = f">{edges[i]}"

                data_idx = dt.loc[(dt[col] > edges[i - 1]) & (dt[col] <= edges[i])].index
                X_discretize.loc[data_idx, col] = f"({edges[i - 1]}-{edges[i]}]"
        else:
            X_discretize[col] = X_discretize[col].astype('object')
    return X_discretize


def cap_gains_fn(x):
    x = x.astype(float)
    d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                    right=True)  # .astype('|S128')
    return d.copy()


def discretize(dfI, bins=4, dataset_name=None, attributes=None, indexes_FP=None):
    indexes_validation = dfI.index if indexes_FP is None else indexes_FP
    attributes = dfI.columns if attributes is None else attributes
    if dataset_name == "compas":
        X_discretized = dfI[attributes].copy()
        X_discretized["priors_count"] = X_discretized["priors_count"].apply(lambda x: quantizePrior(x))
        X_discretized["length_of_stay"] = X_discretized["length_of_stay"].apply(lambda x: quantizeLOS(x))
    elif dataset_name == "adult":
        X_discretized = dfI[attributes].copy()
        X_discretized["capital-gain"] = cap_gains_fn(X_discretized["capital-gain"].values)
        X_discretized["capital-gain"] = X_discretized["capital-gain"].replace({0: '0', 1: 'Low', 2: 'High'})
        X_discretized["capital-loss"] = cap_gains_fn(X_discretized["capital-loss"].values)
        X_discretized["capital-loss"] = X_discretized["capital-loss"].replace({0: '0', 1: 'Low', 2: 'High'})
        X_discretized = KBinsDiscretizer_continuos(X_discretized, attributes, bins=bins)
    else:
        X_discretized = KBinsDiscretizer_continuos(dfI, attributes, bins=bins)
    return X_discretized.loc[indexes_validation].reset_index(drop=True)

def quantizePrior(x):
    if x <= 0:
        return '0'
    elif 1 <= x <= 3:
        return '[1,3]'
    else:
        return '>3'


# Quantize length of stay
def quantizeLOS(x):
    if x <= 7:
        return '<week'
    if 8 < x <= 93:
        return '1w-3M'
    else:
        return '>3Months'


# Class label
# df_raw[["score_text","decile_score"]].sort_values(["score_text", "decile_score"])

def get_decile_score_class(x):
    if x >= 8:
        return 'High'
    else:
        return 'Medium-Low'


def get_decile_score_class2(x):
    if x >= 5:
        return 'Medium-High'
    else:
        return 'Low'