import pandas as pd
from typing import List, Dict, Tuple, Union
import clustering_alg
import evaluation
import numpy as np


def clustering(df: pd.DataFrame, clustering_alg_name: str = "kmeans", clustering_parameters: Dict = None,
               evaluation_alg_name: str = "precision") -> Tuple[Dict, float]:
    # cluster the df of true_class_name, do i need to copy?
    if clustering_parameters is None:
        raise ValueError("clustering parameters are not defined")

    clustering_fnc = getattr(clustering_alg, clustering_alg_name)

    # take all columns of df except the last two (predicted values and true values) and call clustering algorithm
    clustering_result, clusters = clustering_fnc(df.iloc[:, :-2], **clustering_parameters)

    # concatenate the predicted values and true values with the clustering results as concatenated_df
    concatenated_df = pd.concat([df.iloc[:, -2:], pd.DataFrame(clustering_result)], axis=1)

    divergence_dict = {}
    eval_fnc = getattr(evaluation, evaluation_alg_name)
    # compute evaluation score for the whole set
    evaluation_score = eval_fnc(concatenated_df.iloc[:, 0].to_numpy(), concatenated_df.iloc[:, 1].to_numpy())

    divergence_dict["all"] = evaluation_score

    for cluster in clusters:
        # get only certain cluster,
        # get only the lines where the last column (the clustering result) is equal to the cluster
        y_values_of_cluster = concatenated_df[concatenated_df.iloc[:, -1] == cluster].iloc[:, :-1]
        # call evaluation alg of the subset with true and predicted values
        divergence_dict[cluster] = evaluation_score - \
                                   eval_fnc(y_values_of_cluster.iloc[:, 0].to_numpy(),
                                            y_values_of_cluster.iloc[:, 1].to_numpy())

    return divergence_dict, clustering_result
