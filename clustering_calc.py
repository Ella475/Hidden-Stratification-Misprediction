import pandas as pd

import clustering_alg
import evaluation
import numpy as np


def clusteringCalc(df, clustering_alg_name="kmeans", cluster_parameters=None, evaluation_alg_name="precision"):
    # cluster the df of true_class_name, do i need to copy?
    if cluster_parameters is not None:
        clustering_fnc = getattr(clustering_alg, clustering_alg_name)

        # take all but the last  two columns (predicted values and true values), and call clustering algorithm
        clustering_result, clusters = clustering_fnc(df.iloc[:, :-2])

        # concatenate the predicted values and true values with the clustering results
        predicted_df = pd.concat([df.loc[:, :-2], clustering_result])

        divergence_dict = {}
        eval_fnc = getattr(evaluation, evaluation_alg_name)
        # compute evaluation score for the whole set
        evaluation_score_of_set = eval(predicted_df[:, 0], predicted_df[:, 1])

        for cluster in clusters:
            # get only certain cluster,
            y_values_of_cluster = predicted_df.loc[predicted_df[:, 3] == cluster]
            # call evaluation alg of the subset with true and  predicted values
            divergence_dict[cluster] = eval(y_values_of_cluster[:, 0], y_values_of_cluster[:, 1])

        # compute the divergence for each subclass
        divergence_dict.update((key, evaluation_score_of_set - value) for key, value in divergence_dict.items())