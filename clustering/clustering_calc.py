import pandas as pd
from typing import Dict, Tuple
import clustering.clustering_alg as clustering_alg
from utils import evaluation


def clustering(df: pd.DataFrame, clustering_alg_name: str = "kmeans", clustering_parameters: Dict = None,
               evaluation_alg_name: str = "precision") -> Tuple[Dict, pd.DataFrame]:
    # cluster the df of true_class_name, do i need to copy?
    if clustering_parameters is None:
        raise ValueError("clustering parameters are not defined")

    clustering_fnc = getattr(clustering_alg, clustering_alg_name)

    # take all columns of df except the last two (predicted values and true values) and call clustering algorithm
    clustering_result, clusters = clustering_fnc(df.iloc[:, :-2], **clustering_parameters)

    clustering_result_df = pd.DataFrame(clustering_result)
    # concatenate the predicted values and true values with the clustering results as concatenated_df
    df_with_clusters = pd.concat([df, clustering_result_df], axis=1, ignore_index=True)
    concatenated_df = df_with_clusters.iloc[:, -3:]

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
        divergence_dict[cluster] = (evaluation_score - eval_fnc(y_values_of_cluster.iloc[:, 0].to_numpy(),
                                                                y_values_of_cluster.iloc[:, 1].to_numpy()))

    # drop second to last column (the output of the model)
    df_with_clusters = df_with_clusters.drop(df_with_clusters.columns[-2], axis=1)

    return divergence_dict, df_with_clusters
