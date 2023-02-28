import numpy as np


def get_num_of_rows_to_remove(div_results, train_clustering_result):
    # choose the largest division from the train_inputs_div_results dictionary
    largest_division_indices = max(div_results, key=div_results.get)

    #  get rows from clustering_result which equals largest_division_indices
    inputs_with_largest_division_cluster = train_clustering_result[train_clustering_result.iloc[:, -1] == largest_division_indices]
    # get the number of columns in the inputs_with_largest_division_cluster
    num_of_rows = inputs_with_largest_division_cluster.shape[0]

    # half the number of columns
    num_of_rows_to_remove = [num_of_rows, int(num_of_rows / 2), int(num_of_rows / 4)]

    return num_of_rows_to_remove, largest_division_indices


def get_changed_datasets(rows, largest_division_indices, train_clustering_result, test_clustering_result):

    # Create a boolean mask for rows where the last column value equals largest_division_indices
    mask = train_clustering_result.iloc[:, -1] == largest_division_indices

    # Get the indices of the rows that match the mask
    indices = np.where(mask)[0]

    # Randomly select x rows to remove
    to_remove = np.random.choice(indices, size=rows, replace=False)
    filtered_df = train_clustering_result.drop(to_remove)

    # remove the rows which their last column value equal to largest_division_indices from test_clustering_result
    filtered_df = test_clustering_result[test_clustering_result.iloc[:, -1] != largest_division_indices]

    # get random of half_num_of_columns from filtered_df
    random_rows_from_filtered_df = filtered_df.sample(rows, random_state=42)

    # add random_rows_from_filtered_df to filtered_df
    return filtered_df.append(random_rows_from_filtered_df)


