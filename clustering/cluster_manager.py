import pandas as pd
from sklearn.model_selection import train_test_split


class ClusterManager:

    def __init__(self, data: pd.DataFrame, cluster_number: int, train_test_percentage: float):
        self.data = data
        self.cluster_number = cluster_number
        self.train_test_percentage = train_test_percentage

        self.train_data, self.test_data = train_test_split(self.data, test_size=self.train_test_percentage,
                                                           random_state=42)
        # get all the rows of the cluster
        self.cluster_data = self.train_data[self.data.iloc[:, -1] == self.cluster_number]
        self.cluster_data_size = len(self.cluster_data)

        # get all the rest of the rows
        self.rest_data = self.train_data[self.data.iloc[:, -1] != self.cluster_number]
        # check if the rest of the data is smaller than the cluster size
        if len(self.rest_data) < self.cluster_data_size:
            # if so, sample random rows from the cluster to match the size of the rest of the data
            self.cluster_data = self.cluster_data.sample(n=len(self.rest_data), random_state=42)
            self.cluster_data_size = len(self.cluster_data)

        # sample random rows from the rest of the data to match the size of the cluster
        self.sub_data = self.rest_data.sample(n=self.cluster_data_size, random_state=42)
        # get the rest of the rows
        self.rest_data = self.rest_data.drop(self.sub_data.index)

    def get_train_data(self, cluster_percentage: float):
        # sample random rows from the cluster to match the cluster_percentage
        cluster_data = self.cluster_data.sample(frac=cluster_percentage, random_state=42)
        # sample random rows from the sub_data to complete the cluster size
        sub_data = self.sub_data.sample(frac=1-cluster_percentage, random_state=42)
        # concat the sampled cluster data with the rest of the data
        data = pd.concat([cluster_data, sub_data, self.rest_data], axis=0, ignore_index=True)
        # shuffle the data
        data = data.sample(frac=1, random_state=42)
        # drop the last column (cluster number)
        data = data.drop(data.columns[-1], axis=1)
        return data.reset_index(drop=True)

    def get_train_data_complete(self):
        return self.train_data.reset_index(drop=True)

    def get_test_data(self):
        return self.test_data.reset_index(drop=True)

