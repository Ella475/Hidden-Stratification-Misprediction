import numpy as np


# get features and labels
def get_clustering_data(model, dataloader):
    agg_inputs = []
    agg_outputs = []
    agg_labels = []
    agg_features = []
    for i, data in enumerate(dataloader):
        inputs, labels = data
        outputs = model(inputs)
        features = model.get_embedding()

        agg_inputs.append(inputs.detach().numpy())
        agg_outputs.append(outputs.detach().numpy())
        agg_labels.append(labels.detach().numpy())
        agg_features.append(features.detach().numpy())

    # create one big array
    agg_inputs = np.concatenate(agg_inputs)
    agg_outputs = np.concatenate(agg_outputs)
    agg_labels = np.concatenate(agg_labels)
    agg_features = np.concatenate(agg_features)

    return agg_inputs, agg_outputs, agg_labels, agg_features
