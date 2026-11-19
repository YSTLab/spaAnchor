import numpy as np
from tqdm import tqdm
from .mapping import find_similar_pairs


def KNN_regression(impute_result, used_features, used_pairs = 200):

    batch_size = 50
    edge_index = find_similar_pairs(used_features, used_features, used_pairs, 1)

    return np.vstack([
        np.mean(impute_result[edge_index[1][
            used_pairs*batch_size*i: 
            used_pairs*min(impute_result.shape[0], batch_size*(i+1))
        ].reshape(-1, used_pairs)], axis=1)
        for i in tqdm(range(int(np.ceil(impute_result.shape[0]/batch_size))))
    ])
