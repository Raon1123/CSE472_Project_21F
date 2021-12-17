import numpy as np

from sklearn.cluster import KMeans
from sklearn.feature_extraction import image

"""
Train bag of words cluster.
Default method is KMeans algorithm.
For fast train, cluster select randomly num_select images from data.
Then generate num_dict of words.
- cluster: trained cluster of words
"""
def train_bow(trainX, cluster="KMeans", num_dict=1000, num_select=10000, patch_size=(4,4)):
    N = trainX.shape[0]
    
    if cluster == "KMeans":
        cluster_model = KMeans(n_clusters=num_dict, random_state=710)
    else:
        raise Exception("Wrong cluster model.")
    
    # Divide image into patches
    pe = image.PatchExtractor(patch_size=patch_size)
    patches = pe.transform(trainX)

    # Select random `num_select` samples
    select_X_idx = np.random.choice(N, num_select, replace=False)
    select_patches = patches[select_X_idx]

    # Train cluster
    select_patches_flat = select_patches.reshape((num_select, -1))
    cluster = cluster_model.fit(select_patches_flat)

    return cluster


"""
Get bag of words histogram
- ret: (number of data, number of feature)
"""
def get_bow(X, cluster, patch_size=(4, 4), num_dict=1000):
    N = X.shape[0]

    # divide image into patches
    pe = image.PatchExtractor(patch_size=patch_size)
    patches = pe.transform(X)

    # prediction by cluster
    patches_flat = patches.reshape((patches.shape[0], -1))
    pred = cluster.predict(patches_flat).reshape((N,-1))

    # wrapup clusters
    ret = np.zeros((N, num_dict))

    for idx in range(N):
        one_pred = pred[idx]
        val, cnt = np.unique(one_pred, return_counts=True)
        for dict_label, dict_val in zip(val, cnt):
            ret[idx][dict_label] = dict_val

    return ret