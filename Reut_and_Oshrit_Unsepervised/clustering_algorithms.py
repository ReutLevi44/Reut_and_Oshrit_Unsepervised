import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


def kmeans(X, n_clusters=10):
    kmeans_ = KMeans(n_clusters=n_clusters)
    kmeans_.fit(X)
    labels = kmeans_.labels_
    return labels


def GMM(X, n):
    gm = GaussianMixture(n_components=n)
    labels = gm.fit_predict(X)
    return labels


def hierarchical_clustering(X, n_clusters=10):
    hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    hc.fit_predict(X)
    labels = hc.labels_
    return labels


# def dbscan(X, eps=10):
#     dbs = DBSCAN(eps=eps)
#     dbs.fit(X)
#     labels = list(dbs.labels_)
#     # print('dbscan labels', labels)
#     dbscan_X = list(X.copy())
#     idxs = []
#     for idx, label in enumerate(labels):
#         if label != -1:
#             idxs.append(idx)
#     dbscan_X = [dbscan_X[i] for i in idxs]
#     labels = [labels[i] for i in idxs]
#     return np.array(dbscan_X), np.array(labels), idxs


def dbscan(X, n, eps=5, prev=-1, step=0.3, counter=0):
    dbs = DBSCAN(eps=eps)
    dbs.fit(X)
    labels = list(dbs.labels_)
    max_labels = max(labels)
    if prev != -1 and abs(prev - max_labels) > 1 and (prev - n + 1) * (max_labels - n + 1) < 0:
        step = step / 2
    if counter >= 20:  # to change
        return None, None, None
    if max(labels) > n - 1:
        print(f'labels : {max(labels) + 1}')
        return dbscan(X, n, eps=eps + step, prev=max(labels), step=step, counter=counter + 1)
    if max(labels) < n - 1:
        print(f'labels : {max(labels) + 1}')
        return dbscan(X, n, eps=eps - step, prev=max(labels), step=step, counter=counter + 1)
    print(f'correct labels : {max(labels) + 1}')
    dbscan_X = X.copy()
    idxs = []
    for idx, label in enumerate(labels):
        if label != -1:
            idxs.append(idx)
    dbscan_X = dbscan_X.iloc[idxs]
    labels = [labels[i] for i in idxs]
    return dbscan_X, labels, idxs
