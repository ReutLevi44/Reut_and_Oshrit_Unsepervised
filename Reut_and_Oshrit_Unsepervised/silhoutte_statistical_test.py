import pandas as pd
from sklearn.metrics import silhouette_score
import os
import pickle
from clustering_algorithms import *


sample_size = 20000
runs = 50
n_clusters = 8
dim = 10
kmeans_sils = list()
hier_sils = list()
dbscan_sils = list()
gmm_sils = list()

out_dir = 'Figures/Silhouette_statistical_tests'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

labels_dirs = ['Labels_kmeans', 'Labels_hierarchical', 'Labels_dbscan', 'Data_dbscan', 'Indices_dbscan', 'Labels_gmm']
for specific_dir in labels_dirs:
    if not os.path.exists(specific_dir):
        os.makedirs(specific_dir)

print('Running Clustering Algorithms')
for i in range(runs):
    print(f'Run {i + 1} / {runs}')

    data_sample = pd.read_csv(f'MCA_dim={dim}/cv={i}.csv', index_col=0)

    dbscan_data, dbscan_labels, idx = dbscan(data_sample, n_clusters, eps=1)
    if dbscan_data is None:
        print('Cant find an epsilon that matches the required number of clusters')
        continue
    dbscan_silhouette = silhouette_score(dbscan_data, dbscan_labels)
    pickle.dump(dbscan_labels, open(f'Labels_dbscan/cv={i}', "wb"))
    pickle.dump(dbscan_data, open(f'Data_dbscan/cv={i}', "wb"))
    pickle.dump(idx, open(f'Indices_dbscan/cv={i}', "wb"))

    kmeans_labels = kmeans(data_sample, n_clusters)
    pickle.dump(kmeans_labels, open(f'Labels_kmeans/cv={i}', "wb"))
    kmeans_silhouette = silhouette_score(data_sample, kmeans_labels)
    hierarchical_labels = hierarchical_clustering(data_sample, n_clusters)
    pickle.dump(hierarchical_labels, open(f'Labels_hierarchical/cv={i}', "wb"))
    hierarchical_silhouette = silhouette_score(data_sample, hierarchical_labels)
    gmm_labels = GMM(data_sample, n=n_clusters)
    pickle.dump(gmm_labels, open(f'Labels_gmm/cv={i}', "wb"))
    gmm_silhouette = silhouette_score(data_sample, gmm_labels)

    kmeans_sils.append(kmeans_silhouette)
    hier_sils.append(hierarchical_silhouette)
    dbscan_sils.append(dbscan_silhouette)
    gmm_sils.append(gmm_silhouette)

    print(
        f'Avg Silhouette Scores - K-means: {kmeans_silhouette} Hierarchical: {hierarchical_silhouette} '
        f'DBSCAN: {dbscan_silhouette} GMM: {gmm_silhouette}')

    df = pd.DataFrame([kmeans_sils, hier_sils, dbscan_sils, gmm_sils],
                      index=['K-means', 'Hierarchical', 'DBSCAN', 'GMM'])
    df.to_csv(f'{out_dir}/temp_silhouette_data_clusters={n_clusters}_dim={dim}.csv')

df = pd.DataFrame([kmeans_sils, hier_sils, dbscan_sils, gmm_sils], index=['K-means', 'Hierarchical', 'DBSCAN', 'GMM'])
df.to_csv(f'{out_dir}/silhouette_data_clusters={n_clusters}_dim={dim}.csv')
