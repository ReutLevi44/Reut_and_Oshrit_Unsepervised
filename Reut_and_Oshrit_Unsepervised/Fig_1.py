import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import prince
import seaborn as sns
from clustering_algorithms import *


mpl.rc('font', family='Times New Roman')

fig = plt.figure(figsize=(9, 10))

a1 = plt.subplot2grid((3, 2), (0, 0))
a2 = plt.subplot2grid((3, 2), (0, 1))
a3 = plt.subplot2grid((3, 2), (1, 0))
a4 = plt.subplot2grid((3, 2), (1, 1))
a5 = plt.subplot2grid((3, 2), (2, 0))
a6 = plt.subplot2grid((3, 2), (2, 1))


def loss_heatmap(color, ax, save=False, heatmap=False, elbow=False):
    num_of_clusters = list(range(2, 16))
    if save:
        data = pd.read_csv('one_hot_data.csv', index_col=0)
        dim_list = [2, 3, 4, 5, 6, 7, 8, 9]

        df = pd.DataFrame(index=num_of_clusters, columns=dim_list)

        for dim in dim_list:
            for num_clusters in num_of_clusters:
                all_cv_loss = list()
                for cv in range(10):
                    print(f'dim={dim}, num_clusters={num_clusters}, CV={cv}')
                    sampled_data = data.sample(num_of_samples)
                    if dim == 'without':
                        mca = sampled_data
                    else:
                        mca = prince.MCA(n_components=dim)
                        mca = mca.fit(sampled_data)  # same as calling ca.fs_r(1)
                        mca = mca.transform(sampled_data)  # same as calling ca.fs_r_sup(df_new) for *another* test set.
                    kmeans = KMeans(n_clusters=num_clusters)
                    kmeans.fit(mca)
                    loss = kmeans.inertia_
                    all_cv_loss.append(loss)
                avg_over_cv = np.mean(all_cv_loss)
                df[dim][num_clusters] = avg_over_cv
                df.to_csv(f'{out_dir}/temp_loss_kmeans_heatmap_data_range_2.csv')

        df.to_csv(f'{out_dir}/loss_kmeans_heatmap_data_range_2.csv')

    if heatmap:
        df = pd.read_csv(f'{out_dir}/loss_kmeans_heatmap_data.csv', index_col=0)

        df = df.iloc[:, :-1]
        sns.heatmap(df, cmap=color, ax=ax)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Number of clusters')

    if elbow:
        df = pd.read_csv(f'{out_dir}/loss_kmeans_heatmap_data.csv', index_col=0)
        dim = 10
        loss_list = list(df[str(dim)])
        ax.plot(num_of_clusters, loss_list, color='b', alpha=0.7)
        ax.set_xticks(np.arange(min(num_of_clusters), max(num_of_clusters) + 1, 2))

        ax.axvline(x=8, color='black', linestyle='--', linewidth=1)

        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('K-means loss')


def silhouette_heatmap(cluster_name, color, ax, heatmap=False):
    if heatmap:
        df = pd.read_csv(f'{out_dir}/silhouette_heatmap_data_{cluster_name}.csv', index_col=0)

        if 'dbscan' not in cluster_name:
            df = df.iloc[:, :-1]
        sns.heatmap(df, cmap=color, ax=ax)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Number of clusters')


num_of_samples = 20000

out_dir = 'Figures/Number_of_clusters_one_hot'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

loss_heatmap('Blues', ax=a1, save=False, heatmap=True, elbow=False)
loss_heatmap('Blues', ax=a2, save=False, heatmap=False, elbow=True)
silhouette_heatmap('kmeans', 'Blues', a3, heatmap=True)
silhouette_heatmap('hierarchical', 'Reds', a4, heatmap=True)
silhouette_heatmap('dbscan', 'Greens', a5, heatmap=True)
silhouette_heatmap('gmm', 'Greys', a6, heatmap=True)

fig.tight_layout(pad=2)

textstr = 'A'
plt.text(0.02, 0.99, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

textstr = 'B'
plt.text(0.51, 0.99, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

textstr = 'C'
plt.text(0.02, 0.67, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

textstr = 'D'
plt.text(0.51, 0.67, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

textstr = 'E'
plt.text(0.02, 0.33, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

textstr = 'F'
plt.text(0.51, 0.33, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

out_dir = 'Figures_for_Overleaf'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

plt.savefig(f'{out_dir}/Fig_1.pdf', bbox_inches="tight", pad_inches=0.2)

# plt.show()
