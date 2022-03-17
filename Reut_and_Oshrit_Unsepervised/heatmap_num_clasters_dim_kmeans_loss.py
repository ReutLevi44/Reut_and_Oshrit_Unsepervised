import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import prince
import seaborn as sns

from clustering_algorithms import *


def plot_loss_vs_num_clusters(data_, save_data=False):
    num_clusters_list = list(range(2, 21))
    if save_data:
        loss_list = list()
        for i, num_clusters in enumerate(tqdm(num_clusters_list)):
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(data_)
            loss = kmeans.inertia_
            loss_list.append(loss)
            # loss_list.append(i)
        all_df = pd.DataFrame([loss_list], index=['K-means'], columns=num_clusters_list)
        all_df.to_csv(f'{out_dir}/kmeans_loss_vs_n_clusters_data.csv')

    all_df = pd.read_csv(f'{out_dir}/kmeans_loss_vs_n_clusters_data.csv', index_col=0)
    loss_list = list(all_df.loc['K-means'])
    plt.plot(num_clusters_list, loss_list, color='b', alpha=0.7)
    plt.xticks(np.arange(min(num_clusters_list), max(num_clusters_list) + 1, 2))

    plt.axvline(x=5, color='black', linestyle='--', linewidth=1)

    plt.xlabel('Number of clusters')
    plt.ylabel('K-means loss')
    plt.savefig(f'{out_dir}/kmeans_loss_vs_n_clusters.pdf', bbox_inches="tight", pad_inches=0.2)
    plt.close()


def loss_heatmap(save=False, heatmap=False, elbow=False):
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
                        mca = mca.fit(sampled_data)
                        mca = mca.transform(sampled_data)
                    kmeans = KMeans(n_clusters=num_clusters)
                    kmeans.fit(mca)
                    loss = kmeans.inertia_
                    all_cv_loss.append(loss)
                avg_over_cv = np.mean(all_cv_loss)
                df[dim][num_clusters] = avg_over_cv
                df.to_csv(f'{out_dir}/temp_loss_kmeans_heatmap_data_range_2.csv')

        df.to_csv(f'{out_dir}/loss_kmeans_heatmap_data_range_2.csv')

    if heatmap:
        df = pd.read_csv(f'{out_dir}/loss_kmeans_heatmap_data_range_2.csv', index_col=0)

        df = df.iloc[:, :-1]
        ax = sns.heatmap(df, cmap='viridis')
        plt.xlabel('Dimension')
        plt.ylabel('Number of clusters')
        save_dir = f'{out_dir}/Heatmap'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f'{save_dir}/heatmap_loss_kmeans_only_dimension_reduction_range_2.pdf', bbox_inches="tight", pad_inches=0.2)
        plt.close()

    if elbow:
        df = pd.read_csv(f'{out_dir}/loss_kmeans_heatmap_data_range_2.csv', index_col=0)
        dim = 10
        loss_list = list(df[str(dim)])
        plt.plot(num_of_clusters, loss_list, color='b', alpha=0.7)
        plt.xticks(np.arange(min(num_of_clusters), max(num_of_clusters) + 1, 2))

        plt.axvline(x=8, color='black', linestyle='--', linewidth=1)

        plt.xlabel('Number of clusters')
        plt.ylabel('K-means loss')
        save_dir = f'{out_dir}/Elbow'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f'{save_dir}/elbow_kmeans_loss_vs_n_clusters_dim={dim}_range_2.pdf', bbox_inches="tight", pad_inches=0.2)
        plt.close()


if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')

    num_of_samples = 20000

    out_dir = 'Figures/Number_of_clusters_one_hot'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    loss_heatmap(save=True, heatmap=True, elbow=False)
