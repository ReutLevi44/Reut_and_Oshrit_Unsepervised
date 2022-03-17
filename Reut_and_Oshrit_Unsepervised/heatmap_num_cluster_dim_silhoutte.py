import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from sklearn.metrics import silhouette_score
import prince
import seaborn as sns

from clustering_algorithms import *


def plot_silhouette_vs_num_clusters(data_, save_data=False):
    num_clusters_list = list(range(2, 21))
    if save_data:
        for cv in range(10):
            sampled_data = data_.sample(num_of_samples)

            kmeans_silhouette_list = list()
            hierarchical_silhouette_list = list()
            dbscan_silhouette_list = list()
            for i, num_clusters in enumerate(tqdm(num_clusters_list)):
                kmeans_labels = kmeans(sampled_data, n_clusters=num_clusters)
                hierarchical_labels = hierarchical_clustering(sampled_data, n_clusters=num_clusters)
                dbscan_data, dbscan_labels, _ = dbscan(sampled_data, num_clusters, eps=11.5)
                if dbscan_data is None:
                    print('Can not find an epsilon that matches the required number of clusters')
                    dbscan_silhouette_list.append(np.nan)
                else:
                    dbscan_silhouette = silhouette_score(dbscan_data, dbscan_labels)
                    dbscan_silhouette_list.append(dbscan_silhouette)

                kmeans_silhouette = silhouette_score(sampled_data, kmeans_labels)
                hierarchical_silhouette = silhouette_score(sampled_data, hierarchical_labels)

                kmeans_silhouette_list.append(kmeans_silhouette)
                hierarchical_silhouette_list.append(hierarchical_silhouette)

                # kmeans_silhouette_list.append(i)
                # hierarchical_silhouette_list.append(i + 2)
            all_df = pd.DataFrame([kmeans_silhouette_list, hierarchical_silhouette_list, dbscan_silhouette_list],
                                  index=['K-means', 'Hierarchical', 'DBSCAN'], columns=num_clusters_list)
            all_df.to_csv(f'{out_dir}/silhouette_vs_n_clusters_data_n_sample={num_of_samples}_'
                          f'including_dbscan_CV={cv}.csv')

        dfs = list()
        for cv in range(10):
            file = pd.read_csv(f'{out_dir}/silhouette_vs_n_clusters_data_n_sample={num_of_samples}_'
                               f'including_dbscan_CV={cv}.csv', index_col=0)
            dfs.append(file)

        averages = pd.concat([each.stack() for each in dfs], axis=1) \
            .apply(lambda x: x.mean(), axis=1) \
            .unstack()

        averages.columns = [int(i) for i in averages.columns]
        sorted_cols = sorted(list(averages.columns))
        averages = averages[sorted_cols]

        averages.to_csv(f'{out_dir}/silhouette_vs_n_clusters_data_n_sample={num_of_samples}_'
                        f'mean_over_10_CV.csv')

    all_df = pd.read_csv(f'{out_dir}/silhouette_vs_n_clusters_data_n_sample={num_of_samples}_mean_over_10_CV.csv',
                         index_col=0)
    kmeans_silhouette_list = list(all_df.loc['K-means'])
    hierarchical_silhouette_list = list(all_df.loc['Hierarchical'])
    dbscan_silhouette_list = list(all_df.loc['DBSCAN'])
    plt.plot(num_clusters_list, kmeans_silhouette_list, 'o', color='b', alpha=0.7, label='K-means')
    plt.plot(num_clusters_list, hierarchical_silhouette_list, 'o', color='r', alpha=0.7, label='Hierarchical')
    plt.plot(num_clusters_list, dbscan_silhouette_list, 'o', color='g', alpha=0.7, label='DBSCAN')
    plt.xticks(np.arange(min(num_clusters_list), max(num_clusters_list) + 1, 2))

    plt.axvline(x=num_clusters_list[np.argmax(kmeans_silhouette_list)], color='black', linestyle='--', linewidth=1)

    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.legend()
    plt.savefig(f'{out_dir}/silhouette_vs_n_clusters_n_sample={num_of_samples}_mean_CV_scatter_plot.pdf', bbox_inches="tight",
                pad_inches=0.2)
    plt.close()


def silhouette_heatmap(data_, save=False, heatmap=False):
    if save:
        dim_list = [10, 50, 100, 200, 300, 'without']
        num_of_clusters = list(range(2, 16))

        df_kmeans = pd.DataFrame(index=num_of_clusters, columns=dim_list)
        df_hierarchical = pd.DataFrame(index=num_of_clusters, columns=dim_list)
        # df_dbscan = pd.DataFrame(index=num_of_clusters, columns=dim_list)
        df_gmm = pd.DataFrame(index=num_of_clusters, columns=dim_list)

        for dim in dim_list:
            for num_clusters in num_of_clusters:
                all_cv_kmeans_silhouette = list()
                all_cv_hierarchical_silhouette = list()
                # all_cv_dbscan_silhouette = list()
                all_cv_gmm_silhouette = list()
                for cv in range(10):
                    print(f'dim={dim}, num_clusters={num_clusters}, CV={cv}')
                    sampled_data = data_.sample(num_of_samples)
                    if dim == 'without':
                        mca = sampled_data
                    else:
                        mca = prince.MCA(n_components=dim)
                        mca = mca.fit(sampled_data)  # same as calling ca.fs_r(1)
                        mca = mca.transform(sampled_data)  # same as calling ca.fs_r_sup(df_new) for *another* test set.

                    kmeans_labels = kmeans(mca, n_clusters=num_clusters)
                    hierarchical_labels = hierarchical_clustering(mca, n_clusters=num_clusters)
                    gmm_labels = GMM(mca, n=num_clusters)
                    # dbscan_data, dbscan_labels, _ = dbscan(mca, num_clusters, eps=11.5)
                    # if dbscan_data is None:
                    #     print('Can not find an epsilon that matches the required number of clusters')
                    #     all_cv_dbscan_silhouette.append(np.nan)
                    # else:
                    #     dbscan_silhouette = silhouette_score(dbscan_data, dbscan_labels)
                    #     all_cv_dbscan_silhouette.append(dbscan_silhouette)

                    kmeans_silhouette = silhouette_score(mca, kmeans_labels)
                    hierarchical_silhouette = silhouette_score(mca, hierarchical_labels)
                    gmm_silhouette = silhouette_score(mca, gmm_labels)

                    all_cv_kmeans_silhouette.append(kmeans_silhouette)
                    all_cv_hierarchical_silhouette.append(hierarchical_silhouette)
                    all_cv_gmm_silhouette.append(gmm_silhouette)

                avg_over_cv_kmeans = np.nanmean(all_cv_kmeans_silhouette)
                avg_over_cv_hierarchical = np.nanmean(all_cv_hierarchical_silhouette)
                # avg_over_cv_dbscan = np.nanmean(all_cv_dbscan_silhouette)
                avg_over_cv_gmm = np.nanmean(all_cv_gmm_silhouette)

                df_kmeans[dim][num_clusters] = avg_over_cv_kmeans
                df_hierarchical[dim][num_clusters] = avg_over_cv_hierarchical
                # df_dbscan[dim][num_clusters] = avg_over_cv_dbscan
                df_gmm[dim][num_clusters] = avg_over_cv_gmm

                df_kmeans.to_csv(f'{out_dir}/temp_silhouette_heatmap_data_kmeans.csv')
                df_hierarchical.to_csv(f'{out_dir}/temp_silhouette_heatmap_data_hierarchical.csv')
                # df_dbscan.to_csv(f'{out_dir}/silhouette_heatmap_data_dbscan.csv')
                df_gmm.to_csv(f'{out_dir}/temp_silhouette_heatmap_data_gmm.csv')

        df_kmeans.to_csv(f'{out_dir}/silhouette_heatmap_data_kmeans.csv')
        df_hierarchical.to_csv(f'{out_dir}/silhouette_heatmap_data_hierarchical.csv')
        # df_dbscan.to_csv(f'{out_dir}/silhouette_heatmap_data_dbscan.csv')
        df_gmm.to_csv(f'{out_dir}/silhouette_heatmap_data_gmm.csv')

    if heatmap:
        df = pd.read_csv(f'{out_dir}/silhouette_heatmap_data_hierarchical.csv', index_col=0)

        df = df.iloc[:, :-1]
        ax = sns.heatmap(df, cmap='viridis')
        plt.xlabel('Dimension')
        plt.ylabel('Number of clusters')
        save_dir = f'{out_dir}/Heatmap'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f'{save_dir}/heatmap_silhouette_data_hierarchical.pdf', bbox_inches="tight", pad_inches=0.2)
        # plt.show()
        plt.close()


if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')

    num_of_samples = 20000

    data = pd.read_csv('one_hot_data.csv', index_col=0)

    out_dir = 'Figures/Number_of_clusters_one_hot'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    silhouette_heatmap(data, save=False, heatmap=True)
