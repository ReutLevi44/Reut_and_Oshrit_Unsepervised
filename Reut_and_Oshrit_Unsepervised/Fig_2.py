import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import sem
from sklearn.metrics import silhouette_score, silhouette_samples
import pickle
import matplotlib.cm as cm
from sklearn.metrics import mutual_info_score
from scipy.stats import f_oneway
from scipy.stats import ttest_rel

mpl.rc('font', family='Times New Roman')


def plot_silhouette_score(ax):
    colors_list = ['b', 'r', 'g', 'black']

    n_clusters = 8
    dim = 10

    out_dir = 'Figures/Silhouette_statistical_tests'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(f'{out_dir}/silhouette_data_clusters={n_clusters}_dim={dim}.csv', index_col=0)
    df_gmm = pd.read_csv(f'{out_dir}/silhouette_data_clusters={2}_dim={dim}_gmm.csv', index_col=0)
    df_dbscan = pd.read_csv(f'{out_dir}/silhouette_data_clusters={2}_dim={200}.csv', index_col=0)

    df_gmm.drop(['2', '17'], inplace=True, axis=1)
    df_dbscan.drop(['14', '28'], inplace=True, axis=1)

    df.loc['GMM'] = list(df_gmm.loc['GMM'])
    df.loc['DBSCAN'] = list(df_dbscan.loc['DBSCAN'])

    df.to_csv(f'{out_dir}/all_silhouette_scores_for_our_calculation.csv')

    mean_list = list()
    std_list = list()
    mean_names = list(df.index)
    for row in df.index:
        mean_list.append(df.loc[row].mean())
        std_list.append(sem(df.loc[row]))

    ax.bar(mean_names, mean_list, yerr=std_list, align='center', ecolor='black', capsize=10, alpha=0.6,
           color=colors_list)
    ax.set_ylabel('Silhouette score')


def plot_mi_score_best_params(ax):
    runs = 50
    # n_clusters = 8
    dim = 10
    external_vars_list = ['dAge', 'dHispanic', 'iYearwrk', 'iSex']
    clustering_list = ['kmeans', 'hierarchical', 'dbscan', 'gmm']
    names_clusters = ['K-means', 'Hierarchical', 'DBSCAN', 'GMM']
    colors_list = ['b', 'r', 'g', 'black']

    out_dir = 'Figures/Mutual_information'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mi_df = pd.DataFrame(columns=names_clusters, index=external_vars_list)
    std_mi_df = pd.DataFrame(columns=names_clusters, index=external_vars_list)
    scoring_df = pd.DataFrame(columns=[i for i in range(48)], index=names_clusters)
    dAge_dict = dict()
    dHispanic_dict = dict()
    iYearwrk_dict = dict()
    iSex_dict = dict()
    for idx_color, clustering in enumerate(clustering_list):
        MI = dict()
        for var in external_vars_list:
            MI[var] = list()
        for i in range(runs):
            try:
                dim = 10
                if clustering == 'dbscan':
                    dim = 200
                    num_of_clusters = '_n_clusters=2'
                elif clustering == 'gmm':
                    num_of_clusters = '_n_clusters=2'
                else:
                    num_of_clusters = ''
                external_variables = pd.read_csv(f'External_dim={dim}/cv={i}.csv', index_col=0)
                external_sample = external_variables

                labels = pickle.load(open(f'Labels_{clustering}/cv={i}{num_of_clusters}', "rb"))
                if clustering == 'dbscan':
                    dbscan_data = pickle.load(open(f'Data_dbscan/cv={i}{num_of_clusters}', "rb"))
                    idxs = pickle.load(open(f'Indices_dbscan/cv={i}{num_of_clusters}', "rb"))
                    external_sample = external_variables.iloc[dbscan_data.index]

                for var in external_vars_list:
                    mutual_info = mutual_info_score(external_sample[var].to_numpy(), labels)
                    MI[var].append(mutual_info)
                    print(f'{clustering}: External Variable {var} Mutual Information {mutual_info}')

            except:
                pass

        print('ANOVA: ', f_oneway(MI['dAge'], MI['dHispanic'], MI['iYearwrk'], MI['iSex']))
        print('T-test (dAge, iYearwrk):', ttest_rel(MI['dAge'], MI['iYearwrk']))

        dAge_dict[names_clusters[idx_color]] = MI['dAge']
        dHispanic_dict[names_clusters[idx_color]] = MI['dHispanic']
        iYearwrk_dict[names_clusters[idx_color]] = MI['iYearwrk']
        iSex_dict[names_clusters[idx_color]] = MI['iSex']

        if clustering == 'dbscan' or clustering == 'gmm':
            scoring_df.loc[names_clusters[idx_color]] = MI['iYearwrk'][:-2]
        else:
            scoring_df.loc[names_clusters[idx_color]] = MI['iYearwrk']

        means = []
        stds = []
        for var in external_vars_list:
            means.append(np.mean(MI[var]))
            stds.append(sem(MI[var]))

            mi_df[names_clusters[idx_color]][var] = np.mean(MI[var])
            std_mi_df[names_clusters[idx_color]][var] = sem(MI[var])

    print('dAge: ANOVA: ', f_oneway(dAge_dict['K-means'], dAge_dict['Hierarchical'],
                                    dAge_dict['DBSCAN'], dAge_dict['GMM']))
    print('dAge: T-test (K-means, Hierarchical):', ttest_rel(dAge_dict['K-means'], dAge_dict['Hierarchical']))
    print('dHispanic: ANOVA: ', f_oneway(dHispanic_dict['K-means'], dHispanic_dict['Hierarchical'],
                                         dHispanic_dict['DBSCAN'], dHispanic_dict['GMM']))
    print('dHispanic: T-test (K-means, Hierarchical):',
          ttest_rel(dHispanic_dict['K-means'], dHispanic_dict['Hierarchical']))
    print('iYearwrk: ANOVA: ', f_oneway(iYearwrk_dict['K-means'], iYearwrk_dict['Hierarchical'],
                                        iYearwrk_dict['DBSCAN'], iYearwrk_dict['GMM']))
    print('iYearwrk: T-test (K-means, Hierarchical):',
          ttest_rel(iYearwrk_dict['K-means'], iYearwrk_dict['Hierarchical']))
    print('iSex: ANOVA: ', f_oneway(iSex_dict['K-means'], iSex_dict['Hierarchical'],
                                    iSex_dict['DBSCAN'], iSex_dict['GMM']))
    print('iSex: T-test (K-means, Hierarchical):', ttest_rel(iSex_dict['K-means'], iSex_dict['Hierarchical']))
    mi_df.plot.bar(align='center', alpha=0.6, ecolor='black', capsize=3, color=colors_list, yerr=std_mi_df, ax=ax)
    scoring_df.to_csv(f'{out_dir}/all_mi_scores_for_our_calculation_50.csv')
    ax.set_ylabel('MI')


def plot_weighted_score(ax):
    colors_list = ['b', 'r', 'g', 'black']

    # n_clusters = 8
    # dim = 10

    out_dir = 'Figures/Average_score'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    df = pd.read_csv(f'Figures/Silhouette_statistical_tests/all_silhouette_scores_for_our_calculation.csv', index_col=0)
    mi_df = pd.read_csv(f'Figures/Mutual_information/all_mi_scores_for_our_calculation_50.csv', index_col=0)

    avg_df = 0.5 * df + 0.5 * mi_df

    mean_list = list()
    std_list = list()
    mean_names = list(avg_df.index)
    for row in avg_df.index:
        mean_list.append(avg_df.loc[row].mean())
        std_list.append(sem(avg_df.loc[row]))

    ax.bar(mean_names, mean_list, yerr=std_list, align='center', ecolor='black', capsize=10, alpha=0.6,
           color=colors_list)
    ax.set_ylabel('Weighted score')


def plot_silhouette_itself(ax):
    data = pd.read_csv('MCA_dim=10/cv=0.csv', index_col=0)
    labels_kmeans = pickle.load(open(f'Labels_kmeans/cv=0', "rb"))

    out_dir = 'Figures/Silhouette_statistical_tests'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    n_clusters = 8

    silhouette_avg = silhouette_score(data, labels_kmeans)
    print(f'Avg Silhouette Score: {silhouette_avg}')
    sample_silhouette_values = silhouette_samples(data, labels_kmeans)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[np.array(labels_kmeans) == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                         edgecolor=color, alpha=0.6)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_xlabel("Silhouette score")
    ax.set_ylabel("Cluster index")
    ax.set_yticks([])


def plot_mi_score_best_params_real_num_clusters(ax):
    runs = 50
    # n_clusters = 8
    dim = 10
    external_vars_list = ['dAge', 'dHispanic', 'iYearwrk', 'iSex']
    clustering_list = ['kmeans', 'hierarchical', 'dbscan', 'gmm']
    names_clusters = ['K-means', 'Hierarchical', 'DBSCAN', 'GMM']
    colors_list = ['b', 'r', 'g', 'black']

    out_dir = 'Figures/Mutual_information'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    mi_df = pd.DataFrame(columns=names_clusters, index=external_vars_list)
    std_mi_df = pd.DataFrame(columns=names_clusters, index=external_vars_list)
    scoring_df = pd.DataFrame(columns=[i for i in range(48)], index=names_clusters)
    for idx_color, clustering in enumerate(clustering_list):
        MI = dict()
        for var in external_vars_list:
            MI[var] = list()
        for i in range(runs):
            for var in external_vars_list:
                try:
                    dim = 10
                    if clustering == 'dbscan':
                        dim = 200
                    if var == 'dAge':
                        num_of_clusters = ''
                    elif var == 'dHispanic':
                        num_of_clusters = '_n_clusters=10'
                    elif var == 'iYearwrk':
                        num_of_clusters = ''
                    elif var == 'iSex':
                        num_of_clusters = '_n_clusters=2'
                    external_variables = pd.read_csv(f'External_dim={dim}/cv={i}.csv', index_col=0)
                    external_sample = external_variables

                    labels = pickle.load(open(f'Labels_{clustering}/cv={i}{num_of_clusters}', "rb"))
                    if clustering == 'dbscan':
                        dbscan_data = pickle.load(open(f'Data_dbscan/cv={i}{num_of_clusters}', "rb"))
                        idxs = pickle.load(open(f'Indices_dbscan/cv={i}{num_of_clusters}', "rb"))
                        external_sample = external_variables.iloc[dbscan_data.index]

                    mutual_info = mutual_info_score(external_sample[var].to_numpy(), labels)
                    MI[var].append(mutual_info)
                    print(f'{clustering}: External Variable {var} Mutual Information {mutual_info}')

                except:
                    pass

        print(f'{clustering}: ANOVA: ', f_oneway(MI['dAge'], MI['dHispanic'], MI['iYearwrk'], MI['iSex']))
        print(f'{clustering}: T-test (dAge, iYearwrk):', ttest_rel(MI['dAge'], MI['iYearwrk']))

        if clustering == 'dbscan' or clustering == 'gmm':
            scoring_df.loc[names_clusters[idx_color]] = MI['iYearwrk']
        else:
            scoring_df.loc[names_clusters[idx_color]] = MI['iYearwrk']

        means = []
        stds = []
        for var in external_vars_list:
            means.append(np.mean(MI[var]))
            stds.append(sem(MI[var]))

            mi_df[names_clusters[idx_color]][var] = np.mean(MI[var])
            std_mi_df[names_clusters[idx_color]][var] = sem(MI[var])

    mi_df.plot.bar(align='center', alpha=0.6, ecolor='black', capsize=3, color=colors_list, yerr=std_mi_df, ax=ax)
    scoring_df.to_csv(f'{out_dir}/all_mi_scores_for_our_calculation_50_real_clusters.csv')
    ax.set_ylabel('MI')


if __name__ == '__main__':
    fig = plt.figure(figsize=(8, 11))
    a1 = plt.subplot2grid((3, 2), (0, 0))
    a2 = plt.subplot2grid((3, 2), (0, 1))
    a3 = plt.subplot2grid((3, 2), (1, 0))
    a4 = plt.subplot2grid((3, 2), (1, 1))
    a5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

    plot_silhouette_score(a1)
    plot_mi_score_best_params(a2)
    plot_weighted_score(a3)
    plot_silhouette_itself(a4)
    plot_mi_score_best_params_real_num_clusters(a5)

    fig.tight_layout(pad=2)

    textstr = 'A'
    plt.text(0.02, 0.99, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

    textstr = 'B'
    plt.text(0.51, 0.99, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

    textstr = 'C'
    plt.text(0.02, 0.66, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

    textstr = 'D'
    plt.text(0.51, 0.66, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

    textstr = 'E'
    plt.text(0.02, 0.34, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

    out_dir = 'Figures_for_Overleaf'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(f'{out_dir}/Fig_2_updated.pdf', bbox_inches="tight", pad_inches=0.2)

    # plt.show()
