import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import pickle


def hist_plot_kmeans(scores, ax, bins_num=30):
    ax.hist(scores, bins=bins_num, color='b', alpha=0.5, edgecolor='black')

    thresh = np.mean(scores) + 3 * np.std(scores)
    ax.axvline(x=thresh, color='black', linestyle='--', linewidth=1.5)

    print(f'Threshold: {thresh}')
    print(f'The anomalies percent for K-means is {100 * len([i for i in scores if i > thresh]) / len(scores)}%')

    ax.set_xlabel('Distance score')
    ax.set_ylabel('Frequency')


def hist_plot_gmm(scores, ax, bins_num=30):
    ax.hist(scores, edgecolor='black', bins=bins_num, color='black', alpha=0.5)

    thresh = np.mean(scores) - 3 * np.std(scores)
    ax.axvline(x=thresh, color='black', linestyle='--', linewidth=1.5)

    print(f'Threshold: {thresh}')
    print(f'The anomalies percent for GMM is {100 * len([i for i in scores if i < thresh]) / len(scores)}%')

    ax.set_xlabel('Log-likelihood score')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')


def hist_plot_binary_one_class_svm(scores,ax, bins_num=30):
    ax.hist(scores, bins=bins_num, color='goldenrod', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Distance score')
    ax.set_ylabel('Frequency')


def hist_plot_one_class_svm(scores, ax, bins_num=30):
    ax.hist(scores, bins=bins_num, color='olive', alpha=0.7, edgecolor='black')

    thresh = np.mean(scores) + 1 * np.std(scores)
    ax.axvline(x=thresh, color='black', linestyle='--', linewidth=1.5)

    print(f'Threshold: {thresh}')
    print(f'The anomalies percent for one class SVM is {100 * len([i for i in scores if i > thresh]) / len(scores)}%')

    ax.set_xlabel('Distance score')
    ax.set_ylabel('Frequency')


def load_binary_scores(df, algo_name):
    if algo_name != 'binary_one_class_svm':
        scores_cv_list = pickle.load(open(f'{out_dir}/scores_all_CV_list_{algo_name}', "rb"))
    if algo_name == 'kmeans':
        thresh = 0.6669590799155976
        binary_score_cv_list = [1 if i > thresh else -1 for i in scores_cv_list]
    elif algo_name == 'gmm':
        thresh = -6.06433766820488
        binary_score_cv_list = [1 if i < thresh else -1 for i in scores_cv_list]
    elif algo_name == 'binary_one_class_svm':
        algo_name = '_'.join(algo_name.split('_')[1:])
        binary_score_cv_list = pickle.load(open(f'{out_dir}/scores_binary_all_CV_list_{algo_name}', "rb"))
    elif algo_name == 'one_class_svm':
        thresh = 9554.257115693796
        binary_score_cv_list = [1 if i > thresh else -1 for i in scores_cv_list]

    for var in external_vars_list:
        mutual_info = mutual_info_score(df[var].to_numpy(), binary_score_cv_list)
        MI[var].append(mutual_info)
        print(f'{algo_name}: external variable {var}, mutual information {mutual_info}')


def plot_mi(df, ax):
    dim = 10

    for cv in range(50):
        external_variables = pd.read_csv(f'External_dim={dim}/cv={cv}.csv', index_col=0)

        df = pd.concat([df, external_variables])

    for var in external_vars_list:
        MI[var] = list()

    algo_list = ['K-means', 'GMM', 'Binary one class SVM', 'One class SVM']

    load_binary_scores(df, 'kmeans')
    load_binary_scores(df, 'gmm')
    load_binary_scores(df, 'binary_one_class_svm')
    load_binary_scores(df, 'one_class_svm')

    df = pd.DataFrame(MI, index=algo_list)

    df.T.plot.bar(alpha=0.6, color=['b', 'black', 'goldenrod', 'olive'], ax=ax)
    ax.set_yscale('log')
    ax.set_ylabel('MI')


if __name__ == '__main__':

    mpl.rc('font', family='Times New Roman')

    fig = plt.figure(figsize=(8, 10))

    a1 = plt.subplot2grid((3, 2), (0, 0))
    a2 = plt.subplot2grid((3, 2), (0, 1))
    a3 = plt.subplot2grid((3, 2), (1, 0))
    a4 = plt.subplot2grid((3, 2), (1, 1))
    a5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)

    out_dir = f'Figures/Anomalies'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    scores_cv_list = pickle.load(open(f'{out_dir}/scores_all_CV_list_kmeans', "rb"))
    hist_plot_kmeans(scores_cv_list, a1, bins_num=50)
    scores_cv_list = pickle.load(open(f'{out_dir}/scores_all_CV_list_gmm', "rb"))
    hist_plot_gmm(scores_cv_list, a2, bins_num=50)
    scores_cv_list = pickle.load(open(f'{out_dir}/scores_binary_all_CV_list_one_class_svm', "rb"))
    hist_plot_binary_one_class_svm(scores_cv_list,a3, bins_num=30)
    scores_cv_list = pickle.load(open(f'{out_dir}/scores_all_CV_list_one_class_svm', "rb"))
    hist_plot_one_class_svm(scores_cv_list, a4, bins_num=50)

    external_vars_list = ['dAge', 'dHispanic', 'iYearwrk', 'iSex']
    MI = dict()
    df = pd.DataFrame()
    plot_mi(df, a5)

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
    plt.text(0.02, 0.36, textstr, fontsize=18, fontweight='bold', transform=plt.gcf().transFigure)

    out_dir = 'Figures_for_Overleaf'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(f'{out_dir}/Fig_3.pdf', bbox_inches="tight", pad_inches=0.2)

    # plt.show()
