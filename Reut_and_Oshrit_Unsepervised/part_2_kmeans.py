import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import os
import pickle


def hist_plot(scores, bins_num=30):
    plt.hist(scores, bins=bins_num, color='b', alpha=0.5, edgecolor='black')

    thresh = np.mean(scores) + 3 * np.std(scores)
    plt.axvline(x=thresh, color='black', linestyle='--', linewidth=1.5)

    print(f'Threshold: {thresh}')
    print(f'The anomalies percent for K-means is {100 * len([i for i in scores if i > thresh]) / len(scores)}%')

    plt.xlabel('Distance score')
    plt.ylabel('Frequency')
    plt.savefig(f'{out_dir}/histogram_anomalies_kmeans.pdf', bbox_inches="tight", pad_inches=0.2)
    plt.close()


if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')

    out_dir = f'Figures/Anomalies'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save = True
    plot = True

    if save:

        num_of_samples = 20000
        dim = 10

        scores_cv_list = list()

        for i in tqdm(range(50)):
            sampled_data = pd.read_csv(f'MCA_dim={dim}/cv={i}.csv', index_col=0)

            kmeans = KMeans(n_clusters=8)
            kmeans.fit(sampled_data)
            score_kmeans = kmeans.transform(sampled_data).min(axis=1)

            scores_cv_list += list(score_kmeans)

        pickle.dump(scores_cv_list, open(f'{out_dir}/scores_all_CV_list_kmeans', "wb"))

    if plot:
        scores_cv_list = pickle.load(open(f'{out_dir}/scores_all_CV_list_kmeans', "rb"))
        hist_plot(scores_cv_list, bins_num=50)

    print('')
