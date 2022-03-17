import pandas as pd
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import os
import pickle


def hist_plot(scores, bins_num=30):
    plt.hist(scores, bins=bins_num, color='goldenrod', alpha=0.7, edgecolor='black')

    plt.xlabel('Distance score')
    plt.ylabel('Frequency')
    plt.savefig(f'{out_dir}/histogram_anomalies_binary_one_class_svm.pdf', bbox_inches="tight", pad_inches=0.2)
    plt.close()


if __name__ == '__main__':
    mpl.rc('font', family='Times New Roman')

    out_dir = f'Figures/Anomalies'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    save = False
    plot = True

    if save:

        num_of_samples = 20000
        dim = 10

        scores_cv_list = list()
        scores_binary_cv_list = list()

        for i in tqdm(range(50)):
            sampled_data = pd.read_csv(f'MCA_dim={dim}/cv={i}.csv', index_col=0)

            clf = OneClassSVM(gamma='auto')
            clf.fit(sampled_data)
            score_1classSVM = clf.score_samples(sampled_data)
            score_1classSVM_binary = clf.predict(sampled_data)

            # scores_cv_list.append(list(score_gm))
            scores_cv_list += list(score_1classSVM)
            scores_binary_cv_list += list(score_1classSVM_binary)

        pickle.dump(scores_cv_list, open(f'{out_dir}/scores_all_CV_list_one_class_svm', "wb"))
        pickle.dump(scores_binary_cv_list, open(f'{out_dir}/scores_binary_all_CV_list_one_class_svm', "wb"))

    if plot:
        scores_cv_list = pickle.load(open(f'{out_dir}/scores_binary_all_CV_list_one_class_svm', "rb"))
        hist_plot(scores_cv_list, bins_num=50)

    print('')
