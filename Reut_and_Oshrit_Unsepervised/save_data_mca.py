import pandas as pd
import os
import prince
from tqdm import tqdm


dim = 10  # 200
num_of_samples = 20000
CV = 50

mca_dir = f'MCA_dim={dim}'
if not os.path.exists(mca_dir):
    os.makedirs(mca_dir)

ex_dir = f'External_dim={dim}'
if not os.path.exists(ex_dir):
    os.makedirs(ex_dir)

data = pd.read_csv('one_hot_data.csv', index_col=0)
columns_data = data.columns
external = pd.read_csv('external_data.csv', index_col=0)
data = pd.concat([data, external], axis=1)

for cv in tqdm(range(CV)):
    sampled_together = data.sample(num_of_samples)
    sampled_data = sampled_together[columns_data]
    sampled_external = sampled_together[external.columns]

    mca = prince.MCA(n_components=dim)
    mca = mca.fit(sampled_data)  # same as calling ca.fs_r(1)
    mca = mca.transform(sampled_data)  # same as calling ca.fs_r_sup(df_new) for *another* test set.

    mca.to_csv(f'{mca_dir}/cv={cv}.csv')
    sampled_external.to_csv(f'{ex_dir}/cv={cv}.csv')
