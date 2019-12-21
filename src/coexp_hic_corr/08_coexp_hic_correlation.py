import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--chr-src', type=int, default=1)
    parser.add_argument('--chr-tgt', type=int, default=2)
    args = parser.parse_args()

    coexp = np.load('data/{}/coexp/coexpression_chr_{:02d}_{:02d}.npy'.format(args.dataset, args.chr_src, args.chr_tgt))

    hic = np.load('data/{}/hic/combined_observed_KR_{}_{}_10000_sum_oe.npy'.format(args.dataset, args.chr_src, args.chr_tgt))


    concat = np.vstack((coexp[hic < 5].flatten(), hic[hic < 5].flatten())).T

    df_concat= pd.DataFrame(concat, columns=['coexpression', 'interactions'])

    sns.jointplot(x="interactions", y="coexpression", data=df_concat, kind='kde')
    plt.show()

    #sns.scatterplot(x="interactions", y="coexpression", data=df_concat, alpha=0.2)
    #plt.show()