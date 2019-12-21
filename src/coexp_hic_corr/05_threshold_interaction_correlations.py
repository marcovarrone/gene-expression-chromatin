import argparse
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind

np.random.seed(43)


# ToDo: fix greater_equal()
def greater_equal():
    matrix_thresholded = np.triu(contact_matrix, k=threshold)

    percentile_high = np.percentile(matrix_thresholded[matrix_thresholded > 0], 90)
    percentile_low = np.percentile(matrix_thresholded[matrix_thresholded > 0], 10)
    print(percentile_low, percentile_high, np.max(matrix_thresholded))

    matrix_thresholded[matrix_thresholded <= percentile_high] = 0
    matrix_thresholded[matrix_thresholded > percentile_high] = 1

    mask_interacting = matrix_thresholded
    print(mask_interacting.sum())
    np.fill_diagonal(mask_interacting, 0)
    # coexp_interacting = gene_correlation*matrix_thresholded

    index_x, index_y = np.where(mask_interacting == 1)
    coexp_interacting = gene_coexp[index_x, index_y]
    sns.distplot(coexp_interacting, label="interacting")

    mask_non_interacting = (1 - matrix_thresholded)
    np.fill_diagonal(mask_non_interacting, 0)
    print("Select indices...")
    index_x, index_y = np.where(mask_non_interacting == 1)
    print("Indices selected")
    sampling_idxs = np.random.choice(np.arange(index_x.shape[0]), int(mask_interacting.sum()), replace=False)
    coexp_non_interacting = gene_coexp[index_x[sampling_idxs], index_y[sampling_idxs]]
    return coexp_interacting, coexp_non_interacting


def equal(threshold, num_most, least):
    kth_diag_interactions = np.diagonal(contact_matrix, offset=threshold)
    kth_diag_coexp = np.diagonal(gene_coexp, offset=threshold)

    most_interacting_couples = np.argsort(-kth_diag_interactions)[:num_most]

    coexp_interacting = kth_diag_coexp[most_interacting_couples]

    if least:
        least_interacting_couples = np.argsort(-kth_diag_interactions)[-num_most:]
        coexp_non_interacting = kth_diag_coexp[least_interacting_couples]
    else:
        least_interacting_couples = np.setdiff1d(np.arange(len(kth_diag_interactions)), most_interacting_couples)
        sampling_idxs = np.random.choice(np.arange(least_interacting_couples.shape[0]), len(most_interacting_couples),
                                         replace=False)
        coexp_non_interacting = kth_diag_coexp[least_interacting_couples[sampling_idxs]]

    return coexp_interacting, coexp_non_interacting


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GM19238')
    parser.add_argument('--chr', type=int, default=2)
    parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='observed')
    parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR'], default='KR')
    parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='combined')
    parser.add_argument('--resolution', type=int, default=10000)
    parser.add_argument('--equal', default=True, action='store_true')
    parser.add_argument('--least', default=True, action='store_true')
    parser.add_argument('--num-most', type=int, default=25)
    args = parser.parse_args()

    data_folder = 'data/{}/'.format(args.dataset)
    hic_folder = data_folder + 'hic/'
    rna_folder = data_folder + 'rna/'
    hic_file = '{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, args.chr, args.chr,
                                          args.resolution)

    expression = np.load(rna_folder + str(args.dataset) + '_chr_{:02d}_zero_median.npy'.format(args.chr))

    gene_coexp = np.corrcoef(expression)
    contact_matrix = np.load(hic_folder + hic_file + '.npy')
    contact_matrix = np.triu(contact_matrix, 1)
    half_diagonal_length = int(contact_matrix.shape[0] / sqrt(2))
    thresholds = np.arange(start=1, step=int(contact_matrix.shape[0] / 50), stop=contact_matrix.shape[0] - 50)
    pvalues = []
    avg_most = []
    avg_least = []
    np.fill_diagonal(gene_coexp, 0)
    for i in range(len(thresholds)):
        threshold = thresholds[i]
        print("Threshold", threshold)
        if args.equal:
            coexp_interacting, coexp_non_interacting = equal(threshold, args.num_most, least=args.least)
        else:
            coexp_interacting, coexp_non_interacting = greater_equal()
        avg_most.append(np.mean(coexp_interacting))
        avg_least.append(np.mean(coexp_non_interacting))

        #_, pvalue = ttest_ind(coexp_interacting, coexp_non_interacting, equal_var=False)
        #pvalues.append(pvalue)
        #print(pvalue)

    plt.plot(thresholds, avg_most)
    plt.plot(thresholds, avg_least)
    plt.xlabel("Gene distance")
    plt.ylabel("Avg. coexpression")
    plt.legend(['most interacting', 'least interacting' if args.least else 'sampled others'])
    # plt.yscale('log')
    plt.title('Chromosome {}'.format(args.chr))
    plt.savefig('plots/thr_interaction_corr_chr_{:02d}_{}.png'.format(args.chr, hic_file))
    plt.show()
