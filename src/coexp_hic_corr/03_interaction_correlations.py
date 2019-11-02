import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GM19238')
parser.add_argument('--chr', type=int, default=2)
parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='oe')
parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR'], default='NONE')
parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='combined')
parser.add_argument('--resolution', type=int, default=10000)
parser.add_argument('---num-most', type=int, default=2000)
parser.add_argument('--save', default=True, action='store_true')
args = parser.parse_args()

data_folder = 'data/{}/'.format(args.dataset)
hic_folder = data_folder + 'hic/'
rna_folder = data_folder + 'rna/'
hic_file = '{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, args.chr, args.chr,
                                             args.resolution)

train = np.load(rna_folder + str(args.dataset) + '_chr_{:02d}_zero_median.npy'.format(args.chr))
contact_matrix = np.load(hic_folder + hic_file + '.npy')
contact_matrix = np.triu(contact_matrix, 1)

if __name__ == '__main__':
    gene_correlation = np.corrcoef(train)
    indices_most = np.argpartition(-contact_matrix.ravel(), range(args.num_most))[:args.num_most]
    indices_x, indices_y = np.unravel_index(indices_most, contact_matrix.shape)

    coexp_interacting = gene_correlation[indices_x, indices_y]
    sns.distplot(coexp_interacting, label="most interacting")

    mask_interacting = np.zeros(contact_matrix.shape)
    mask_interacting[indices_x, indices_y] = 1
    mask_interacting[indices_y, indices_x] = 1

    mask_non_interacting = np.logical_not(mask_interacting)
    np.fill_diagonal(mask_non_interacting, 0)
    print("Select indices...")
    index_x, index_y = np.where(mask_non_interacting == 1)
    print("Indices selected")
    sampling_idxs = np.random.choice(np.arange(index_x.shape[0]), int(mask_interacting.sum()), replace=False)
    coexp_non_interacting = gene_correlation[index_x[sampling_idxs], index_y[sampling_idxs]]
    sns.distplot(coexp_non_interacting, label="sampled others")
    _, p_value = ttest_ind(coexp_interacting, coexp_non_interacting, equal_var=False)
    plt.title('Chromosome {} - p-value: {}'.format(args.chr, p_value))
    plt.legend()
    if args.save:
        plt.savefig('plots/interaction_correlation_chr_{:02d}_'.format(args.chr)+hic_file+'.png')
    plt.show()

