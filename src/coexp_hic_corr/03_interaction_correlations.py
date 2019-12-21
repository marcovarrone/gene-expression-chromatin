import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MCF7')
parser.add_argument('--chr', type=int, default=8)
parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='oe')
parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR'], default='NONE')
parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='primary')
parser.add_argument('--resolution', type=int, default=10000)
parser.add_argument('--num-most', type=int, default=12900)
parser.add_argument('--window', type=int, default=40000)
parser.add_argument('--aggregation', type=str, choices=['median', 'sum', 'max', None], default='sum')
parser.add_argument('--save', default=True, action='store_true')
args = parser.parse_args()

data_folder = 'data/{}/'.format(args.dataset)
hic_folder = data_folder + 'hic/'
rna_folder = data_folder + 'coexp/'
hic_file = '{}_{}_{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, args.chr, args.chr,
                                            args.resolution, args.window, args.aggregation)

gene_correlation = np.load(rna_folder + 'coexpression_chr_{:02d}_{:02d}.npy'.format(args.chr, args.chr))
contact_matrix = np.load(hic_folder + hic_file + '.npy')
plt.imshow(np.log(contact_matrix+1), cmap='Oranges')
plt.show()
contact_matrix = np.triu(contact_matrix, 1)

if __name__ == '__main__':
    plt.imshow(1-gene_correlation, cmap='RdBu')
    plt.show()
    indices_most = np.argpartition(-contact_matrix.ravel(), range(args.num_most))[:args.num_most]
    indices_x, indices_y = np.unravel_index(indices_most, contact_matrix.shape)

    coexp_interacting = gene_correlation[indices_x, indices_y]
    sns.distplot(coexp_interacting, label="most interacting")

    mask_interacting = np.zeros(contact_matrix.shape)
    mask_interacting[indices_x, indices_y] = 1
    mask_interacting[indices_y, indices_x] = 1
    #plt.imshow(mask_interacting, cmap='Oranges')
    #plt.show()

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
        plt.savefig('plots/interaction_correlation_chr_{:02d}_'.format(args.chr) + hic_file + '.png')
    plt.show()
