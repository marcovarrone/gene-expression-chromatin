import argparse
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def get_values_from_distance(matrix):
    values = []
    distances = []
    for d in range(1, int(matrix.shape[0]-1)):
        diag = matrix.diagonal(d)
        values.append(np.mean(diag))
        distances.append(d)
    return distances, values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GM19238')
    parser.add_argument('--chr', type=int, default=2)
    parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='oe')
    parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR'], default='NONE')
    parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='combined')
    parser.add_argument('--resolution', type=int, default=10000)
    parser.add_argument('--save-coexp', default=False, action='store_true')
    parser.add_argument('--save-interactions', default=False, action='store_true')
    parser.add_argument('--save-hic', default=False, action='store_true')
    args = parser.parse_args()

    data_folder = 'data/{}/'.format(args.dataset)
    hic_folder = data_folder + 'hic/'
    rna_folder = data_folder + 'rna/'
    hic_file = '{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, args.chr, args.chr,
                                          args.resolution)

    expression = np.load(rna_folder + str(args.dataset) + '_chr_{:02d}_zero_median.npy'.format(args.chr))
    coexpression = np.corrcoef(expression)

    plt.plot(*get_values_from_distance(coexpression))
    plt.ylabel('Avg. coexpression (Pearson)')
    plt.xlabel('Gene distance')
    plt.title('Chromosome {}'.format(args.chr))
    if args.save_coexp:
        plt.savefig('plots/coexp_distance_chr_{:02d}.png'.format(args.chr))
    else:
        plt.show()
    plt.clf()

    contact_matrix = np.load(hic_folder + hic_file + '.npy')
    contact_matrix = np.triu(contact_matrix, 1)

    plt.plot(*get_values_from_distance(contact_matrix))
    plt.ylabel('Avg. normalized number of interactions')
    plt.xlabel('Gene distance')
    plt.title('Chromosome {}'.format(args.chr))
    if args.save_interactions:
        plt.savefig('plots/interactions_distance_chr_{:02d}.png'.format(args.chr))
    else:
        plt.show()

