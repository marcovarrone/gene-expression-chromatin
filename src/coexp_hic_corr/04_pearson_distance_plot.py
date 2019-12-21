import argparse
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import itertools
from collections import defaultdict

def get_values_from_distance(matrix):
    values = []
    distances = []
    for d in range(1, int(matrix.shape[0]-1)):
        diag = matrix.diagonal(d)
        values.append(np.mean(diag))
        distances.append(d)
    return distances, values

def get_values_from_real_distance_v1(dataset, chr, coexpression):
    rna = pd.read_csv('/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/rna/{}_chr_{:02d}_rna.csv'.format(dataset, dataset, chr))
    tss = rna['Transcription start site (TSS)']
    #tss_binned = np.digitize(tss.to_numpy(), np.arange(0, np.max(tss)+50000, 50000))*50000
    coexpression += 2
    combinations = np.array(list(itertools.combinations(range(len(tss)), 2)))
    distances = np.abs(tss[combinations[:, 1]].to_numpy() - tss[combinations[:, 0]].to_numpy())
    values = coexpression[combinations[:, 0], combinations[:, 1]]
    #for distance in np.unique(distances):
    #    values_distance =

    values, bin_edges, bin_number = stats.binned_statistic(distances, values, 'mean', bins=len(tss)//3)
    coexpression_norm = coexpression.copy()

    #plt.imshow(coexpression, cmap='Oranges')
    #plt.show()

    coexpression_norm[combinations[:, 0], combinations[:, 1]] /= values[bin_number-1]

    coexpression_norm += coexpression_norm.T
    np.fill_diagonal(coexpression_norm, 0)

    threshold = np.nanpercentile(coexpression, 90)
    coexpression[coexpression < threshold] = 0
    coexpression[coexpression >= threshold] = 1
    print(threshold)

    plt.imshow(coexpression, cmap='Oranges')
    plt.show()

    threshold = np.nanpercentile(coexpression_norm, 90)
    print(threshold)
    coexpression_norm[coexpression_norm < threshold] = 0
    coexpression_norm[coexpression_norm >= threshold] = 1

    plt.imshow(coexpression_norm, cmap='Oranges')
    plt.show()

    distances = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)
    return distances, values

def get_values_from_real_distance(dataset, chr, coexpression):
    rna = pd.read_csv('/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/rna/{}_chr_{:02d}_rna.csv'.format(dataset, dataset, chr))
    tss = rna['Transcription start site (TSS)']
    tss_binned = np.digitize(tss.to_numpy(), np.arange(0, np.max(tss)+50000, 50000))*50000

    coexpression += 2
    combinations = np.array(list(itertools.combinations(range(len(tss)), 2)))
    values_list = defaultdict(list)
    for combination in combinations:
        value = coexpression[combination[0], combination[1]]
        distance = np.abs(tss_binned[combination[1]] - tss_binned[combination[0]])
        values_list[distance].append(value)

    distances = values_list.keys()
    values = {}
    for distance in distances:
        values[distance] = np.mean(values_list[distance])

    expected = np.ones(combinations.shape[0])
    for i, combination in enumerate(combinations):
        distance = np.abs(tss_binned[combination[1]] - tss_binned[combination[0]])
        expected[i] = values[distance]


    #values, bin_edges, bin_number = stats.binned_statistic(distances, values, 'mean', bins=len(tss)//3)
    coexpression_norm = coexpression.copy()





    coexpression_norm[combinations[:, 0], combinations[:, 1]] /= expected

    #coexpression_norm += coexpression_norm.T
    np.fill_diagonal(coexpression_norm, 0)
    np.fill_diagonal(coexpression, 0)

    threshold = np.nanpercentile(coexpression, 90)
    coexpression[coexpression < threshold] = 0
    coexpression[coexpression >= threshold] = 1
    print(threshold)

    plt.imshow(coexpression, cmap='Oranges')
    plt.show()

    coexpression_norm += coexpression_norm.T


    threshold = np.nanpercentile(coexpression_norm, 90)
    print(threshold)
    coexpression_norm[coexpression_norm < threshold] = 0
    coexpression_norm[coexpression_norm >= threshold] = 1

    plt.imshow(coexpression_norm, cmap='Oranges')
    plt.show()

    #distances = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)
    return list(values.keys()), list(values.values())






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--chr', type=int, default=2)
    parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='oe')
    parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR'], default='NONE')
    parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='combined')
    parser.add_argument('--data-type', type=str, choices=['coexp', 'interactions'], default='coexp')
    parser.add_argument('--resolution', type=int, default=10000)
    parser.add_argument('--save-coexp', default=False, action='store_true')
    parser.add_argument('--save-interactions', default=False, action='store_true')
    parser.add_argument('--save-hic', default=False, action='store_true')
    parser.add_argument('--real-distance', default=True, action='store_true')
    args = parser.parse_args()



    data_folder = 'data/{}/'.format(args.dataset)
    hic_folder = data_folder + 'hic/'
    rna_folder = data_folder + 'rna/'
    hic_file = '{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, args.chr, args.chr,
                                          args.resolution)
    if args.data_type == 'coexp':
        coexpression = np.load(
            '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/coexp/coexpression_chr_{:02d}_{:02d}.npy'.format(
                args.dataset, args.chr, args.chr))
        np.fill_diagonal(coexpression, 0)

        if args.real_distance:
            plt.plot(*get_values_from_real_distance(args.dataset, args.chr, coexpression))
        else:
            plt.plot(*get_values_from_distance(coexpression))
        plt.ylabel('Avg. coexpression (Pearson)')
        plt.xlabel('Gene distance')
        plt.title('Chromosome {}'.format(args.chr))
        if args.save_coexp:
            plt.savefig('plots/coexp_distance_chr_{:02d}.png'.format(args.chr))
        else:
            plt.show()
        plt.clf()
    else:
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

