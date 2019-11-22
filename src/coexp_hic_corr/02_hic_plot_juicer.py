import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps

parser = argparse.ArgumentParser()

# ToDo: add description
parser.add_argument('-rd', '--rna-dataset', type=str, default='MCF7')
parser.add_argument('-hd', '--hic-dataset', type=str, default='MCF7')
parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='observed')
parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR'], default='KR')
parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='primary')
parser.add_argument('--chr-src', type=int, default=2)
parser.add_argument('--chr-tgt', type=int, default=2)
parser.add_argument('--resolution', type=int, default=10000)
parser.add_argument('--window', type=int, default=80000)
parser.add_argument('--aggregation', type=str, choices=['median', 'sum', 'max'], default='sum')
parser.add_argument('--save-matrix', default=False, action='store_true')
parser.add_argument('--csv', default=False, action='store_true')
parser.add_argument('--save-plot', default=False, action='store_true')
parser.add_argument('--force', default=False, action='store_true')
parser.add_argument('--all-regions', default=False, action='store_true')

args = parser.parse_args()

data_folder = '/home/varrone/Data/{}/'.format(args.hic_dataset)
hic_folder = '{}_{}_{}/'.format(args.file, args.type, args.norm)
output_file = '{}_{}_{}_{}'.format(args.file, args.chr_src, args.chr_tgt, args.resolution)

sps_path = data_folder + hic_folder + output_file + '.npz'


def get_gene_bins(gene, bin_size, window):
    tss = gene['Transcription start site (TSS)']
    tss_bin = round(tss / bin_size)
    start = tss_bin - int(np.ceil((window/2) / bin_size))
    end = tss_bin + int(np.ceil((window/2) / bin_size))
    return start if start > 0 else 0, end


def generate_hic(hic, gene_info_src, gene_info_tgt, resolution, window):
    contact_matrix = np.zeros((gene_info_src.shape[0], gene_info_tgt.shape[0]))
    for i, (idx1, gene1) in enumerate(gene_info_src.iterrows()):
        start1, end1 = get_gene_bins(gene1, resolution, window)
        print("Processing gene", i, "/", gene_info_src.shape[0])

        for j, (idx2, gene2) in enumerate(gene_info_tgt.iterrows()):
            start2, end2 = get_gene_bins(gene2, resolution, window)
            mat = hic[start1:end1, start2:end2].A
            if args.aggregation == 'median':
                value = np.median(np.median(mat))
            elif args.aggregation == 'max':
                value = np.max(mat)
            else:
                value = np.sum(mat)
            contact_matrix[i, j] += value
            if gene1['Chromosome/scaffold name'] == gene2['Chromosome/scaffold name']:
                contact_matrix[j, i] += value
    return contact_matrix


if __name__ == '__main__':

    data_folder = 'data/{}/'.format(args.rna_dataset)
    hic_folder = data_folder + 'hic/'
    rna_folder = data_folder + 'rna/'
    if not os.path.exists(hic_folder):
        os.makedirs(hic_folder)
    if not os.path.exists(rna_folder):
        os.makedirs(rna_folder)

    hic_output_file = '{}_{}_{}_{}_{}_{}_{}{}_{}'.format(args.file, args.type, args.norm, args.chr_src, args.chr_tgt,
                                                     args.resolution, args.window, '_all' if args.all_regions else '',
                                                     args.aggregation)

    print(hic_output_file)

    df_rna_src = pd.read_csv(
        rna_folder + str(args.rna_dataset) + '_chr_{:02d}_rna.csv'.format(args.chr_src))

    df_rna_tgt = pd.read_csv(
        rna_folder + str(args.rna_dataset) + '_chr_{:02d}_rna.csv'.format(args.chr_tgt))

    if os.path.exists(hic_folder + hic_output_file + '.npy') and not args.force:
        contact_matrix = np.load(hic_folder + hic_output_file + '.npy')
    else:
        if not os.path.exists(sps_path):
            print("Hi-C data not present. Downloading...")
            os.system(
                'python3 01_juicer2numpy.py --dataset {} --type {} --norm {} --file {} --chr-src {} --chr-tgt {} --resolution {}'.format(
                    args.hic_dataset, args.type, args.norm, args.file, args.chr_src, args.chr_tgt, args.resolution))
            print("Downloading completed")
        hic_matrix = sps.load_npz(sps_path)
        if args.all_regions:
            contact_matrix = hic_matrix.A
            contact_matrix[np.isnan(contact_matrix)] = 0
            contact_matrix += contact_matrix.T
            np.fill_diagonal(contact_matrix, hic_matrix.A.diagonal())
            gene_idxs_src = np.round(df_rna_src['Transcription start site (TSS)'].to_numpy() / args.resolution) \
                .astype(int)
            gene_idxs_tgt = np.round(df_rna_tgt['Transcription start site (TSS)'].to_numpy() / args.resolution) \
                .astype(int)

            #zero_rows = np.where((contact_matrix == 0).all(axis=1))[0]
            #gene_idxs_src = gene_idxs_src[~np.isin(gene_idxs_src, zero_rows)]
            #contact_matrix = np.delete(contact_matrix, zero_rows, axis=0)

            #zero_cols = np.where((contact_matrix == 0).all(axis=0))[0]
            #gene_idxs_tgt = gene_idxs_src[~np.isin(gene_idxs_tgt, zero_cols)]
            #contact_matrix = np.delete(contact_matrix, zero_cols, axis=1)

            np.save(data_folder + 'gene_idxs/{}.npy'.format(hic_output_file), gene_idxs_src)
        else:
            contact_matrix = generate_hic(hic_matrix, df_rna_src, df_rna_tgt, args.resolution, args.window)

    if args.save_matrix:
        if not os.path.exists(hic_folder):
            os.makedirs(hic_folder)

        if args.csv:
            np.savetxt(hic_folder + hic_output_file + '.csv', contact_matrix, delimiter=",")
        else:
            np.save(hic_folder + hic_output_file + '.npy', contact_matrix)

    plt.imshow(np.rot90(np.log2(contact_matrix + 1)), cmap="Oranges")
    if args.save_plot:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/{}.png'.format(hic_output_file))
    plt.show()
