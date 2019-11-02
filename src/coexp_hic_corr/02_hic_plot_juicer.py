import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps

parser = argparse.ArgumentParser()

# ToDo: add description
parser.add_argument('-rd', '--rna-dataset', type=str, default='GM19238')
parser.add_argument('-hd', '--hic-dataset', type=str, default='GM12878')
parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='oe')
parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR'], default='NONE')
parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='combined')
parser.add_argument('--chr-src', type=int, default=2)
parser.add_argument('--chr-tgt', type=int, default=2)
parser.add_argument('--resolution', type=int, default=10000)
parser.add_argument('--save-matrix', default=True, action='store_true')
parser.add_argument('--csv', default=False, action='store_true')
parser.add_argument('--save-plot', default=True, action='store_true')
parser.add_argument('--force', default=False, action='store_true')

args = parser.parse_args()

data_folder = '/home/varrone/Data/{}/'.format(args.hic_dataset)
hic_folder = '{}_{}_{}/'.format(args.file, args.type, args.norm)
output_file = '{}_{}_{}_{}'.format(args.file, args.chr_src, args.chr_tgt, args.resolution)

sps_path = data_folder + hic_folder + output_file + '.npz'


def get_gene_bins(gene, bin_size):
    tss = gene['Transcription start site (TSS)']
    tss_bin = tss // bin_size
    start = tss_bin - 40000 // bin_size
    end = tss_bin + 40000 // bin_size
    return start, end


def generate_hic(hic, gene_info_src, gene_info_tgt):
    contact_matrix = np.zeros((gene_info_src.shape[0], gene_info_tgt.shape[0]))
    for i, (idx1, gene1) in enumerate(gene_info_src.iterrows()):
        start1, end1 = get_gene_bins(gene1, 10000)
        print("Processing gene", i, "/", gene_info_src.shape[0])

        for j, (idx2, gene2) in enumerate(gene_info_tgt.iterrows()):
            start2, end2 = get_gene_bins(gene2, 10000)
            mat = hic[start1:end1, start2:end2].A
            value = np.median(np.median(mat))

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

    hic_output_file = '{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, args.chr_src, args.chr_tgt,
                                                 args.resolution)

    df_rna_src = pd.read_csv(
        rna_folder + str(args.rna_dataset) + '_chr_{:02d}_rna.csv'.format(args.chr_src))

    df_rna_tgt = pd.read_csv(
        rna_folder + str(args.rna_dataset) + '_chr_{:02d}_rna.csv'.format(args.chr_tgt))



    if os.path.exists(hic_folder + hic_output_file + '.npy') and not args.force:
        contact_matrix = np.load(hic_folder + hic_output_file + '.npy')
    else:
        hic_matrix = sps.load_npz(sps_path)
        contact_matrix = generate_hic(hic_matrix, df_rna_src, df_rna_tgt)

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
