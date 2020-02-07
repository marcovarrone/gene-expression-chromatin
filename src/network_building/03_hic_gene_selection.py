import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps


def get_gene_bins(gene, bin_size, window, bins):
    tss = gene['Transcription start site (TSS)']
    tss_bin = np.digitize(tss, bins)
    start = tss_bin - int(np.ceil((window / 2) / bin_size))
    end = tss_bin + int(np.floor((window / 2) / bin_size))
    return start if start > 0 else 0, end


def generate_hic(hic, gene_info_src, gene_info_tgt, resolution, window, chr_src, chr_tgt):
    contact_matrix = np.zeros((gene_info_src.shape[0], gene_info_tgt.shape[0]))

    tsses = np.concatenate(
        (gene_info_src['Transcription start site (TSS)'], gene_info_tgt['Transcription start site (TSS)']))

    bins = np.arange(0, np.max(tsses) + resolution, resolution)
    if resolution == window:
        # All combinations of gene indexes
        combs = np.array(np.meshgrid(range(contact_matrix.shape[0]), range(contact_matrix.shape[1]))).T.reshape(-1, 2)

        # Extract the bin relative to each gene's TSS
        idxs_src = np.digitize(gene_info_src['Transcription start site (TSS)'][combs[:, 0]], bins) - 1
        idxs_tgt = np.digitize(gene_info_tgt['Transcription start site (TSS)'][combs[:, 1]], bins) - 1

        idxs_src_out_boundary = np.where(idxs_src >= hic.shape[0])[0]
        idxs_tgt_out_boundary = np.where(idxs_tgt >= hic.shape[1])[0]

        if idxs_src_out_boundary.shape[0] != 0 or idxs_tgt_out_boundary.shape[0] != 0:
            print("Warning: some genes are out of boundary from Hi-C!")
            idxs_out_boundary = np.concatenate((idxs_src_out_boundary, idxs_tgt_out_boundary))

            combs = np.delete(combs, idxs_out_boundary, axis=0)
            idxs_src = np.delete(idxs_src, idxs_out_boundary, axis=0)
            idxs_tgt = np.delete(idxs_tgt, idxs_out_boundary, axis=0)

        # Set the contact matrix values to the values of Hi-C at the extracted bins
        contact_matrix[combs[:, 0], combs[:, 1]] = np.ravel(hic[idxs_src, idxs_tgt])
    else:
        # ToDo: vectorize
        for i, (idx1, gene1) in enumerate(gene_info_src.iterrows()):
            start1, end1 = get_gene_bins(gene1, resolution, window, bins)
            print("Processing gene", i, "/", gene_info_src.shape[0])

            for j, (idx2, gene2) in enumerate(gene_info_tgt.iterrows()):
                start2, end2 = get_gene_bins(gene2, resolution, window, bins)
                if args.window == 0:
                    end1 += 1
                    end2 += 1

                mat = hic[start1:end1, start2:end2].A
                if args.aggregation == 'median':
                    value = np.median(np.median(mat))
                elif args.aggregation == 'max':
                    value = np.max(mat)
                elif args.aggregation == 'sum':
                    value = np.sum(mat)
                else:
                    if window == resolution:
                        value = np.sum(mat)
                    else:
                        raise ValueError
                contact_matrix[i, j] += value

    if chr_src == chr_tgt:
        #plt.imshow(np.log1p(contact_matrix), cmap='Reds')
        #plt.show()
        contact_matrix[np.tril_indices_from(contact_matrix, k=1)] = np.nan
        #plt.imshow(np.log1p(contact_matrix), cmap='Reds')
        #plt.show()

    return contact_matrix

def main(args):
    data_folder = '../../data/{}/'.format(args.dataset)
    hic_folder = data_folder + 'hic/'
    rna_folder = data_folder + 'rna/'

    hic_output_file = '{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, args.chr_src, args.chr_tgt,
                                                 args.window)

    print(hic_output_file)

    df_rna_src = pd.read_csv(
        rna_folder + 'expression_info_chr_{}.csv'.format(args.chr_src))

    df_rna_tgt = pd.read_csv(
        rna_folder + 'expression_info_chr_{}.csv'.format(args.chr_tgt))

    if os.path.exists(hic_folder + hic_output_file + '.npy') and not args.force:
        print("Data already present. Loading from file.")
        contact_matrix = np.load(hic_folder + hic_output_file + '.npy')
    else:
        hic = sps.load_npz(
            data_folder + 'hic_raw/{}_{}_{}/{}_{}_{}_{}_{}_{}.npz'.format(args.file, args.type, args.norm, args.file,
                                                                          args.type, args.norm, args.chr_src,
                                                                          args.chr_tgt, args.resolution))
        '''if args.chr_src == args.chr_tgt:
            plt.imshow(np.log1p(hic_matrix.toarray()), cmap="Reds")
            plt.show()
            # ToDo: check if it is necessary
            hic_matrix = sps.triu(hic_matrix)
            hic_matrix_full = hic_matrix + hic_matrix.T
            hic_matrix_full = hic_matrix_full.todense()
            np.fill_diagonal(hic_matrix_full, np.diagonal(hic_matrix.todense()))
        else:
            hic_matrix_full = hic_matrix'''
        contact_matrix = generate_hic(hic, df_rna_src, df_rna_tgt, args.resolution, args.window,
                                      args.chr_src, args.chr_tgt)

    if args.save_matrix:
        os.makedirs(hic_folder, exist_ok=True)

        np.save(hic_folder + hic_output_file + '.npy', contact_matrix)

    if args.save_plot:
        plt.imshow(np.log1p(contact_matrix), cmap="Reds")
        os.makedirs('../../plots/{}/hic/'.format(args.dataset), exist_ok=True)
        plt.savefig('../../plots/{}/hic/{}.png'.format(args.dataset, hic_output_file))
        plt.clf()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # ToDo: add description
    parser.add_argument('-d', '--dataset', type=str, default='prostate')
    parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='observed')
    parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR', 'ICE'], default='ICE')
    parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='primary')
    parser.add_argument('--chr-src', type=int, default=None)
    parser.add_argument('--chr-tgt', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=40000)
    parser.add_argument('--window', type=int, default=40000)
    parser.add_argument('--save-matrix', default=False, action='store_true')
    parser.add_argument('--save-plot', default=True, action='store_true')
    parser.add_argument('--force', default=True, action='store_true')

    args = parser.parse_args()

    if args.chr_src is None or args.chr_tgt is None:
        rows = []
        for chr_src in range(1,23):
            args.chr_src = chr_src
            for chr_tgt in range(chr_src, 23):
                args.chr_tgt = chr_tgt
                main(args)
    else:
        main(args)
