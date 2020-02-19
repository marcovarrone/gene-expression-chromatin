import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sps


def generate_hic(hic, gene_info_src, gene_info_tgt, resolution, chr_src, chr_tgt):
    contact_matrix = np.zeros((gene_info_src.shape[0], gene_info_tgt.shape[0]))

    tsses = np.concatenate(
        (gene_info_src['Transcription start site (TSS)'], gene_info_tgt['Transcription start site (TSS)']))

    bins = np.arange(0, np.max(tsses) + resolution, resolution)
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

    if chr_src == chr_tgt:
        contact_matrix[np.tril_indices_from(contact_matrix, k=0)] = np.nan

    return contact_matrix


def build_hic_genome(args, hic_folder):
    hic_output_file = '{}_{}_{}_all_{}'.format(args.file, args.type, args.norm, args.resolution)
    print(hic_output_file)

    shapes = []
    rows = []
    for i in range(1, 23):
        row = []
        for j in range(1, 23):

            if i <= j:
                hic = np.load(
                    '../../data/{}/hic/{}_{}_{}_{}_{}_{}.npy'.format(args.dataset, args.file, args.type, args.norm, i,
                                                                     j, args.resolution))
                row.append(hic)
            else:
                hic = np.load(
                    '../../data/{}/hic/{}_{}_{}_{}_{}_{}.npy'.format(args.dataset, args.file, args.type, args.norm, j,
                                                                     i, args.resolution)).T
                hic = np.empty(hic.shape)
                hic[:] = np.nan
                row.append(hic)

            if i == j:
                shapes.append(hic.shape)
        rows.append(np.hstack(row))
    hic_full = np.vstack(rows)

    if args.save_matrix:
        os.makedirs(hic_folder, exist_ok=True)

        np.save(hic_folder + hic_output_file + '.npy', hic_full)

    if args.save_plot:
        plt.figure(figsize=(7, 7), dpi=600)
        plt.imshow(np.log1p(hic_full), cmap="Reds")
        os.makedirs('../../plots/{}/hic/'.format(args.dataset), exist_ok=True)
        plt.savefig('../../plots/{}/hic/{}.png'.format(args.dataset, hic_output_file))
        plt.clf()
    return


def main(args, hic_folder, rna_folder):
    hic_output_file = '{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, args.chr_src, args.chr_tgt,
                                                 args.resolution)

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

        hic = np.log1p(hic.toarray())
        np.fill_diagonal(hic, 0)

        contact_matrix = generate_hic(hic, df_rna_src, df_rna_tgt, args.resolution, args.chr_src, args.chr_tgt)

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
    parser.add_argument('--save-matrix', default=False, action='store_true')
    parser.add_argument('--save-plot', default=False, action='store_true')
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--genome-wide', default=False, action='store_true')

    args = parser.parse_args()

    data_folder = '../../data/{}/'.format(args.dataset)
    hic_folder = data_folder + 'hic/'
    rna_folder = data_folder + 'rna/'

    if args.chr_src is None or args.chr_tgt is None:
        rows = []
        for chr_src in range(1, 23):
            args.chr_src = chr_src
            if args.genome_wide:
                chromosomes_target = range(chr_src, 23)
            else:
                chromosomes_target = [chr_src]

            for chr_tgt in chromosomes_target:
                args.chr_tgt = chr_tgt
                main(args, hic_folder, rna_folder)
        if args.genome_wide:
            build_hic_genome(args, hic_folder)
    else:
        main(args, hic_folder, rna_folder)
