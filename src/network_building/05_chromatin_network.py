import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from utils_network_building import chromatin_threshold, intra_mask


def save_chr_sizes(hic):
    chr_sizes_path = '../../data/{}/chr_sizes.npy'.format(args.dataset)
    if os.path.exists(chr_sizes_path):
        chr_sizes = np.load(chr_sizes_path)
        chr_sizes[args.chr_src] = hic.shape[0]
    else:
        chr_sizes = np.empty(23)
    np.save(chr_sizes_path, chr_sizes)


def single_chromosome(args):
    hic_name = '{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, args.chr_src, args.chr_tgt, args.window)

    hic = np.load(
        '../../data/{}/hic/{}.npy'.format(args.dataset, hic_name))

    if args.chr_src == args.chr_tgt:
        save_chr_sizes(hic)

    '''if args.chr_src == args.chr_tgt:
        hic[np.tril_indices_from(hic, k=1)] = np.nan'''

    is_intra = (args.chr_src == args.chr_tgt)
    if is_intra and args.perc_intra is not None:
        threshold = chromatin_threshold(args.dataset, args.file, args.type, args.norm, args.window,
                                        percentile_intra=args.perc_intra)
    elif not is_intra and args.perc_inter is not None:
        threshold = chromatin_threshold(args.dataset, args.file, args.type, args.norm, args.window,
                                        percentile_inter=args.perc_inter)
    else:
        # ToDo: give useful error text
        raise ValueError()

    hic[hic <= threshold] = 0
    if not args.weighted:
        hic[hic > 0] = 1

    if args.save_matrix:
        chromatin_networks_folder = '../../data/{}/chromatin_networks/'.format(args.dataset)
        os.makedirs(chromatin_networks_folder, exist_ok=True)
        np.save(
            chromatin_networks_folder + '{}{}_{}.npy'.format(hic_name, '_w' if args.weighted else '', threshold), hic)

    if args.save_plot:
        plt.imshow(np.log1p(hic), cmap="Reds")
        os.makedirs('../../plots/{}/chromatin_networks'.format(args.dataset), exist_ok=True)
        plt.savefig('../../plots/{}/chromatin_networks/{}{}_{}.png'.format(args.dataset, hic_name,
                                                                           '_w' if args.weighted else '', threshold))


def multi_chromosome(args):
    hic_name = '{}_{}_{}_all_{}'.format(args.file, args.type, args.norm, args.window)
    hic_full = np.load('../../data/{}/hic/{}.npy'.format(args.dataset, hic_name))

    shapes = [
        np.load('../../data/{}/hic/{}_{}_{}_{}_{}_{}.npy'.format(args.dataset, args.file, args.type, args.norm, i, i,
                                                                 args.window)).shape for i in range(1, 23)]

    mask = intra_mask(shapes)

    if args.perc_inter is not None:
        threshold_inter = chromatin_threshold(args.dataset, args.file, args.type, args.norm, args.window,
                                              percentile_inter=args.perc_inter)
        hic_inter = hic_full * np.logical_not(mask)
        hic_inter[hic_inter < threshold_inter] = 0
        hic_inter[hic_inter > 0] = 1

    else:
        hic_inter = intra_mask(shapes, nans=True)


    threshold_intra = chromatin_threshold(args.dataset, args.file, args.type, args.norm, args.window,
                                    percentile_intra=args.perc_intra)
    print(threshold_intra)
    hic_intra = hic_full * mask

    hic_intra[hic_intra < threshold_intra] = 0
    hic_intra[hic_intra > 0] = 1

    print(np.nansum(hic_intra), np.nansum(hic_inter))

    hic_thr = hic_intra + hic_inter

    if args.perc_inter is not None:
        filename = '{}_{}_{}'.format(hic_name, threshold_intra, threshold_inter)
    else:
        filename = '{}_{}'.format(hic_name, threshold_intra)

    if args.save_matrix:
        data_folder = '../../data/{}/chromatin_networks/'.format(args.dataset)
        os.makedirs(data_folder, exist_ok=True)
        np.save(
            data_folder + filename + '.npy'.format(filename), hic_thr)

    if args.save_plot:
        plt.figure(figsize=(7, 7), dpi=600)
        plt.imshow(np.log1p(hic_thr), cmap='Oranges')
        plt.savefig(
            '../../plots/{}/chromatin_networks/{}.png'.format(args.dataset, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='prostate')
    parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='observed')
    parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR', 'ICE'], default='ICE')
    parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='primary')
    parser.add_argument('--chr-src', type=int, default=None)
    parser.add_argument('--chr-tgt', type=int, default=None)
    parser.add_argument('--resolution', type=int, default=40000)
    parser.add_argument('--window', type=int, default=40000)
    parser.add_argument('--perc-intra', type=int, default=80)
    parser.add_argument('--perc-inter', type=int, default=None)
    parser.add_argument('--single-chrom', default=False, action='store_true')
    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--save-plot', default=True, action='store_true')
    parser.add_argument('--save-matrix', default=False, action='store_true')
    args = parser.parse_args()

    if args.single_chrom:
        if args.chr_src and args.chr_tgt:
            single_chromosome(args)
        else:
            for i in range(1, 23):
                args.chr_src = i
                args.chr_tgt = i
                single_chromosome(args)
    else:
        multi_chromosome(args)
