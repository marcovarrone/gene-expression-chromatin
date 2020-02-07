import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from utils_network_building import coexpression_threshold, intra_mask


def single_chromosome(args):
    coexpression = np.load(
        '../../data/{}/coexpression/coexpression_chr_{}_{}.npy'.format(args.dataset, args.chr_src, args.chr_tgt))

    print('Computing co-expression network between chromosome', args.chr_src, 'and chromosome', args.chr_tgt)

    if args.abs:
        coexpression = np.abs(coexpression)

    is_intra = (args.chr_src == args.chr_tgt)
    if is_intra and args.perc_intra is not None:
        threshold = coexpression_threshold(args.dataset, percentile_intra=args.perc_intra)
    elif not is_intra and args.perc_inter is not None:
        threshold = coexpression_threshold(args.dataset, percentile_inter=args.perc_inter)
    elif args.threshold:
        threshold = args.threshold
    else:
        raise ValueError('Either one parameter between --percentile and --threshold must be passed.')

    threshold = np.round(threshold, 2)

    print('Treshold:', threshold)

    coexpression[coexpression < threshold] = 0
    coexpression[coexpression >= threshold] = 1

    print('N. edges after thresholding', (coexpression == 1).sum())

    if args.save_matrix:
        data_folder = '../../data/{}/coexpression_networks'.format(args.dataset)
        os.makedirs(data_folder, exist_ok=True)
        np.save(
            data_folder + '/coexpression_chr_{}_{}_{}{}.npy'.format(args.chr_src,
                                                                                         args.chr_tgt,
                                                                                         threshold,
                                                                                         '_abs' if args.abs else ''),
            coexpression)

    if args.save_plot:
        os.makedirs('../../plots/{}/coexpression_networks/'.format(args.dataset), exist_ok=True)
        plt.imshow(coexpression, cmap='Oranges')
        plt.savefig(
            '../../plots/{}/coexpression_networks/coexpression_chr_{}_{}_{}{}.png'.format(args.dataset, args.chr_src,
                                                                                          args.chr_tgt,
                                                                                          threshold,
                                                                                          '_abs' if args.abs else ''))
        plt.close()


def multi_chromosome(args):
    coexpression_full = np.load('data/{}/coexp/coexpression_chr_all.npy'.format(args.dataset))

    shapes = [np.load('data/{}/coexpression/coexpression_chr_{}_{}.npy'.format(args.dataset, i, i)).shape for i in
              range(1, 23)]

    mask = intra_mask(shapes)

    if args.perc_inter is not None:
        threshold_inter = coexpression_threshold(args.dataset, percentile_inter=args.perc_inter)
        coexpression_inter = coexpression_full * np.logical_not(mask)
        coexpression_inter[coexpression_inter < threshold_inter] = 0
        coexpression_inter[coexpression_inter > 0] = 1
    else:
        coexpression_inter = -np.ones(coexpression_full.shape)

    threshold_intra = coexpression_threshold(args.dataset, percentile_intra=args.perc_intra)
    coexpression_intra = coexpression_full * mask
    coexpression_intra[coexpression_intra < threshold_intra] = 0
    coexpression_intra[coexpression_intra > 0] = 1

    print(coexpression_inter.sum(), coexpression_intra.sum())

    coexpression_thr = coexpression_intra + coexpression_inter

    if args.perc_inter is not None:
        filename = 'coexpression_chr_all_{}_{}'.format(threshold_intra, threshold_inter)
    else:
        filename = 'coexpression_chr_all_{}'.format(threshold_intra)

    if args.save_matrix:
        data_folder = '../../data/{}/coexpression_networks'.format(args.dataset)
        os.makedirs(data_folder, exist_ok=True)
        np.save(
            data_folder + '/' + filename + '.npy'.format(filename), coexpression_thr)

    if args.save_plot:
        plt.figure(figsize=(7, 7), dpi=600)
        plt.imshow(coexpression_thr, cmap='Oranges')
        plt.savefig(
            '../../plots/{}/{}.png'.format(args.dataset, filename))
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--chr-src', type=int, default=None)
    parser.add_argument('--chr-tgt', type=int, default=None)
    parser.add_argument('--perc-intra', type=int, default=None, required=True)
    parser.add_argument('--perc-inter', type=int, default=None)
    # parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--single-chrom', default=False, action='store_true')
    parser.add_argument('--abs', default=False, action='store_true')
    parser.add_argument('--save-plot', default=False, action='store_true')
    parser.add_argument('--save-matrix', default=False, action='store_true')
    args = parser.parse_args()

    if args.single_chrom:
        if args.chr_src and args.chr_tgt:
            single_chromosome(args)
        else:
            for i in range(1,23):
                args.chr_src = i
                args.chr_tgt = i
                single_chromosome(args)
    else:
        multi_chromosome(args)
