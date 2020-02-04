import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from utils_network_building import coexpression_threshold

def main(args):
    coexpression = np.load(
        '../../data/{}/coexp/coexpression_chr_{}_{}.npy'.format(args.dataset, args.chr_src, args.chr_tgt))

    if args.chr_src == args.chr_tgt:
        np.fill_diagonal(coexpression, 0)

    if args.abs:
        coexpression = np.abs(coexpression)

    if args.percentile:
        is_intra = (args.chr_src == args.chr_tgt)
        if is_intra:
            threshold = coexpression_threshold(args.dataset, percentile_intra=args.percentile)
        else:
            threshold = coexpression_threshold(args.dataset, percentile_inter=args.percentile)

    elif args.threshold:
        threshold = args.threshold
    else:
        raise ValueError('Either one parameter between --percentile and --threshold must be passed.')

    print('Treshold:', threshold)

    coexpression[coexpression < threshold] = -1
    coexpression[coexpression >= threshold] = 1

    print('N. edges after thresholding', (coexpression == 1).sum())

    coexpression_csr = sps.csr_matrix(coexpression)

    if args.save_matrix:
        data_folder = '../../data/{}/coexpression_networks'.format(args.dataset)
        os.makedirs(data_folder, exist_ok=True)
        sps.save_npz(
            data_folder + 'coexpression_networks/coexpression_chr_{}_{}_{}{}.npz'.format(args.dataset, args.chr_src,
                                                                                         args.chr_tgt,
                                                                                         threshold,
                                                                                         '_abs' if args.abs else ''),
            coexpression_csr)

    if args.save_plot:
        os.makedirs('../../plots/{}/coexpression_networks/'.format(args.dataset), exist_ok=True)
        plt.savefig(
            '../../plots/{}/coexpression_networks/coexpression_chr_{}_{}_{}{}.png'.format(args.dataset, args.chr_src,
                                                                                          args.chr_tgt,
                                                                                          threshold,
                                                                                          '_abs' if args.abs else ''))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--chr-src', type=int, default=1)
    parser.add_argument('--chr-tgt', type=int, default=1)
    parser.add_argument('--percentile', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--interchrom', default=False, action='store_true')
    parser.add_argument('--abs', default=False, action='store_true')
    parser.add_argument('--save-plot', default=False, action='store_true')
    parser.add_argument('--save-matrix', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
