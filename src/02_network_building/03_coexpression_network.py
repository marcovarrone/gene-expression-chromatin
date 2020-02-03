import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--chr-src', type=int, default=1)
    parser.add_argument('--chr-tgt', type=int, default=1)
    parser.add_argument('--percentile', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--abs', default=False, action='store_true')
    parser.add_argument('--save-plot', default=False, action='store_true')
    parser.add_argument('--save-matrix', default=False, action='store_true')
    parser.add_argument('--across', default=False, action='store_true')
    parser.add_argument('--zero-median', default=False, action='store_true')
    args = parser.parse_args()

    coexpression = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/coexp/coexpression_chr_{:02d}_{:02d}{}.npy'.format(
            args.dataset, args.chr_src, args.chr_tgt, '_zero_median' if args.zero_median else ''))

    if args.chr_src == args.chr_tgt:
        np.fill_diagonal(coexpression, 0)

    if args.abs:
        coexpression = np.abs(coexpression)

    if args.percentile:
        threshold = np.nanpercentile(coexpression, args.percentile)
    else:
        threshold = args.threshold
    print(threshold)

    coexpression[coexpression < threshold] = -1
    coexpression[coexpression >= threshold] = 1

    coexpression_csr = sps.csr_matrix(coexpression)

    if args.save_matrix:
        sps.save_npz(
            'data/{}/coexpression/coexpression_chr_{:02d}_{:02d}_{}{}{}.npz'.format(args.dataset, args.chr_src,
                                                                                    args.chr_tgt,
                                                                                    threshold,
                                                                                    '_zero_median' if args.zero_median else '',
                                                                                    '_abs' if args.abs else ''),
            coexpression_csr)

    plt.imshow(coexpression, cmap="Oranges")
    if args.save_plot:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(
            'plots/{}/coexpression_chr_{:02d}_{:02d}_{}{}{}.png'.format(args.dataset, args.chr_src, args.chr_tgt,
                                                                        threshold,
                                                                        '_zero_median' if args.zero_median else '',
                                                                        '_abs' if args.abs else ''))
    plt.show()
