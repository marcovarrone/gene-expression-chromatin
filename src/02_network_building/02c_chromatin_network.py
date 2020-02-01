import argparse
import os

import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MCF7')
    parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='observed')
    parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR', 'ICE'], default='KR')
    parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='primary')
    parser.add_argument('--chr-src', type=int, default=1)
    parser.add_argument('--chr-tgt', type=int, default=1)
    parser.add_argument('--resolution', type=int, default=50000)
    parser.add_argument('--window', type=int, default=50000)
    parser.add_argument('--aggregation', type=str, choices=['median', 'sum', 'max', None], default=None)
    parser.add_argument('--thr-perc', type=int, default=90)
    parser.add_argument('--thr', type=float, default=2.74)
    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--save-plot', default=True, action='store_true')
    parser.add_argument('--save-matrix', default=True, action='store_true')
    parser.add_argument('--across', default=False, action='store_true')
    args = parser.parse_args()
    hic_name = '{}_{}_{}_{}_{}_{}_{}{}'.format(args.file, args.type, args.norm, args.chr_src, args.chr_tgt,
                                               args.resolution, args.window,
                                               ('_' + args.aggregation) if args.aggregation else '')

    hic = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/hic/{}.npy'.format(args.dataset,
                                                                                                   hic_name))

    if args.chr_src == args.chr_tgt:
        if os.path.exists('/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/chr_sizes.npy'.format(args.dataset)):
            chr_sizes = np.load('/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/chr_sizes.npy'.format(args.dataset))
            chr_sizes[args.chr_src] = hic.shape[0]
        else:
            chr_sizes = np.empty(23)
            np.save('/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/chr_sizes.npy'.format(args.dataset), chr_sizes)

    if args.chr_src == args.chr_tgt:
        np.fill_diagonal(hic, 0)
    plt.imshow(np.log1p(hic), cmap="Oranges")
    plt.show()

    if args.across:
        values_across = np.array([0])
        for i in range(1, 23):
            hic_name_across = '{}_{}_{}_{}_{}_{}_{}{}'.format(args.file, args.type, args.norm, i, i,
                                                              args.resolution, args.window,
                                                              ('_' + args.aggregation) if args.aggregation else '')
            hic_across = np.load(
                '/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/hic/{}.npy'.format(args.dataset,
                                                                                                           hic_name_across))
            np.fill_diagonal(hic_across, 0)
            values_coexp = np.concatenate((values_across, hic_across.flatten()))
        threshold = np.nanpercentile(values_across, args.thr_perc)
    else:
        threshold = np.nanpercentile(hic, args.thr_perc)

    if args.thr is not None:
        threshold = args.thr
    print(threshold)
    interactions = hic.copy()

    np.fill_diagonal(interactions, -1)

    interactions[np.isnan(interactions)] = 0
    interactions[interactions <= threshold] = -1
    if not args.weighted:
        interactions[interactions > 0] = 1

    if args.save_matrix:
        sps.save_npz(
            'data/{}/interactions/interactions_{}{}_{}.npz'.format(args.dataset, hic_name,
                                                                   '_w' if args.weighted else '',
                                                                   args.thr if args.thr is not None else args.thr_perc),
            sps.csr_matrix(interactions))
    interactions_full = interactions + interactions.T
    np.diagonal(interactions_full, 0)
    plt.imshow(interactions_full, cmap="Oranges")
    if args.save_plot:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/{}/interactions_{}{}_{}.png'.format(args.dataset, hic_name, '_w' if args.weighted else '',
                                                               args.thr if args.thr is not None else args.thr_perc))
    plt.show()
