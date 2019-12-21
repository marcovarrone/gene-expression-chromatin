import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PRAD')
    parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='observed')
    parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR'], default='KR')
    parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='primary')
    parser.add_argument('--chr-src', type=int, default=1)
    parser.add_argument('--chr-tgt', type=int, default=1)
    parser.add_argument('--resolution', type=int, default=40000)
    parser.add_argument('--window', type=int, default=40000)
    parser.add_argument('--aggregation', type=str, choices=['median', 'sum', 'max', None], default=None)
    parser.add_argument('--thr-perc', type=int, default=90)
    parser.add_argument('--thr', type=float, default=2.42)
    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--save-plot', default=True, action='store_true')
    parser.add_argument('--save-matrix', default=True, action='store_true')
    parser.add_argument('--intra', default=True, action='store_true')
    parser.add_argument('--across', default=False, action='store_true')
    args = parser.parse_args()
    hic_name = '{}_{}_{}_{}_{}_{}_{}{}'.format(args.file, args.type, args.norm, args.chr_src, args.chr_tgt,
                                               args.resolution, args.window,
                                               ('_' + args.aggregation) if args.aggregation else '')

    hic = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/hic/{}.npy'.format(args.dataset,
                                                                                                   hic_name))
    if args.intra:
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
                '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/hic/{}.npy'.format(args.dataset,
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

    np.fill_diagonal(interactions, 0)

    interactions[np.isnan(interactions)] = 0
    interactions[interactions <= threshold] = 0
    if not args.weighted:
        interactions[interactions > 0] = 1

    # ToDo: implement for inter-chromosomal interactions
    if args.intra:
        genes_chr = np.array([args.chr_src]*interactions.shape[0])
        degrees = interactions.sum(axis=1)
        disconnected_nodes = np.where(degrees == 0)[0]
        genes_chr[disconnected_nodes] = 0
        print("Disconnected nodes: ", disconnected_nodes)

        if args.save_matrix:
            np.save('/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/genes_chr/'.format(
                args.dataset) + hic_name + '.npy', genes_chr)

    '''degrees = interactions.sum(axis=1)
    # graph = nx.from_numpy_array(interactions)
    disconnected_nodes = np.where(degrees == 0)[0]
    for idx in disconnected_nodes:
        neighbor = np.argsort(-hic[idx])[0]
        if args.weighted:
            value = hic[idx, neighbor]
        else:
            value = 1
        interactions[idx, neighbor] = value
        if args.intra:
            # ToDo: check if it is redundand since there is another block later
            interactions[neighbor, idx] = value

    degrees = interactions.sum(axis=0)
    disconnected_nodes = np.where(degrees == 0)[0]
    for idx in disconnected_nodes:
        neighbor = np.argsort(-hic[:, idx])[0]
        if args.weighted:
            value = hic[neighbor, idx]
        else:
            value = 1
        interactions[neighbor, idx] = value

        if args.intra:
            interactions[idx, neighbor] = value'''

    if args.save_matrix:
        np.save(
            'data/{}/interactions/interactions_{}{}_{}.npy'.format(args.dataset, hic_name,
                                                                   '_w' if args.weighted else '',
                                                                   args.thr if args.thr is not None else args.thr_perc),
            interactions)
    interactions_full = interactions + interactions.T
    np.diagonal(interactions_full, 0)
    plt.imshow(interactions_full, cmap="Oranges")
    if args.save_plot:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/{}/interactions_{}{}_{}.png'.format(args.dataset, hic_name, '_w' if args.weighted else '',
                                                               args.thr if args.thr is not None else args.thr_perc))
    plt.show()
