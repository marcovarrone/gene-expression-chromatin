import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from utils_network_building import chromatin_threshold, intra_mask

def save_disconnected_nodes(hic, dataset, filename):
    degrees = np.sum(hic, axis=0)
    disconnected_nodes = np.ravel(np.argwhere(degrees == 0))
    np.save('../../data/{}/disconnected_nodes/{}.npy'.format(dataset, filename), disconnected_nodes)


#ToDo: make single function for common code
def single_chromosome(args):
    hic_name = '{}_{}_{}_{}_{}_{}'.format(args.file, args.type, args.norm, args.chr_src, args.chr_tgt, args.window)

    hic = np.load(
        '../../data/{}/hic/{}.npy'.format(args.dataset, hic_name))

    print('Chromosome', args.chr_src)

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

    filename = '{}{}_{}'.format(hic_name, '_w' if args.weighted else '', threshold)

    save_disconnected_nodes(hic, args.dataset, filename)

    if args.save_matrix:
        data_folder = '../../data/{}/chromatin_networks/'.format(args.dataset)
        os.makedirs(data_folder, exist_ok=True)
        np.save(
            data_folder + filename+'.npy', hic)

    if args.save_plot:
        plt.imshow(np.log1p(hic), cmap="Reds")
        os.makedirs('../../plots/{}/chromatin_networks'.format(args.dataset), exist_ok=True)
        plt.savefig('../../plots/{}/chromatin_networks/{}.png'.format(args.dataset, filename))


def multi_chromosome(args):
    hic_name = '{}_{}_{}_all_{}'.format(args.file, args.type, args.norm, args.window)

    shapes = [
        np.load('../../data/{}/hic/{}_{}_{}_{}_{}_{}.npy'.format(args.dataset, args.file, args.type, args.norm, i, i,
                                                                 args.window)).shape for i in range(1, 23)]

    mask = intra_mask(shapes)
    hic_full = np.load('../../data/{}/hic/{}.npy'.format(args.dataset, hic_name))

    if args.perc_inter is not None:
        threshold_inter = chromatin_threshold(args.dataset, args.file, args.type, args.norm, args.window,
                                              percentile_inter=args.perc_inter)

        hic_inter = hic_full * np.logical_not(mask)
        hic_inter[hic_inter < threshold_inter] = 0
        if not args.weighted:
            hic_inter[hic_inter > 0] = 1

        hic_name += '_{}_{}_{}_all_{}'.format(args.file_inter, args.type_inter, args.norm_inter, args.window_inter)

    else:
        hic_inter = intra_mask(shapes, nans=True, values=np.zeros)


    threshold_intra = chromatin_threshold(args.dataset, args.file, args.type, args.norm, args.window,
                                    percentile_intra=args.perc_intra)
    print(threshold_intra)
    hic_intra = hic_full * mask

    hic_intra[hic_intra < threshold_intra] = 0
    if not args.weighted:
        hic_intra[hic_intra > 0] = 1

    print(np.nansum(hic_intra), np.nansum(hic_inter))

    hic_thr = hic_intra + hic_inter

    if args.perc_inter is not None:
        filename = '{}{}_{}_{}'.format(hic_name,'_w' if args.weighted else '', threshold_intra, threshold_inter)
    else:
        filename = '{}{}_{}'.format(hic_name,'_w' if args.weighted else '', threshold_intra)

    save_disconnected_nodes(hic_thr, args.dataset, filename)

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
    parser.add_argument('--chr-src', type=int, default=None)
    parser.add_argument('--chr-tgt', type=int, default=None)

    parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='observed')
    parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR', 'ICE'], default='ICE')
    parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='primary')
    parser.add_argument('--resolution', type=int, default=40000)
    parser.add_argument('--window', type=int, default=40000)

    parser.add_argument('--type-inter', type=str, choices=['observed', 'oe'], default='observed')
    parser.add_argument('--norm-inter', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR', 'ICE'], default='ICE')
    parser.add_argument('--file-inter', type=str, choices=['primary', 'replicate', 'combined'], default='primary')
    parser.add_argument('--resolution-inter', type=int, default=40000)
    parser.add_argument('--window-inter', type=int, default=40000)

    parser.add_argument('--perc-intra', type=int, default=90)
    parser.add_argument('--perc-inter', type=int, default=None)
    parser.add_argument('--single-chrom', default=False, action='store_true')
    parser.add_argument('--weighted', default=False, action='store_true')
    parser.add_argument('--save-plot', default=True, action='store_true')
    parser.add_argument('--save-matrix', default=False, action='store_true')
    args = parser.parse_args()

    if args.chr_src and args.chr_tgt:
        single_chromosome(args)
    else:
        if not args.single_chrom:
            multi_chromosome(args)

        if args.perc_inter is None:
            for i in range(1, 23):
                args.chr_src = i
                args.chr_tgt = i
                single_chromosome(args)
