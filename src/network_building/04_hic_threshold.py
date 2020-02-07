import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='PRAD')
    parser.add_argument('--type', type=str, choices=['observed', 'oe'], default='observed')
    parser.add_argument('--norm', type=str, choices=['NONE', 'VC', 'VC_SQRT', 'KR', 'ICE'], default='KR')
    parser.add_argument('--file', type=str, choices=['primary', 'replicate', 'combined'], default='primary')
    parser.add_argument('--inter', default=True, action='store_true')
    parser.add_argument('--threshold', type=int, default=90)
    parser.add_argument('--resolution', type=int, default=40000)
    parser.add_argument('--window', type=int, default=40000)
    parser.add_argument('--aggregation', type=str, choices=['median', 'sum', 'max', None], default=None)

    args = parser.parse_args()

    interactions = np.array([])
    if args.inter:
        for i in range(1, 23):
            print('Chr. ', i)
            for j in range(i + 1, 23):
                hic_name = '{}_{}_{}_{}_{}_{}_{}{}'.format(args.file, args.type, args.norm, i, j,
                                                            args.resolution, args.window, ('_' + args.aggregation) if args.aggregation else '')
                try:
                    hic = np.load('data/{}/hic/{}.npy'.format(args.dataset, hic_name))
                except FileNotFoundError:
                    os.system(
                        'python3 02_hic_plot_juicer.py --file {} --type {} --norm {} --chr-src {} --chr-tgt {} --resolution {} --window {} --aggregation {} --save-matrix --save-plot'.format(
                            args.file, args.type, args.norm, i, j, args.resolution, args.window, args.aggregation))
                    hic = np.load('data/{}/hic/{}.npy'.format(args.dataset, hic_name))

                interactions = np.concatenate((interactions, hic[hic>0].flatten()))
    else:
        for i in range(1, 23):
            hic_name = '{}_{}_{}_{}_{}_{}_{}{}'.format(args.file, args.type, args.norm, i, i,
                                                        args.resolution, args.window, ('_' + args.aggregation) if args.aggregation else '')
            try:
                hic = np.load('data/{}/hic/{}.npy'.format(args.dataset, hic_name))
            except FileNotFoundError:
                os.system(
                    'python3 02_hic_plot_juicer.py --file {} --type {} --norm {} --chr-src {} --chr-tgt {} --resolution {} --window {} --aggregation {} --save-matrix --save-plot'.format(
                        args.file, args.type, args.norm, i, i, args.resolution, args.window, args.aggregation))
                hic = np.load('data/{}/hic/{}.npy'.format(args.dataset, hic_name))
            #hic += hic.T
            np.fill_diagonal(hic, 0)
            plt.imshow(np.log1p(hic), cmap='Oranges')
            plt.show()
            #print(np.nanpercentile(hic.flatten(), args.threshold))
            interactions = np.concatenate((interactions, hic[hic>0].flatten()))

    threshold = np.nanpercentile(interactions, args.threshold)
    print('N. edges', (interactions >= threshold).sum())
    print(threshold)