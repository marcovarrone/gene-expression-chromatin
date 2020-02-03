import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='PRAD')
    parser.add_argument('--oe', default=False, action='store_true')
    parser.add_argument('--hic-intra', type=str, default='primary_observed_KR_{}_40000_40000')
    parser.add_argument('--hic-inter', type=str, default='primary_observed_KR_{}_40000_40000')
    parser.add_argument('--interactions', default=False, action='store_true')
    parser.add_argument('--thr-intra', type=float, default=2.74)
    parser.add_argument('--weight-intra', type=float, default=1)
    parser.add_argument('--weight-inter', type=float, default=0.5)
    parser.add_argument('--thr-inter', type=float, default=1.23)
    parser.add_argument('--no-intra', default=False, action='store_true')
    parser.add_argument('--no-inter', default=False, action='store_true')
    parser.add_argument('--save', default=True, action='store_true')
    args = parser.parse_args()

    if args.interactions:
        data_folder = '/home/varrone/Prj/gene-expression-chromatin/src_old/link_prediction/data/{}/interactions/interactions_'.format(
            args.dataset)
    else:
        data_folder = '/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/hic/'.format(args.dataset)

    if not os.path.exists('/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/chr_sizes.npy'.format(
            args.dataset)):
        chr_sizes = np.empty(23)
        np.save(
            '/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/chr_sizes.npy'.format(args.dataset),
            chr_sizes)
    else:
        chr_sizes = np.load(
            '/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/chr_sizes.npy'.format(args.dataset))

    if args.no_inter:
        filename = '{}_{}'.format(args.hic_intra.format('all'), args.thr_intra)
    elif args.no_intra:
        raise NotImplementedError
    else:
        filename = '{}_{}_{}_{}_{}_{}'.format(args.hic_intra.format('all'), args.thr_intra, args.weight_intra, args.hic_inter.format('all'),
                                        args.thr_inter, args.weight_inter)

    rows = []
    sum_inter = 0
    sum_intra = 0
    for i in range(1, 23):
        row = None
        for j in range(1, 23):
            if i > j:
                if args.no_inter:
                    hic = np.load(data_folder + args.hic_intra.format(str(i) + '_' + str(i)) + '{}.npy'.format(
                        ('_' + str(args.thr_intra)) if args.interactions else ''))
                    shape_0 = hic.shape[0]
                    hic = np.load(data_folder + args.hic_intra.format(str(j) + '_' + str(j)) + '{}.npy'.format(
                        ('_' + str(args.thr_intra)) if args.interactions else ''))
                    shape_1 = hic.shape[1]

                    hic = np.zeros((shape_0, shape_1))
                else:
                    hic = np.load(data_folder + args.hic_inter.format(str(j) + '_' + str(i)) + '{}.npy'.format(
                        ('_' + str(args.thr_inter)) if args.interactions else '')).T

                    # hic_weighted = hic.copy()
                    if not args.interactions and args.thr_inter is not None:
                        hic[hic < args.thr_inter] = -1
                        hic[hic > 0] = 1*args.weight_inter
                sum_inter += (hic> 0).sum()
            elif i < j:
                if args.no_inter:
                    hic = np.load(data_folder + args.hic_intra.format(str(i) + '_' + str(i)) + '{}.npy'.format(
                        ('_' + str(args.thr_intra)) if args.interactions else ''))
                    shape_0 = hic.shape[0]
                    hic = np.load(data_folder + args.hic_intra.format(str(j) + '_' + str(j)) + '{}.npy'.format(
                        ('_' + str(args.thr_intra)) if args.interactions else ''))
                    shape_1 = hic.shape[1]

                    hic = np.zeros((shape_0, shape_1))
                else:
                    hic = np.load(data_folder + args.hic_inter.format(str(i) + '_' + str(j)) + '{}.npy'.format(
                        ('_' + str(args.thr_inter)) if args.interactions else ''))

                    # hic_weighted = hic.copy()
                    if not args.interactions and args.thr_inter is not None:
                        hic[hic < args.thr_inter] = -1
                        hic[hic > 0] = 1*args.weight_inter

                sum_inter += (hic > 0).sum()
            else:
                hic = np.load(data_folder + args.hic_intra.format(str(i) + '_' + str(j)) + '{}.npy'.format(
                    ('_' + str(args.thr_intra)) if args.interactions else ''))

                # hic_weighted = hic.copy()
                if args.no_intra:
                    hic = np.empty(hic.shape)
                    hic[:] = np.nan
                else:
                    if not args.interactions and args.thr_intra:
                        hic[hic <= args.thr_intra] = -1
                        hic[hic > 0] = 1*args.weight_intra
                sum_intra += (hic > 0).sum()

                # Update chromosome sizes
                chr_sizes[i] = hic.shape[0]

            if args.oe and i != j:
                hic_exp = hic.copy()

                total_reads = np.nansum(hic)
                for k in range(hic_exp.shape[0]):
                    reads_k = np.nansum(hic[k, :])
                    for l in range(hic_exp.shape[1]):
                        reads_l = np.nansum(hic[:, l])
                        if reads_k and reads_l:
                            hic_exp[k, l] = (reads_k * reads_l) / total_reads
                        else:
                            hic_exp[k, l] = 1

                # plt.imshow(np.log1p(hic), cmap='Oranges')
                # plt.show()
                hic /= hic_exp
                # plt.imshow(np.log1p(hic*10), cmap='Oranges')
                # plt.show()

            if row is None:
                row = hic
                # rows_weighted = hic_weighted
            else:
                row = np.hstack((row, hic))
                # rows_weighted = np.hstack((rows_weighted, hic_weighted))

        rows.append(row)

    print(sum_inter, sum_intra)
    np.save('/home/varrone/Prj/gene-expression-chromatin/src_old/coexp_hic_corr/data/{}/chr_sizes.npy'.format(args.dataset),
            chr_sizes)

    genome_hic = np.vstack(rows)

    # print(np.nanpercentile(genome_hic, 90))
    if args.save:
        print('save interactions_{}.npz'.format(filename))
        sps.save_npz(
            '/home/varrone/Prj/gene-expression-chromatin/src_old/link_prediction/data/{}/interactions/interactions_{}.npz'.format(
                args.dataset, filename),
            sps.csr_matrix(genome_hic))
        plt.figure(figsize=(7, 7), dpi=600)
        plt.imshow(genome_hic, cmap='Oranges')
        if args.save:
            plt.savefig(
                '/home/varrone/Prj/gene-expression-chromatin/src_old/link_prediction/plots/{}/interactions_{}.png'.format(
                    args.dataset, filename))
        plt.show()
