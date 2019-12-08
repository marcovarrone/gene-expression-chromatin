import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MCF7')
    parser.add_argument('--oe', default=False, action='store_true')
    parser.add_argument('--hic-intra', type=str, default='primary_oe_KR_{}_50000_50000_sum')
    parser.add_argument('--hic-inter', type=str, default='primary_oe_NONE_{}_250000_250000_sum')
    parser.add_argument('--interactions', default=False, action='store_true')
    parser.add_argument('--thr-intra', type=int, default=15.6159)
    parser.add_argument('--thr-inter', type=int, default=7.114)
    parser.add_argument('--no-intra', default=False, action='store_true')
    parser.add_argument('--save', default=False, action='store_true')
    args = parser.parse_args()

    if args.interactions:
        data_folder = '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/{}/interactions/interactions_'.format(
            args.dataset)
    else:
        data_folder = '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/hic/'.format(args.dataset)

    rows = []
    rows_weighted = []
    gene_idxs = {}
    cumulative_idx = 0
    for i in range(1, 23):
        row = None

        for j in range(1, 23):
            if i > j:
                hic = np.load(data_folder + args.hic_inter.format(str(j) + '_' + str(i)) + '{}.npy'.format(
                    ('_' + str(args.thr_inter)) if args.interactions else '')).T

                hic_weighted = hic.copy()
                if not args.interactions and args.thr_inter:
                    hic[hic <= args.thr_inter] = 0
                    hic[hic > 0] = 1
            elif i < j:
                hic = np.load(data_folder + args.hic_inter.format(str(i) + '_' + str(j)) + '{}.npy'.format(
                    ('_' + str(args.thr_inter)) if args.interactions else ''))

                hic_weighted = hic.copy()
                if not args.interactions and args.thr_inter:
                    hic[hic <= args.thr_inter] = 0
                    hic[hic > 0] = 1
            else:
                hic = np.load(data_folder + args.hic_intra.format(str(i) + '_' + str(j)) + '{}.npy'.format(
                    ('_' + str(args.thr_intra)) if args.interactions else ''))

                hic_weighted = hic.copy()
                if args.no_intra:
                    hic = np.empty(hic.shape)
                    hic[:] = np.nan
                else:
                    if not args.interactions and args.thr_intra:
                        hic[hic <= args.thr_intra] = 0
                        hic[hic > 0] = 1

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
                rows_weighted = hic_weighted
            else:
                row = np.hstack((row, hic))
                rows_weighted = np.hstack((rows_weighted, hic_weighted))

            if i == j:
                gene_idxs[i] = np.arange(cumulative_idx, cumulative_idx + hic.shape[0])
                cumulative_idx += hic.shape[0]

        rows.append(row)

    if args.save:
        if args.no_intra:
            with open(
                    '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/gene_idxs/'.format(
                        args.dataset) + args.hic_inter.format(
                        'all') + '.pkl', 'wb') as file_save:
                pickle.dump(gene_idxs, file_save)
        else:
            with open('/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/gene_idxs/'.format(
                    args.dataset) + args.hic_intra.format('all') + '.pkl', 'wb') as file_save:
                pickle.dump(gene_idxs, file_save)
    genome_hic = np.vstack(rows)

    # print(np.nanpercentile(genome_hic, 90))
    if args.save:
        print('save interactions_' + args.hic_intra.format(
                'all') + '{}{}.npy'.format('_oe' if args.oe else '',
                                           '_inter' if args.no_intra else ''))
        np.save(
            '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/{}/interactions/interactions_'.format(args.dataset) + args.hic_intra.format(
                'all') + '{}{}.npy'.format('_oe' if args.oe else '',
                                           '_inter' if args.no_intra else ''), genome_hic)
    plt.figure(figsize=(7, 7), dpi=600)
    plt.imshow(genome_hic, cmap='Oranges')
    plt.show()
