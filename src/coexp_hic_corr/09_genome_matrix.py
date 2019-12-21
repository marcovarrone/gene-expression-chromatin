import argparse

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='PRAD')
    parser.add_argument('--oe', default=False, action='store_true')
    parser.add_argument('--hic-intra', type=str, default='primary_observed_KR_{}_40000_40000')
    parser.add_argument('--hic-inter', type=str, default='primary_observed_KR_{}_40000_40000')
    parser.add_argument('--interactions', default=False, action='store_true')
    parser.add_argument('--thr-intra', type=int, default=7.885)
    parser.add_argument('--thr-inter', type=int, default=0)
    parser.add_argument('--no-intra', default=False, action='store_true')
    parser.add_argument('--save', default=True, action='store_true')
    parser.add_argument('--save-genes-chr', default=True, action='store_true')
    args = parser.parse_args()

    if args.interactions:
        data_folder = '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/{}/interactions/interactions_'.format(
            args.dataset)
    else:
        data_folder = '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/hic/'.format(args.dataset)

    rows = []
    # rows_weighted = []
    genes_chr = np.array([])
    for i in range(1, 23):
        row = None
        for j in range(1, 23):
            if i > j:
                hic = np.load(data_folder + args.hic_inter.format(str(j) + '_' + str(i)) + '{}.npy'.format(
                    ('_' + str(args.thr_inter)) if args.interactions else '')).T

                # hic_weighted = hic.copy()
                if not args.interactions and args.thr_inter is not None:
                    hic[hic <= args.thr_inter] = 0
                    hic[hic > 0] = 1
            elif i < j:
                hic = np.load(data_folder + args.hic_inter.format(str(i) + '_' + str(j)) + '{}.npy'.format(
                    ('_' + str(args.thr_inter)) if args.interactions else ''))

                # hic_weighted = hic.copy()
                if not args.interactions and args.thr_inter is not None:
                    hic[hic <= args.thr_inter] = 0
                    hic[hic > 0] = 1
            else:
                hic = np.load(data_folder + args.hic_intra.format(str(i) + '_' + str(j)) + '{}.npy'.format(
                    ('_' + str(args.thr_intra)) if args.interactions else ''))

                # hic_weighted = hic.copy()
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
                # rows_weighted = hic_weighted
            else:
                row = np.hstack((row, hic))
                # rows_weighted = np.hstack((rows_weighted, hic_weighted))

            if i == j:
                genes_chr = np.concatenate((genes_chr, np.array([i] * hic.shape[0])))

        rows.append(row)

    genome_hic = np.vstack(rows)
    degrees = genome_hic.sum(axis=1)
    disconnected_nodes = np.where(degrees == 0)[0]
    genes_chr[disconnected_nodes] = 0
    print("Disconnected nodes: ", disconnected_nodes)

    if args.save_genes_chr:
        if args.no_intra:
            np.save('/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/genes_chr/'.format(
                args.dataset) + args.hic_inter.format('all') + '.npy', genes_chr)
        else:
            np.save('/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/genes_chr/'.format(
                args.dataset) + args.hic_intra.format('all') + '_' + args.hic_inter.format('all') + '.npy', genes_chr)

    # print(np.nanpercentile(genome_hic, 90))
    if args.save:
        print('save interactions_' + args.hic_intra.format(
            'all') + '_' + args.hic_inter.format('all') + '{}{}_no_inter.npy'.format('_oe' if args.oe else '',
                                       '_inter' if args.no_intra else ''))
        np.save(
            '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/data/{}/interactions/interactions_'.format(
                args.dataset) + args.hic_intra.format(
                'all') + '_' + args.hic_inter.format('all') + '{}{}.npy'.format('_oe' if args.oe else '',
                                           '_inter' if args.no_intra else '') , genome_hic)
    plt.figure(figsize=(7, 7), dpi=600)
    plt.imshow(genome_hic, cmap='Oranges')
    if args.save:
        plt.savefig(
            '/home/varrone/Prj/gene-expression-chromatin/src/link_prediction/plots/{}/interactions_{}.png'.format(
                args.dataset, args.hic_intra.format(
                'all') + '_' + args.hic_inter.format('all') + '{}{}'.format('_oe' if args.oe else '',
                                           '_inter' if args.no_intra else '')))
    plt.show()
