import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='MCF7')
    parser.add_argument('--hic', type=str, default='primary_observed_NONE_all_10000_200000_sum')
    parser.add_argument('--inter', default=True, action='store_true')
    parser.add_argument('--save', default=False, action='store_true')


    args = parser.parse_args()

    data_folder = 'data/{}/hic/'.format(args.dataset)
    #hic_output_file = '{}_{}_{}_{}_{}_{}{}'.format(args.file, 'observed', args.norm, args.chr_src, args.chr_tgt,
    #                                                 args.resolution, '_sum' if args.aggregation == 'sum' else '')

    if not os.path.exists(data_folder + args.hic + '_oe.npy'):
        hic = np.load(data_folder + args.hic + ('_inter' if args.inter else '') + '.npy')
        hic[np.isnan(hic)] = 0
        hic_exp = hic.copy()

        total_reads = np.nansum(hic)
        for i in tqdm(range(hic_exp.shape[0])):
            reads_i = np.nansum(hic[i, :])
            for j in range(hic_exp.shape[1]):
                reads_j = np.nansum(hic[:, j])
                if reads_i and reads_j:
                    hic_exp[i, j] = (reads_i * reads_j) / total_reads
                else:
                    hic_exp[i, j] = 1

        hic_oe = hic / hic_exp

        plt.imshow(np.log1p(hic_oe), cmap='Oranges')
        plt.show()

        np.save(data_folder + args.hic + '_oe.npy', hic_oe)
    else:
        hic_oe = np.load(data_folder + args.hic + '_oe.npy')


    thr = np.nanpercentile(hic_oe, 90)
    print(thr)

    if args.inter:
        with open('/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/gene_idxs/{}.pkl'.format(args.dataset, args.hic), 'rb') as file_load:
            gene_idxs = pickle.load(file_load)
            for i in range(1, 2):
            #for i in range(1, 23):
                idxs_i = gene_idxs[i]
                for j in range(i+1,23):
                    idxs_j = gene_idxs[j]
                    hic_oe_slice = hic_oe[idxs_i[:, None], idxs_j]
                    plt.imshow(np.log1p(hic_oe_slice), cmap='Oranges')
                    plt.show()
                    if args.save:
                        file_name = args.hic.split('_all')
                        np.save(data_folder + file_name[0] + '_{}_{}'.format(i, j) + file_name[1] + '_oe.npy', hic_oe_slice)

                print('Chr {} done'.format(i))



