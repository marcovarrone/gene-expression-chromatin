import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='PRAD')
    parser.add_argument('--thr-intra', type=float, default=0.59)
    parser.add_argument('--thr-inter', type=float, default=0.78)
    parser.add_argument('--no-inter', default=False, action='store_true')
    parser.add_argument('--save', default=True, action='store_true')

    args = parser.parse_args()

    coexpression_full = np.load('data/{}/coexp/coexpression_chr_all.npy'.format(args.dataset))

    arrs = [np.load('data/{}/coexp/coexpression_chr_{:02d}_{:02d}.npy'.format(args.dataset, i, i)) for i in range(1, 23)]

    shapes = np.array([a.shape for a in arrs])
    mask = np.zeros(np.sum(shapes, axis=0))

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        mask[r:r + rr, c:c + cc] = np.ones((rr, cc))
        r += rr
        c += cc

    interactions = np.array([])
    if args.no_inter:
        coexpression_inter = np.zeros(coexpression_full.shape)
    else:
        coexpression_inter = coexpression_full*np.logical_not(mask)
    coexpression_inter[coexpression_inter < args.thr_inter] = 0
    coexpression_inter[coexpression_inter > 0] = 1

    coexpression_intra = coexpression_full*mask
    coexpression_intra[coexpression_intra < args.thr_intra] = 0
    coexpression_intra[coexpression_intra > 0] = 1

    print(coexpression_inter.sum(), coexpression_intra.sum())

    coexpression_thr = coexpression_intra + coexpression_inter


    if args.no_inter:
        filename = 'coexpression_chr_all_{}'.format(args.thr_intra)
    else:
        filename = 'coexpression_chr_all_{}_{}'.format(args.thr_intra, args.thr_inter)

    plt.figure(figsize=(7, 7), dpi=600)
    plt.imshow(coexpression_thr, cmap='Oranges')
    if args.save:
        print('save {}.npz'.format(filename))
        sps.save_npz(
            '/home/varrone/Prj/gene-expression-chromatin/src_old/link_prediction/data/{}/coexpression/{}.npz'.format(
                args.dataset,
                filename),
            sps.csr_matrix(coexpression_thr))

        if args.save:
            plt.savefig(
                '/home/varrone/Prj/gene-expression-chromatin/src_old/link_prediction/plots/{}/{}.png'.format(
                    args.dataset, filename))
    plt.show()
