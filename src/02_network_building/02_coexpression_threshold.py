import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='PRAD')
    parser.add_argument('--inter', default=True, action='store_true')
    parser.add_argument('--threshold', type=int, default=99.4)

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
    if args.inter:
        coexpression_masked = coexpression_full*np.logical_not(mask)
    else:
        coexpression_masked = coexpression_full*mask

    coexpression_masked[coexpression_masked ==0] = np.nan
    threshold = np.nanpercentile(coexpression_masked, args.threshold)
    print((np.triu(coexpression_masked, 1) >= threshold).sum())
    print(threshold)

