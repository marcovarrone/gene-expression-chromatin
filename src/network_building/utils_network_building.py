import os

import numpy as np
import matplotlib.pyplot as plt

from utils import intra_mask



def coexpression_threshold(dataset, percentile_intra=None, percentile_inter=None):
    if not percentile_intra and not percentile_inter:
        raise ValueError(
            'Either one parameter between percentile_intra and percentile_inter must be different from zero.')

    coexpression_full = np.load('../../data/{}/coexpression/coexpression_chr_all_all.npy'.format(dataset))

    try:
        shapes = [np.load('../../data/{}/coexpression/coexpression_chr_{}_{}.npy'.format(dataset, i, i)).shape for i in
                  range(1, 23)]
    except FileNotFoundError:
        raise ValueError('You need to compute the co-expression for all the chromosomes first.')

    mask = intra_mask(shapes)

    if percentile_intra:
        coexpression_intra = coexpression_full * mask
        coexpression_intra[coexpression_intra == 0] = np.nan
        threshold_intra = np.round(np.nanpercentile(coexpression_intra, percentile_intra), 2)

    if percentile_inter:
        coexpression_inter = coexpression_full * np.logical_not(mask)
        coexpression_inter[coexpression_inter == 0] = np.nan
        threshold_inter = np.round(np.nanpercentile(coexpression_inter, percentile_inter), 2)

    if percentile_intra and percentile_inter:
        return threshold_intra, threshold_inter
    elif percentile_intra:
        return threshold_intra
    else:
        return threshold_inter


def chromatin_threshold(dataset, file, type, norm, window, percentile_intra=None, percentile_inter=None, ):
    if not percentile_intra and not percentile_inter:
        raise ValueError(
            'Either one parameter between percentile_intra and percentile_inter must be different from zero.')

    if os.path.exists('../../data/prostate/hic/{}_{}_{}_all_{}.npy'.format(file, type, norm, window)):
        hic_full = np.load('../../data/prostate/hic/{}_{}_{}_all_{}.npy'.format(file, type, norm, window))
        try:
            shapes = [
                np.load('../../data/{}/hic/{}_{}_{}_{}_{}_{}.npy'.format(dataset, file, type, norm, i, i, window)).shape
                for i in range(1, 23)]
        except FileNotFoundError:
            raise ValueError('You need to compute the co-expression for all the chromosomes first.')
    else:
        print('Full genome-wide Hi-C not present. Building...')
        shapes = []
        rows = []
        for i in range(1, 23):
            row = []
            for j in range(1, 23):

                if i <= j:
                    hic = np.load(
                        '../../data/{}/hic/{}_{}_{}_{}_{}_{}.npy'.format(dataset, file, type, norm, i, j, window))
                    row.append(hic)
                else:
                    hic = np.load(
                        '../../data/{}/hic/{}_{}_{}_{}_{}_{}.npy'.format(dataset, file, type, norm, j, i, window)).T
                    hic = np.empty(hic.shape)
                    hic[:] = np.nan
                    row.append(hic)

                if i == j:
                    shapes.append(hic.shape)
            rows.append(np.hstack(row))
        hic_full = np.vstack(rows)
        np.save('../../data/{}/hic/{}_{}_{}_all_{}.npy'.format(dataset, file, type, norm, window), hic_full)
        plt.figure(figsize=(7, 7), dpi=600)
        plt.imshow(np.log1p(hic_full), cmap='Reds')
        plt.savefig('../../plots/{}/hic/{}_{}_{}_all_{}.png'.format(dataset, file, type, norm, window))
        print('Full genome-wide Hi-C saved in ../../data/{}/hic/{}_{}_{}_all_{}.npy'.format(dataset, file, type, norm,
                                                                                            window))

    mask = intra_mask(shapes)

    if percentile_intra:
        hic_intra = hic_full * mask
        hic_intra[hic_intra == 0] = np.nan
        threshold_intra = np.round(np.nanpercentile(hic_intra, percentile_intra), 2)

    if percentile_inter:
        hic_inter = hic_full * np.logical_not(mask)
        hic_inter[hic_inter == 0] = np.nan
        threshold_inter = np.round(np.nanpercentile(hic_inter, percentile_inter), 2)

    if percentile_intra and percentile_inter:
        return threshold_intra, threshold_inter
    elif percentile_intra:
        return threshold_intra
    else:
        return threshold_inter
