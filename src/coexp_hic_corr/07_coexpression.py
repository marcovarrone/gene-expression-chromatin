import argparse

import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='GM19238')
    parser.add_argument('--chr-src', type=int, default=1)
    parser.add_argument('--chr-tgt', type=int, default=2)
    args = parser.parse_args()
    expression_src = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/rna/GM19238_chr_{:02d}_zero_median.npy'.format(
            args.dataset, args.chr_src))
    expression_tgt = np.load(
        '/home/varrone/Prj/gene-expression-chromatin/src/coexp_hic_corr/data/{}/rna/GM19238_chr_{:02d}_zero_median.npy'.format(
            args.dataset, args.chr_tgt))

    coexpression = np.corrcoef(expression_src, expression_tgt)[:expression_src.shape[0], -expression_tgt.shape[0]:]
    np.save('data/{}/coexp/coexpression_chr_{:02d}_{:02d}.npy'.format(args.dataset, args.chr_src, args.chr_tgt), coexpression)
