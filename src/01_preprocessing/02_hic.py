import os
import pandas as pd
import scipy.sparse as sps
import numpy as np

if __name__ == '__main__':
    coords = pd.read_csv('/home/varrone/Data/PRAD/GSE118629_hg19_10k.bed', delimiter='\t', header=None)
    coords[1] = (coords[1] // 10000).astype(int)
    coords[2] = (coords[2] // 10000).astype(int)

    if not os.path.exists('/home/varrone/Data/PRAD/Hi-C_10kb.npz'):
        values = pd.read_csv('/home/varrone/Data/PRAD/GSE118629_22Rv1_HiC_10k.normalized.matrix.txt', delimiter='\t',
                             header=None)
        values_csr = sps.csr_matrix((values.iloc[:, 2], (values.iloc[:, 0], values.iloc[:, 1])))
        sps.save_npz('/home/varrone/Data/PRAD/Hi-C_10kb.npz', values_csr)

    values = sps.load_npz('/home/varrone/Data/PRAD/Hi-C_10kb.npz')

    for i in range(1, 23):
        for j in range(i+1, 23):
            chr_src = i
            chr_tgt = j

            coords_chr_src = coords[coords[0] == 'chr{}'.format(chr_src)]
            coords_chr_tgt = coords[coords[0] == 'chr{}'.format(chr_tgt)]

            min_src = np.min(coords_chr_src[3])
            max_src = np.max(coords_chr_src[3])

            min_tgt = np.min(coords_chr_tgt[3])
            max_tgt = np.max(coords_chr_tgt[3])
            values_chr = values[min_src:max_src, min_tgt:max_tgt]
            sps.save_npz('/home/varrone/Data/PRAD/primary_observed_KR/primary_{}_{}_10000.npz'.format(chr_src, chr_tgt), values_chr)

    pass



