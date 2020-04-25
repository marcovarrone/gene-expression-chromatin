import argparse
import os

import numpy as np
import pandas as pd
import scipy.sparse as sps


def main(args):
    print('Loading bin coordinates')
    coords = pd.read_csv(args.bed_path, delimiter='\t', header=None)

    coords[1] = (coords[1] // args.resolution).astype(int)
    coords[2] = (coords[2] // args.resolution).astype(int)

    print('Loading Hi-C data')
    values = pd.read_csv(args.input, delimiter='\t',
                         header=None)
    values = sps.csr_matrix((values.iloc[:, 2], (values.iloc[:, 0], values.iloc[:, 1])))

    chromosomes = range(1, 23) if args.chromosomes is None else args.chromosomes

    dataset_path = '../../data/prostate_cancer/hic_raw'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)

    for i, chr_source in enumerate(chromosomes):
        print('Chromosome ', chr_source)
        if args.inter:
            chromosomes_target = chromosomes[i:]
        else:
            chromosomes_target = [chr_source]

        for chr_target in chromosomes_target:
            if os.path.exists(dataset_path + '/hic_raw_{}_{}_{}.npz'.format(chr_source, chr_target,
                                                                                         args.window)):
                print('File already existing. Skip.')
                continue
            coords_chr_src = coords[coords[0] == 'chr{}'.format(chr_source)]
            coords_chr_tgt = coords[coords[0] == 'chr{}'.format(chr_target)]
            min_src = np.min(coords_chr_src[3])
            max_src = np.max(coords_chr_src[3])

            min_tgt = np.min(coords_chr_tgt[3])
            max_tgt = np.max(coords_chr_tgt[3])

            values_chr = values[min_src:max_src, min_tgt:max_tgt]
            values_chr = values_chr.toarray()
            values_chr = np.add.reduceat(values_chr, np.arange(0, values_chr.shape[0], args.window // args.resolution),
                                         axis=0)
            values_chr = np.add.reduceat(values_chr, np.arange(0, values_chr.shape[1], args.window // args.resolution),
                                         axis=1)
            sps.save_npz(
                dataset_path + '/hic_raw_{}_{}_{}.npz'.format(chr_source, chr_target, args.window),
                sps.csr_matrix(values_chr))
            print('Hi-C data saved in sparse format in', dataset_path + '/hic_raw_{}_{}_{}.npz'.format(chr_source, chr_target, args.window))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--bed-path', type=str, required=True,
                        help='Path for the bed file associating bin coordinates with bin id.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path of the original Hi-C matrix')
    parser.add_argument('--resolution', type=int, required=True,
                        help='Resolution of the Hi-C data.')
    parser.add_argument('--window', type=int, required=True,
                        help='Resolution of the Hi-C data.')
    parser.add_argument('--chromosomes', nargs='*', default=None,
                        help='List of chromosomes for which to extract the Hi-C data. If empty all the non-sexual chromosomes data will be extracted.')
    parser.add_argument('--inter', default=False, action='store_true',
                        help='Extract also interchromosomal interactions')

    args = parser.parse_args()

    if args.window % args.resolution != 0:
        raise ValueError('window must be a multiple of the resolution')

    main(args)
