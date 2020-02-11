import argparse
import os

import numpy as np
import scipy.sparse as sps


def main(args):
    chromosomes = range(1, 23) if args.chromosomes is None else args.chromosomes

    dataset_path = '../../data/breast/primary_observed_ICE/hic_raw'
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    for i in chromosomes:
        print('Chromosome ', i)

        contact_matrix = np.genfromtxt(args.input.format(i), delimiter='\t')
        contact_matrix_sparse = sps.csr_matrix(contact_matrix)

        sps.save_npz(dataset_path + '/primary_observed_ICE_{}_{}_{}.npz'.format(i, i, args.resolution),
                     contact_matrix_sparse)
    print('Hi-C data saved in sparse format in', dataset_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True,
                        help='Path of the Hi-C matrices in .matrix format with "{}" instead of the number of the chromosome, e.g. ../../data/hic_data_chr{}.matrix')

    parser.add_argument('--resolution', type=int, required=True, help='Resolution of the Hi-C data.')
    parser.add_argument('--chromosomes', nargs='*', default=None,
                        help='List of chromosomes for which to extract the Hi-C data. If empty all the non-sexual chromosomes data will be extracted.')
    args = parser.parse_args()

    main(args)
