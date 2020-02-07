import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main(args):
    data_folder = '../../data/{}/'.format(args.dataset)
    rna_folder = data_folder + 'rna/'
    os.makedirs(rna_folder, exist_ok=True)

    df = pd.read_csv(data_folder + 'expression_raw.csv')
    df = df.dropna()
    df = df.sort_values('Transcription start site (TSS)')

    # ToDo: implement single source with whole genome target
    if args.chr_src and args.chr_tgt:
        chr_src = args.chr_src
        chr_tgt = args.chr_tgt
        df_chr_src = df[df['Chromosome/scaffold name'] == chr_src]
        df_chr_tgt = df[df['Chromosome/scaffold name'] == chr_tgt]
        print('Computing co-expression between chromosome', chr_src,'and chromosome', chr_tgt)
    else:
        chr_src = 'all'
        chr_tgt = 'all'
        df_chr_src = df
        df_chr_tgt = df
        print('Computing co-expression for all the chromosomes together')

    if chr_src == chr_tgt:
        df_chr_src.to_csv(rna_folder + 'expression_info_chr_{}.csv'.format(chr_src))

    gene_exp_src = df_chr_src.iloc[:, 5:].to_numpy()
    gene_exp_tgt = df_chr_tgt.iloc[:, 5:].to_numpy()

    '''if chr_src == chr_tgt:
        np.save(rna_folder + str(args.dataset) + '_chr_{}.npy'.format(chr_src), gene_exp_src)'''

    coexp = np.corrcoef(gene_exp_src, gene_exp_tgt)[:gene_exp_src.shape[0], -gene_exp_tgt.shape[0]:]
    coexp[np.tril_indices_from(coexp, k=1)] = np.nan

    if args.save_plot:
        os.makedirs('../../plots/{}/coexpression'.format(args.dataset), exist_ok=True)

        plt.imshow(1 - coexp, cmap='RdBu')
        plt.savefig('../../plots/{}/coexpression/coexpression_chr_{}_{}.png'.format(args.dataset, chr_src, chr_tgt))
        plt.show()

    if args.save_coexp:
        os.makedirs(data_folder + 'coexpression', exist_ok=True)

        np.save(data_folder + 'coexpression/coexpression_chr_{}_{}.npy'.format(chr_src, chr_tgt),
                coexp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=None, required=True,
                        help='Name used to identify the dataset')
    parser.add_argument('--chr-src', type=int, default=None,
                        help='Source chromosome. If empty all the chromosomes are considered at once')
    parser.add_argument('--chr-tgt', type=int, default=None,
                        help='Target chromosome')
    parser.add_argument('--save-plot', default=False, action='store_true',
                        help='Save co-expression plot in results folder')
    parser.add_argument('--save-coexp', default=False, action='store_true',
                        help='Save co-expression numpy data in data/coexp folder')

    args = parser.parse_args()
    main(args)

    if args.chr_src is None:
        for i in range(1,23):
            args.chr_src = i
            args.chr_tgt = i
            main(args)

