import argparse
import configparser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

parser = argparse.ArgumentParser()

# ToDo: add description
parser.add_argument('--dataset', type=str, default='PRAD')
parser.add_argument('--chr-src', type=int, default=3)
parser.add_argument('--chr-tgt', type=int, default=3)
parser.add_argument('--zero-median', default=False, action='store_true')
parser.add_argument('--save-plot', default=True, action='store_true')
parser.add_argument('--save-coexp', default=True, action='store_true')

args = parser.parse_args()

data_folder = 'data/{}/'.format(args.dataset)
rna_folder = data_folder + 'rna/'

if __name__ == '__main__':
    df = pd.read_csv(config[args.dataset]['DATAFRAME'])
    df = df.dropna()
    df = df.sort_values('Transcription start site (TSS)')


    if args.chr_src == -1:
        df_chr_src = df
        df_chr_tgt = df
    else:
        df_chr_src = df[df['Chromosome/scaffold name'] == args.chr_src]
        df_chr_tgt = df[df['Chromosome/scaffold name'] == args.chr_tgt]

    if args.chr_src == args.chr_tgt:
        df_chr_src.to_csv(rna_folder + str(args.dataset) + '_chr_{:02d}_rna.csv'.format(args.chr_src))

    gene_exp_src = df_chr_src.iloc[:, 5:].to_numpy()
    gene_exp_tgt = df_chr_tgt.iloc[:, 5:].to_numpy()
    if args.zero_median:
        exp_med = np.nanmedian(gene_exp_src, axis=0)
        gene_exp_src -= exp_med

        exp_med = np.nanmedian(gene_exp_tgt, axis=0)
        gene_exp_tgt -= exp_med

    if args.chr_src == args.chr_tgt:
        np.save(rna_folder + str(args.dataset) + '_chr_{:02d}{}.npy'.format(args.chr_src,
                                                                            '_zero_median' if args.zero_median else ''),
                gene_exp_src)

    coexp = np.corrcoef(gene_exp_src, gene_exp_tgt)[:gene_exp_src.shape[0], -gene_exp_tgt.shape[0]:]

    if args.save_plot:
        plt.imshow(1 - coexp, cmap='RdBu')
        plt.savefig('plots/{}/coexpression_chr_{:02d}_{:02d}{}.png'.format(args.dataset, args.chr_src, args.chr_tgt,
                                                                           '_zero_median' if args.zero_median else ''))
        plt.show()

    if args.save_coexp:
        np.save('data/{}/coexp/coexpression_chr_{:02d}_{:02d}{}.npy'.format(args.dataset, args.chr_src, args.chr_tgt, '_zero_median' if args.zero_median else ''),
                coexp)
