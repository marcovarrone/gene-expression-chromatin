import argparse
import configparser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

chromosomes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

parser = argparse.ArgumentParser()

# ToDo: add description
parser.add_argument('--dataset', type=str, default='GM19238')
parser.add_argument('--gene-info', type=str, default='GRCh37_gene_info.txt')
parser.add_argument('--chromosome', type=int, choices=chromosomes, default=2)
parser.add_argument('--zero-median', default=True, action='store_true')
parser.add_argument('--save-plot', default=True, action='store_true')

args = parser.parse_args()

data_folder = 'data/{}/'.format(args.dataset)
rna_folder = data_folder + 'rna/'

if __name__ == '__main__':
    gene_info = pd.read_csv('/home/varrone/Data/' + str(args.dataset) + '/' + str(args.gene_info), '\t')
    gene_info = gene_info.drop_duplicates('Gene stable ID')
    gene_info = gene_info.rename(columns={'Gene stable ID': 'Gene_IDs'})
    gene_info = gene_info[gene_info['Chromosome/scaffold name'] == str(args.chromosome)]

    df = pd.read_csv(config['GM19238']['DATAFRAME'])
    df_rna = df.iloc[:, 3:df.columns.get_loc('RPKM_GM19257') + 1]
    df_rna = df_rna.dropna()
    df_rna = gene_info.merge(df_rna, on='Gene_IDs')
    df_rna = df_rna.sort_values('Transcription start site (TSS)')
    df_rna.to_csv(rna_folder + str(args.dataset) + '_chr_{:02d}_rna.csv'.format(args.chromosome))

    if args.zero_median:
        gene_exp = df_rna.iloc[:, 8:].to_numpy()
        exp_med = np.nanmedian(gene_exp, axis=0)
        gene_exp -= exp_med
        np.save(rna_folder + str(args.dataset) + '_chr_{:02d}_zero_median.npy'.format(args.chromosome), gene_exp)
        if args.save_plot:
            coexp = np.corrcoef(gene_exp)
            plt.imshow(np.rot90(1 - coexp), cmap='RdBu')
            plt.savefig('plots/coexpression_chr_{:02d}'.format(args.chromosome))
