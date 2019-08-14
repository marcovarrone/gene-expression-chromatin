import argparse
import configparser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--coeffs', type=str, required=True)
parser.add_argument('--plot-sum-coeffs', default=False, action='store_true')
parser.add_argument('--plot-rankings', default=False, action='store_true')
parser.add_argument('--save-landmarks', default=False, action='store_true')
parser.add_argument('--n-landmarks', type=int, default=0)

args = parser.parse_args()

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

coeffs = np.load('landmark_freqs/' + str(args.dataset) + '/' + str(args.coeffs) + '.npy')

filename = str(args.coeffs.replace('landmark_coeffs_', '')) + '_' + str(args.n_landmarks)

if args.plot_sum_coeffs:
    sum_coeffs = np.sum(np.abs(coeffs), axis=0)
    best_landmarks = np.argsort(-sum_coeffs)[:args.n_landmarks]
    coeffs_ordered = -sum_coeffs[best_landmarks]
    # plt.plot(np.arange(len(coeffs_ordered)), coeffs_ordered)
    # plt.savefig(
    #    'landmark_freqs/' + str(args.dataset) + '/sum_coeffs_' + str(args.coeffs).replace('landmark_coeffs', '')
    #    + '_' + str(args.n_landmarks) + '.png')

    if args.save_landmarks:
        gene_info = pd.read_csv(config[args.dataset]['GENE_INFO'], delimiter='\t')
        gene_symbols = gene_info.iloc[best_landmarks, :]['gene_symbol']
        gene_symbols.to_csv('landmark_freqs/' + str(args.dataset) + '/sum_coeffs_david_' + str(filename) + '.csv',
                            index=False, header=False)

if args.plot_rankings:
    best_run_landmarks = np.argsort(-coeffs, axis=1)
    median_ranks = np.zeros(coeffs.shape[1])
    ranks_fast = np.zeros(best_run_landmarks.shape)

    for landmark in range(coeffs.shape[1]):
        print(landmark)
        ranks = np.where(best_run_landmarks == landmark)[1]
        median_ranks[landmark] = np.median(ranks)

    best_landmarks = np.argsort(median_ranks)[:args.n_landmarks]
    best_ranks = median_ranks[best_landmarks]
    plt.plot(np.arange(len(best_ranks)), best_ranks)
    plt.savefig(
        'landmark_freqs/' + str(args.dataset) + '/ranks_' + str(args.coeffs).replace('landmark_coeffs', '') + '_' + str(
            args.n_landmarks) + '.png')

    if args.save_landmarks:
        gene_info = pd.read_csv(config[args.dataset]['GENE_INFO'], delimiter='\t')
        gene_symbols = gene_info.iloc[best_landmarks, :]['gene_symbol']
        gene_symbols.to_csv('landmark_freqs/' + str(args.dataset) + '/ranks_david_' + str(filename) + '.csv',
                            index=False, header=False)


