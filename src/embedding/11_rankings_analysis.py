import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--coeffs', type=str, required=True)
parser.add_argument('--plot-sum-coeffs', default=False, action='store_true')
parser.add_argument('--plot-rankings', default=False, action='store_true')
parser.add_argument('--n-landmarks', type=int, default=0)

args = parser.parse_args()

# ToDo: remove
coeffs = args.coeffs
new_coeffs = np.zeros((coeffs.shape[0], coeffs.shape[0]))
for i in range(coeffs.shape[0]):
    coeff_row = coeffs[i]
    new_coeff_row = np.insert(coeff_row, i, 0)
    if i < 10:
        print(new_coeff_row)
    new_coeffs[i, :] = new_coeff_row

if args.plot_sum_coeffs:
    sum_coeffs = np.sum(np.abs(new_coeffs), axis=1)
    best_landmarks = np.argsort(sum_coeffs)[-args.n_landmarks:]

if args.plot_rankings:
    best_run_landmarks = np.argsort(new_coeffs, axis=1)
    median_ranks = np.zeros(new_coeffs.shape[1])
    for landmark in range(new_coeffs.shape[1]):
        ranks_inverted = np.array(list(zip(*np.where(new_coeffs == landmark)))[1])
        ranks = new_coeffs.shape[1] - ranks_inverted - 1
        median_ranks[landmark] = np.median(ranks)
    best_landmarks = np.argsort(median_ranks)[-args.n_landmarks:]