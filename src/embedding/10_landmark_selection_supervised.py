import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

parser = argparse.ArgumentParser()

# ToDo: add description
parser.add_argument('-t', '--training-set', type=str, default='X_train_1000_0_normalized_0.2')
parser.add_argument('-v', '--validation-set', type=str, default='X_valid_1000_0_normalized_0.2')
parser.add_argument('-n', '--n-landmarks', type=int)
parser.add_argument('-d', '--dataset', type=str, default='GSE92743')
parser.add_argument('--n-jobs', type=int, default=10)
parser.add_argument('--save-freqs', default=False, action='store_true')
parser.add_argument('--save-fig', default=False, action='store_true')
parser.add_argument('--save-all-ranks', default=False, action='store_true')
parser.add_argument('--save-coeffs', default=False, action='store_true')

parser.add_argument('--n-plotted', type=int, default=None)

args = parser.parse_args()

train = np.load('data/GSE92743/' + str(args.training_set) + '.npy').T
valid = np.load('data/GSE92743/' + str(args.validation_set) + '.npy').T

landmarks_occurrences = np.zeros(train.shape[1])
landmarks_rankings = np.zeros((train.shape[1], train.shape[1]))
landmarks_coeffs = np.zeros((train.shape[1], train.shape[1]))

filename_freqs = 'landmark_freqs_' + str(args.training_set) + '_' + str(args.n_landmarks)
filename_rankings = 'landmark_rankings_' + str(args.training_set) + '_' + str(args.n_landmarks)
filename_coeffs = 'landmark_coeffs_' + str(args.training_set) + '_' + str(args.n_landmarks)

if not args.n_landmarks:
    args.n_landmarks = train.shape[1]
if __name__ == '__main__':
    if os.path.isfile('landmark_freqs/' + str(args.dataset) + '/' + str(filename_freqs) + '.npy'):
        print("The values have been already computed. Loading from file")
        landmarks_occurrences = np.load('landmark_freqs/' + str(args.dataset) + '/' + str(filename_freqs) + '.npy')
    else:
        for i in range(8212, train.shape[1]):
            print("Target", i)
            target = [i]
            landmarks = np.arange(train.shape[1])
            landmarks = np.setdiff1d(landmarks, target)

            X_train = train[:, landmarks]
            Y_train = np.ravel(train[:, target])

            X_valid = valid[:, landmarks]
            Y_valid = np.ravel(valid[:, target])

            model = RandomForestRegressor(n_estimators=10, n_jobs=args.n_jobs, random_state=42)
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_valid)

            feature_importances = model.feature_importances_
            feature_importances = np.insert(feature_importances, i, 0)

            print(mean_absolute_error(Y_valid, Y_pred))

            if args.save_freqs or args.save_all_ranks:
                best_landmarks_idxs = np.argsort(feature_importances)[-args.n_landmarks:]
                best_landmarks = landmarks[best_landmarks_idxs]

            if args.save_freqs:
                landmarks_occurrences[best_landmarks] += 1

            if args.save_all_ranks:
                landmarks_rankings[i,:] = best_landmarks

            if args.save_coeffs:
                landmarks_coeffs[i,:] = feature_importances

        if args.save_freqs:
            np.save('landmark_freqs/' + str(args.dataset) + '/' + str(filename_freqs) + '8212.npy', landmarks_occurrences)

        if args.save_all_ranks:
            np.save('landmark_freqs/' + str(args.dataset) + '/' + str(filename_rankings) + '8212.npy', landmarks_coeffs)

        if args.save_coeffs:
            np.save('landmark_freqs/' + str(args.dataset) + '/' + str(filename_coeffs) + '8212.npy', landmarks_coeffs)

    #landmarks_idxs_sorted = np.argsort(-landmarks_occurrences)
    #landmarks_idxs_sorted = [str(x) for x in landmarks_idxs_sorted]
    '''n_landmarks = np.arange(landmarks_occurrences.shape[0])
    landmarks_occurrences_sorted = landmarks_occurrences[np.argsort(-landmarks_occurrences)]
    if args.n_plotted:
        n_landmarks = n_landmarks[:args.n_plotted]
        landmarks_occurrences_sorted = landmarks_occurrences_sorted[:args.n_plotted]
        filename_freqs += '_plt'+str(args.n_plotted)
    if args.save_fig:
        plt.xscale('log')
        plt.plot(n_landmarks, landmarks_occurrences_sorted)
        plt.savefig('landmark_freqs/' + str(args.dataset) + '/' + str(filename_freqs) + '.png')
'''