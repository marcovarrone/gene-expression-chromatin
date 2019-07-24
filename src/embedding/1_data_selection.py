import argparse
import configparser

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

X_train = np.load(config['GSE_UNSCALED']['X_TRAIN'])
Y_train = np.load(config['GSE_UNSCALED']['Y_TRAIN'])
data_folder = config['EMBEDDING']['DATA']

parser = argparse.ArgumentParser()

# ToDo: add description
parser.add_argument('--n_samples', type=int)
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('--offset', type=int, nargs='?', default=0)
parser.add_argument('--no-normalize', default=False, action='store_true')
parser.add_argument('--valid-size', type=float, nargs='?', default=0)

args = parser.parse_args()

X = np.hstack((X_train, Y_train))
np.random.shuffle(X)

representation = ''

if args.n_samples:
    representation += '_genes_' + str(args.n_samples) + '_' + str(args.offset)
    print('Selecting samples from ' + str(args.offset) + ' to ' + str(args.offset + args.n_samples))

    X = X[args.offset:args.offset + args.n_samples]
    X = X.T

    genes_dim = 0
    samples_dim = 1
else:
    genes_dim = 1
    samples_dim = 0

if not args.no_normalize:
    print('Normalizing')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    representation += '_normalized'

if args.valid_size:
    print('Assigning ' + str(args.valid_size * 100) + '% of genes to validation set')
    X_train, X_valid = train_test_split(X, test_size=args.valid_size)

    representation += '_' + str(args.valid_size)

    print('Saving training set with ' + str(X_train.shape[genes_dim]) + ' genes and ' + str(X_train.shape[samples_dim]) + ' samples')
    np.save(str(data_folder) + '/' + str(args.dataset) + '/X_train' + str(representation), X_train)

    print('Saving validation set with ' + str(X_valid.shape[genes_dim]) + ' genes and ' + str(X_valid.shape[samples_dim]) + ' samples')
    np.save(str(data_folder) + '/' + str(args.dataset) + '/X_valid' + str(representation), X_valid)
else:
    print('Saving training set with size ' + str(X.shape[genes_dim]) + ' genes and ' + str(X.shape[samples_dim]) + ' samples')
    np.save(str(data_folder) + '/' + str(args.dataset) + '/X_train' + str(representation), X)
