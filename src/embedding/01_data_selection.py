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
parser.add_argument('-n', '--n-samples', type=int)
parser.add_argument('-d', '--dataset', type=str, default='GSE92743')
parser.add_argument('-o', '--offset', type=int, nargs='?', default=0)
parser.add_argument('--no-normalize', default=False, action='store_true')
parser.add_argument('-v', '--valid-size', type=float, nargs='?', default=0)
parser.add_argument('--split-genes', default=False, action='store_true')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()
np.random.seed(args.seed)

X = np.hstack((X_train, Y_train))
np.random.shuffle(X)

representation = ''

if args.split_genes:
    representation += '_genes'

if args.n_samples:
    representation += '_' + str(args.n_samples) + '_' + str(args.offset)
    print('Selecting samples from ' + str(args.offset) + ' to ' + str(args.offset + args.n_samples))

    X = X[args.offset:args.offset + args.n_samples]

# We should consider the samples as features of the genes
# (n_genes, n_samples)
X = X.T

if not args.no_normalize:
    print('Normalizing')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    representation += '_normalized'

if args.valid_size:
    print('Assigning ' + str(args.valid_size * 100) + '% of ' + (
        'genes' if args.split_genes else 'samples') + ' to validation set')

    shuffle = True
    if not args.split_genes:
        # (n_samples, n_genes)
        X = X.T
        shuffle = False
    X_train, X_valid = train_test_split(X, test_size=args.valid_size, shuffle=shuffle)
    if not args.split_genes:
        # (n_genes, n_samples)
        X_train, X_valid = X_train.T, X_valid.T

    representation += '_' + str(args.valid_size)

    print('Saving training set with ' + str(X_train.shape[0]) + ' genes and ' + str(X_train.shape[1]) + ' samples')
    np.save(str(data_folder) + '/' + str(args.dataset) + '/X_train' + str(representation), X_train)

    print('Saving validation set with ' + str(X_valid.shape[0]) + ' genes and ' + str(X_valid.shape[1]) + ' samples')
    np.save(str(data_folder) + '/' + str(args.dataset) + '/X_valid' + str(representation), X_valid)
else:
    print('Saving training set with size ' + str(X.shape[0]) + ' genes and ' + str(X.shape[1]) + ' samples')
    np.save(str(data_folder) + '/' + str(args.dataset) + '/X_train' + str(representation), X)
