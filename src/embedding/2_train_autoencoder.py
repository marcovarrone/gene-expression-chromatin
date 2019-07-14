import argparse
import configparser
import os
import numpy as np

from autoencoder import Autoencoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

data_folder = config['EMBEDDING']['DATA']

parser = argparse.ArgumentParser()

#ToDo: add description and parameters for training
parser.add_argument('n_samples', type=int)
parser.add_argument('--offset', type=int, nargs='?', default=0)
parser.add_argument('--normalize', type=bool, nargs='?', default=True)
parser.add_argument('--valid_size', type=float, nargs='?', default=0.2)

args = parser.parse_args()

representation = '_'+str(args.n_samples)+'_'+str(args.offset)
if args.normalize:
    representation += '_normalized'

#repr_full_dataset = representation

if args.valid_size > 0:
    representation += '_' + str(args.valid_size)

X_train = np.load(str(data_folder)+'/GSE92743_X_train_genes' + str(representation)+'.npy')


if args.valid_size > 0:
    autoencoder = Autoencoder(X_train.shape[1], embedding_size=50, learning_rate=0.0001,
                              batch_norm=False, run_folder=None, save_model=True, offset=args.offset)
    X_valid = np.load(str(data_folder)+'/GSE92743_X_valid_genes' + str(representation)+'.npy')
    autoencoder.fit(X_train, batch_size=128, epochs=1, validation_data=(X_valid, X_valid))
else:
    autoencoder = Autoencoder(X_train.shape[1], embedding_size=50, learning_rate=0.0001,
                              batch_norm=False, run_folder=None, save_model=True, offset=args.offset, patience=0)
    X = np.load(str(data_folder) + '/GSE92743_X_train_genes' + str(representation)+'.npy')
    autoencoder.fit(X_train, batch_size=128, epochs=1)


