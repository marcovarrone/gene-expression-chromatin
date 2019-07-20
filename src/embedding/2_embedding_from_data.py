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

# ToDo: add description and parameters for training
parser.add_argument('n_samples', type=int)
parser.add_argument('--offset', type=int, nargs='?', default=0)
parser.add_argument('--no-normalize', default=False, action='store_true')
parser.add_argument('--valid-size', type=float, nargs='?', default=0)
parser.add_argument('--save-embedding', default=False, action='store_true')

parser.add_argument('--embedding-size', type=int, default=50)
parser.add_argument('--learning-rate', type=float, default=0.0001)
parser.add_argument('--batch-norm', default=False, action='store_true')
parser.add_argument('--save-model', default=False, action='store_true')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=120)

args = parser.parse_args()

representation = '_' + str(args.n_samples) + '_' + str(args.offset)
if not args.no_normalize:
    representation += '_normalized'

if args.valid_size > 0:
    representation += '_' + str(args.valid_size)

X_train = np.load(str(data_folder) + '/GSE92743_X_train_genes' + str(representation) + '.npy')

if args.valid_size > 0:
    autoencoder = Autoencoder(X_train.shape[1], embedding_size=args.embedding_size, learning_rate=args.learning_rate,
                              batch_norm=args.batch_norm, run_folder=None, save_model=args.save_model, offset=args.offset)
    X_valid = np.load(str(data_folder) + '/GSE92743_X_valid_genes' + str(representation) + '.npy')
    autoencoder.fit(X_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_valid, X_valid))
else:
    autoencoder = Autoencoder(X_train.shape[1], embedding_size=args.embedding_size, learning_rate=args.learning_rate,
                              batch_norm=args.batch_norm, run_folder=None, save_model=args.save_model, offset=args.offset, patience=0)
    X = np.load(str(data_folder) + '/GSE92743_X_train_genes' + str(representation) + '.npy')
    autoencoder.fit(X_train, batch_size=args.batch_size, epochs=args.epochs)

if args.save_embedding:
    embedding = autoencoder.encoder.predict(X_train, batch_size=128)
    print('Saving embedding GSE92743_' + str(autoencoder)+'.npy')
    np.save('embeddings_from_data/GSE92743_' + str(autoencoder) + '.npy', embedding)
