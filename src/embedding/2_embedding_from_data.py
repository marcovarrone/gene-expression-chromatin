import argparse
import configparser
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
parser = argparse.ArgumentParser()

# ToDo: add description and parameters for training
parser.add_argument('-data-repr', '--data-representation', type=str, required=True)
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('--save-embedding', default=False, action='store_true')

parser.add_argument('--gpu', default=False, action='store_true')

parser.add_argument('--embedding-size', type=int, default=50)
parser.add_argument('--learning-rate', type=float, default=0.0001)
parser.add_argument('--no-batch-norm', default=False, action='store_true')
parser.add_argument('--save-model', default=False, action='store_true')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--validate', default=False, action='store_true')

args = parser.parse_args()
if not args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

from autoencoder import Autoencoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

data_folder = config['EMBEDDING']['DATA']

print(args)

X_train = np.load(
    str(data_folder) + '/' + str(args.dataset) + '/X_train_genes_' + str(args.data_representation) + '.npy')

if args.validate:
    autoencoder = Autoencoder(X_train.shape[1], embedding_size=args.embedding_size,
                              learning_rate=args.learning_rate,
                              batch_norm=(not args.no_batch_norm), run_folder=None, save_model=args.save_model)
    X_valid = np.load(
        str(data_folder) + '/' + str(args.dataset) + '/X_valid_genes' + str(args.data_representation) + '.npy')
    autoencoder.fit(X_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_valid, X_valid))
else:
    autoencoder = Autoencoder(X_train.shape[1], embedding_size=args.embedding_size,
                              learning_rate=args.learning_rate,
                              batch_norm=(not args.no_batch_norm), run_folder=None, save_model=args.save_model,
                              patience=0)
    autoencoder.fit(X_train, batch_size=args.batch_size, epochs=args.epochs)

if args.save_embedding:
    embedding = autoencoder.encoder.predict(X_train, batch_size=128)
    print('Saving embedding in embeddings/' + str(args.dataset) + '/' + str(autoencoder) + '.npy')
    np.save('embeddings/' + str(args.dataset) + '/' + str(autoencoder) + '.npy', embedding)
