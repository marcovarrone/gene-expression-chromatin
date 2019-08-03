import argparse
import os

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

# ToDo: add description and parameters for training
parser.add_argument('-d', '--data-representation', type=str)
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('-se', '--save-embedding', default=False, action='store_true')
parser.add_argument('-s', '--random-seed', type=int, default=42)

parser.add_argument('--gpu', default=False, action='store_true')
parser.add_argument('--embedding-size', type=int, default=50)
parser.add_argument('--learning-rate', type=float, default=0.0001)
parser.add_argument('--no-batch-norm', default=False, action='store_true')
parser.add_argument('-sm', '--save-model', default=False, action='store_true')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('-v', '--valid-size', type=float, nargs='?', default=0)
parser.add_argument('--run-folder', type=str, default=None)

args = parser.parse_args()
if not args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from autoencoder import Autoencoder

config_tf = tf.compat.v1.ConfigProto(device_count={'GPU': 0, 'CPU': 10})
config_tf.intra_op_parallelism_threads = 10
config_tf.inter_op_parallelism_threads = 10
tf.compat.v1.Session(config=config_tf)

np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)

X = np.load('data/' + str(args.dataset) + '/' + str(args.data_representation) + '.npy')

data_repr = args.data_representation.replace('X_train_', '').replace('X_valid_', '')

if args.valid_size:
    X_train, X_valid = train_test_split(X, test_size=args.valid_size, shuffle=True)
    autoencoder = Autoencoder(X_train.shape[1], embedding_size=args.embedding_size,
                              learning_rate=args.learning_rate,
                              batch_norm=(not args.no_batch_norm), run_folder=args.run_folder,
                              save_model=args.save_model, data_representation=data_repr)
    autoencoder.fit(X_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_valid, X_valid))
else:
    X_train = X
    autoencoder = Autoencoder(X_train.shape[1], embedding_size=args.embedding_size,
                              learning_rate=args.learning_rate,
                              batch_norm=(not args.no_batch_norm), run_folder=args.run_folder,
                              save_model=args.save_model, patience=0, data_representation=data_repr)
    autoencoder.fit(X_train, batch_size=args.batch_size, epochs=args.epochs)

if args.save_embedding:
    embedding = autoencoder.encoder.predict(X_train, batch_size=128)
    print('Saving embedding in embeddings/' + str(args.dataset) + '/' + str(autoencoder) + '.npy')
    np.save('embeddings/' + str(args.dataset) + '/' + str(autoencoder) + '.npy',
            embedding)
