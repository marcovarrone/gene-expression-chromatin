import argparse
import os

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

# ToDo: add description and parameters for training
parser.add_argument('-d', '--data', type=str)
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('-se', '--save-embedding', default=False, action='store_true')
parser.add_argument('-sm', '--save-model', default=False, action='store_true')
parser.add_argument('-v', '--valid-size', type=float, nargs='?', default=0)
parser.add_argument('-s', '--random-seed', type=int, default=42)

parser.add_argument('--gpu', default=False, action='store_true')
parser.add_argument('--encode-sizes', nargs='*', type=int, default=None)
parser.add_argument('--decode-sizes', nargs='*', type=int, default=None)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--embedding-size', type=int, default=50)
parser.add_argument('--initializer', type=str, default='glorot_uniform')
parser.add_argument('--learning-rate', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--dropout-in', type=float, default=0.0)
parser.add_argument('--batch-norm', default=False, action='store_true')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--run-folder', type=str, default=None)

args = parser.parse_args()
if not args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from models.variational_autoencoder import VariationalAutoencoder

config_tf = tf.compat.v1.ConfigProto(device_count={'GPU': 0, 'CPU': 10})
config_tf.intra_op_parallelism_threads = 10
config_tf.inter_op_parallelism_threads = 10
tf.compat.v1.Session(config=config_tf)

np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)

X = np.load('data/' + str(args.dataset) + '/' + str(args.data_representation) + '.npy')

data_repr = args.data_representation.replace('X_train_', '').replace('X_valid_', '')

decode_sizes = args.decode_sizes
if args.encode_sizes and not args.decode_sizes:
    decode_sizes = args.encode_sizes[::-1]

if args.valid_size:
    X_train, X_valid = train_test_split(X, test_size=args.valid_size, shuffle=True)
    autoencoder = VariationalAutoencoder(X_train.shape[1],
                                         activation=args.activation,
                                         batch_size=args.batch_size,
                                         initializer=args.initializer,
                                         encoder_sizes=args.encode_sizes,
                                         decoder_sizes=decode_sizes,
                                         embedding_size=args.embedding_size,
                                         learning_rate=args.learning_rate,
                                         dropout=args.dropout,
                                         dropout_in=args.dropout_in,
                                         batch_norm=args.batch_norm,
                                         run_folder=args.run_folder,
                                         save_model=args.save_model,
                                         data_representation=data_repr)
    autoencoder.fit(X_train, epochs=args.epochs, validation_data=(X_valid, None))
    print(autoencoder.evaluate(X_valid, None, batch_size=32))
else:
    X_train = X
    autoencoder = VariationalAutoencoder(X_train.shape[1],
                                         activation=args.activation,
                                         batch_size=args.batch_size,
                                         initializer=args.initializer,
                                         encoder_sizes=args.encode_sizes,
                                         decoder_sizes=decode_sizes,
                                         embedding_size=args.embedding_size,
                                         learning_rate=args.learning_rate,
                                         dropout=args.dropout,
                                         dropout_in=args.dropout_in,
                                         batch_norm=args.batch_norm,
                                         run_folder=args.run_folder,
                                         save_model=args.save_model,
                                         patience=0,
                                         data_representation=data_repr)
    autoencoder.fit(X_train, epochs=args.epochs)


if args.save_embedding:
    embedding, _, _ = autoencoder.encoder.predict(X_train, batch_size=32)
    print('Saving embedding in embeddings/' + str(args.dataset) + '/' + str(autoencoder) + '.npy')
    np.save('embeddings/' + str(args.dataset) + '/' + str(autoencoder) + '.npy',
            embedding)
