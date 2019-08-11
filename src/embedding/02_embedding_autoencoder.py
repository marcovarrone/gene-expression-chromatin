import argparse
import os

import wandb
from keras import regularizers
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback

parser = argparse.ArgumentParser()

# ToDo: add description and parameters for training
parser.add_argument('-d', '--data', type=str)
parser.add_argument('--dataset', type=str, default='GSE92743')
parser.add_argument('-se', '--save-embedding', default=False, action='store_true')
parser.add_argument('-sm', '--save-model', default=False, action='store_true')
parser.add_argument('-v', '--valid-size', type=float, nargs='?', default=0)
parser.add_argument('-s', '--random-seed', type=int, default=42)
parser.add_argument('--n-jobs', type=int, default=10)

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
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=120)
parser.add_argument('--run-folder', type=str, default=None)
parser.add_argument('--l2-reg', type=float, default=0.0)
parser.add_argument('--patience', type=int, default=10)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.n_jobs)
os.environ["MKL_NUM_THREADS"] = str(args.n_jobs)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.n_jobs)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.n_jobs)

import numpy as np
import tensorflow as tf
from models.autoencoder import Autoencoder

wandb.init(project='gene-expression-chromatin')
wandb.log(vars(args))
wandb.log({'model': 'Autoencoder'})

np.random.seed(args.random_seed)
tf.set_random_seed(args.random_seed)

X = np.load('data/' + str(args.dataset) + '/' + str(args.data) + '.npy')
data_repr = args.data.replace('X_train_', '').replace('X_valid_', '')

decode_sizes = args.decode_sizes
if args.encode_sizes and not args.decode_sizes:
    decode_sizes = args.encode_sizes[::-1]

if args.valid_size:
    X_train, X_valid = train_test_split(X, test_size=args.valid_size, shuffle=True)
    autoencoder = Autoencoder(X_train.shape[1],
                              activation=args.activation,
                              initializer=args.initializer,
                              encoder_sizes=args.encode_sizes,
                              decoder_sizes=decode_sizes,
                              embedding_size=args.embedding_size,
                              learning_rate=args.learning_rate,
                              regularizer=regularizers.l2(args.l2_reg),
                              dropout=args.dropout,
                              dropout_in=args.dropout_in,
                              batch_norm=args.batch_norm,
                              run_folder=args.run_folder,
                              save_model=args.save_model,
                              dataset=args.dataset,
                              patience=args.patience,
                              data_representation=data_repr)
    autoencoder.fit(X_train, batch_size=args.batch_size, epochs=args.epochs, validation_data=(X_valid, X_valid),
                    callbacks=[WandbCallback(monitor=['mean_squared_error', 'val_mean_squared_error'])])
else:
    X_train = X
    autoencoder = Autoencoder(X_train.shape[1],
                              activation=args.activation,
                              initializer=args.initializer,
                              encoder_sizes=args.encode_sizes,
                              decoder_sizes=decode_sizes,
                              embedding_size=args.embedding_size,
                              learning_rate=args.learning_rate,
                              regularizer=regularizers.l2(args.l2_reg),
                              dropout=args.dropout,
                              dropout_in=args.dropout_in,
                              batch_norm=args.batch_norm,
                              run_folder=args.run_folder,
                              save_model=args.save_model,
                              patience=0,
                              dataset=args.dataset,
                              data_representation=data_repr)
    autoencoder.fit(X_train, batch_size=args.batch_size, epochs=args.epochs)

if args.save_embedding:
    embedding = autoencoder.encoder.predict(X_train, batch_size=128)
    print('Saving embedding in embeddings/' + str(args.dataset) + '/' + str(autoencoder) + '.npy')
    np.save('embeddings/' + str(args.dataset) + '/' + str(autoencoder) + '.npy',
            embedding)
