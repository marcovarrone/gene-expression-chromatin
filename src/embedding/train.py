import configparser

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from autoencoder import Autoencoder

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 5})
config.intra_op_parallelism_threads = 5
config.inter_op_parallelism_threads = 5
tf.Session(config=config)

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

X = np.load(config['GSE_UNSCALED']['X_TRAIN'])
y = np.load(config['GSE_UNSCALED']['Y_TRAIN'])
run_folder = config['GSE']['RUNS']

TEST_SIZE = 0.2
N_SAMPLES = 20000

np.random.seed(42)

X = np.hstack((X, y))
np.random.shuffle(X)
X = X[:N_SAMPLES]
X = X.T

scaler = StandardScaler()
X = scaler.fit_transform(X)

np.random.seed(42)
X_train, X_valid = train_test_split(X, test_size=TEST_SIZE)

print(run_folder)

autoencoder = Autoencoder(X_train.shape[1], encoder_sizes=[50], decoder_sizes=[], learning_rate=0.0005,
                          batch_norm=False, run_folder=run_folder)
autoencoder.fit(X_train, batch_size=128, epochs=200, validation_data=(X_valid, X_valid))
