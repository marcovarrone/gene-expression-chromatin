import configparser
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config_tf = tf.compat.v1.ConfigProto(device_count={'GPU': 0, 'CPU': 10})
config_tf.intra_op_parallelism_threads = 10
config_tf.inter_op_parallelism_threads = 10
tf.compat.v1.Session(config=config_tf)

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

X = np.load(config['GSE_UNSCALED']['X_TRAIN'])
y = np.load(config['GSE_UNSCALED']['Y_TRAIN'])
run_folder=config['GSE']['RUNS']

np.random.seed(42)

TEST_SIZE = 0.2
N_SAMPLES = 20000
OFFSET = 0

X = np.hstack((X, y))
#np.random.shuffle(X)
X = X[OFFSET:OFFSET + N_SAMPLES]
X = X.T

scaler = StandardScaler()
X = scaler.fit_transform(X)

np.random.seed(42)
X_train, X_valid = train_test_split(X, test_size=TEST_SIZE)