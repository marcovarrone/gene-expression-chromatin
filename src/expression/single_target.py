import configparser

import numpy as np
import scipy.sparse as sps
import tensorflow as tf
from sklearn.model_selection import train_test_split

from mlp import MLP
from utils import join_source_genes, delete_csr

config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 5})
config.intra_op_parallelism_threads = 5
config.inter_op_parallelism_threads = 5
tf.Session(config=config)

config = configparser.ConfigParser()
config.read('/home/varrone/config.ini')

X_train = np.load(config['GSE']['X_TRAIN'])
y_train = np.load(config['GSE']['Y_TRAIN'])
X_test = np.load(config['GSE']['X_VAL'])
y_test = np.load(config['GSE']['Y_VAL'])

R = sps.load_npz(config['GRAPH']['GENEMANIA']).todense()

LANDMARK_REG = 0.0
TRAIN_SIZE = 0.1
VAL_SIZE = 10000

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=TRAIN_SIZE)
np.random.seed(42)


def exclude_target(R, target):
    if isinstance(R, sps.csr_matrix):
        R_excluded = delete_csr(R, 970 + target, axis=1)
        R_excluded = delete_csr(R_excluded, 970 + target, axis=0)
    else:
        R_excluded = np.delete(R, 970 + target, axis=1)
        R_excluded = np.delete(R_excluded, 970 + target, axis=0)
    return R_excluded


if __name__ == '__main__':
    n_targets = 50

    f = open("reg_results/mlp_log_" + str(n_targets) + ".txt", "w+")
    losses = []
    for i, target in enumerate(np.random.choice(range(y_train.shape[1]), n_targets)):
        print(str(i + 1) + '/' + str(n_targets), 'Target', target)

        R_target_excluded = None
        if LANDMARK_REG != 0:
            R_target_excluded = exclude_target(R, target)

        X_train_target, y_train_target = join_source_genes(X_train, y_train, 1, [target])
        X_valid_target, y_valid_target = join_source_genes(X_valid[:VAL_SIZE], y_valid[:VAL_SIZE], 1, [target])
        X_test_target, y_test_target = join_source_genes(X_test, y_test, 1, [target])

        mlp = MLP(X_train_target.shape[1], y_train_target.shape[1], n_hidden=1, hidden_size=100, learning_rate=0.0005,
                  landmark_reg=0.0001, landmark_graph=R_target_excluded, landmark_threshold=0.001, patience=10,
                  checkpoint_every=0)

        mlp.fit(X_train_target, y_train_target, batch_size=128, epochs=200,
                validation_data=(X_valid_target, y_valid_target))

        error = mlp.evaluate(X_test_target, y_test_target)

        print(error)
        f.write("%d: %.4f\n" % (target, error))
        losses.append(error)
        print(np.mean(losses))
    f.close()
    np.save('reg_results/mlp_losses_' + str(n_targets) + '.npy', np.array(losses))
    print(losses)
    print(np.mean(losses))
