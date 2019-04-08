import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard

import matplotlib
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from skopt import BayesSearchCV

X_train = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_X_tr_float64.npy')
y_train = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_Y_tr_float64.npy')

X_val = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_X_va_float64.npy')
y_val = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_Y_va_float64.npy')

LEARNING_RATE = 0.0001
N_EPOCHS = 200
BATCH_SIZE = 128
N_NEURONS = 4000
REGULARIZATION = 0.00001

FILENAME = 'mlp_'+str(N_NEURONS)+'_'+str(LEARNING_RATE)+'_'+str(N_EPOCHS)+'_'+str(BATCH_SIZE)+'_'+str(REGULARIZATION)

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 5
config.inter_op_parallelism_threads = 5
tf.Session(config=config)


def build_model(neurons=1000, regularization=None):
    model = keras.Sequential()
    model.add(keras.layers.Dense(neurons, activation=tf.nn.relu, input_shape=(X_train.shape[1],),
                                 kernel_regularizer=regularization))
    model.add(keras.layers.Dense(y_train.shape[1]))

    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE)

    model.compile(loss=keras.losses.mean_absolute_error, optimizer=optimizer, metrics=['mean_absolute_error'])
    return model


def load_model(path):
    model = build_model(N_NEURONS)
    model.load_weights(path)
    return model


def train_model(neurons=1000, regularization=None):
    model = build_model(neurons, regularization)
    print('Running model ' + str(FILENAME))

    mc = keras.callbacks.ModelCheckpoint('weights/' + str(FILENAME) + '.h5',
                                         save_weights_only=True, period=5)

    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_data=(X_val, y_val),
                        callbacks=[TensorBoard(log_dir='runs/'+str(FILENAME)), mc])
    return model, history


def save_model(model, history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_pickle('history/'+str(FILENAME) + '.pkl')
    model.save_weights('weights/'+str(FILENAME) + '.h5')


def plot_errors(model, aggregation=np.mean):
    prediction = model.predict(X_val, batch_size=BATCH_SIZE)
    errors = np.abs(prediction - y_val)
    error_genes = aggregation(errors, axis=0)
    #print(error_genes)

    worst_genes = np.argsort(error_genes)[-100:]

    worst_errors = errors[:, worst_genes]
    #print(np.mean(worst_errors, axis=0))

    #for gene_errors in worst_errors.T:
    #    sns.distplot(gene_errors, bins=20)
    #    plt.show()
    error_samples = aggregation(errors, axis=1)
    #print(np.argsort(error_samples))
    sns.distplot(error_genes, bins=20)
    plt.show()


if __name__ == '__main__':
    #model, history = train_model(N_NEURONS, regularization=tf.keras.regularizers.l1(REGULARIZATION))
    model = load_model('mlp_4000_mse_relu.h5')
    plot_errors(model, np.mean)
