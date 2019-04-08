import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard

LEARNING_RATE = 0.001
N_EPOCHS = 150
BATCH_SIZE = 256
ENCODING = 64
ACTIVATION = 'relu'
REGULARIZATION = 0
BATCH_NORM = False
INITIALIZER = 'he_normal'

FILENAME = 'encoder_decoder_' + str(LEARNING_RATE).split('.')[1] + '_' + str(N_EPOCHS) + '_' + str(BATCH_SIZE) + '_' + \
           str(REGULARIZATION) + '_' + str(INITIALIZER)
FILENAME += '_bn' if BATCH_NORM else ''

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 5
config.inter_op_parallelism_threads = 5
tf.Session(config=config)

X_train = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_X_tr_float64.npy')
y_train = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_Y_tr_float64.npy')

X_val = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_X_va_float64.npy')
y_val = np.load('/home/nanni/Projects/gexi-top/data/processed/d-gex/bgedv2_Y_va_float64.npy')


def build_model(encoding_size=32, learning_rate=0.001, activation='relu', regularizer=None, batch_norm=False,
                initializer=None):
    input_expression = Input(shape=(X_train.shape[1],))
    if batch_norm:
        encoder = BatchNormalization()(input_expression)
        encoder = Dense(8 * encoding_size, activation=activation, kernel_initializer=initializer)(encoder)
    else:
        encoder = Dense(8 * encoding_size, activation=activation, kernel_initializer=initializer)(input_expression)
    encoder = Dense(4 * encoding_size, activation=activation, kernel_initializer=initializer)(encoder)
    encoder = Dense(2 * encoding_size, activation=activation, kernel_initializer=initializer)(encoder)
    encoder = Dense(encoding_size, activation=activation, kernel_initializer=initializer)(encoder)
    decoder = Dense(2 * encoding_size, activation=activation, kernel_regularizer=regularizer,
                    kernel_initializer=initializer)(encoder)
    decoder = Dense(4 * encoding_size, activation=activation, kernel_regularizer=regularizer,
                    kernel_initializer=initializer)(decoder)
    decoder = Dense(8 * encoding_size, activation=activation, kernel_regularizer=regularizer,
                    kernel_initializer=initializer)(decoder)
    decoder = Dense(y_train.shape[1])(decoder)

    encoder_decoder = Model(input_expression, decoder)

    optimizer = keras.optimizers.Adam(lr=learning_rate)

    encoder_decoder.compile(loss=keras.losses.mean_absolute_error, optimizer=optimizer, metrics=['mean_absolute_error'])
    return encoder_decoder


def save_model(model, history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_pickle('history/' + str(filename) + '.pkl')
    model.save_weights('weights/' + str(filename) + '.h5')


if __name__ == '__main__':
    model = build_model(ENCODING, LEARNING_RATE, ACTIVATION, regularizer=keras.regularizers.l1(REGULARIZATION),
                        batch_norm=BATCH_NORM, initializer=INITIALIZER)
    print('Running model ' + str(FILENAME))
    mc = keras.callbacks.ModelCheckpoint('weights/' + str(FILENAME) + '.h5',
                                         save_weights_only=True, period=5)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, verbose=1,
                        validation_data=(X_val, y_val), shuffle=True,
                        callbacks=[TensorBoard(log_dir='runs/' + str(FILENAME)), mc])
    save_model(model, history, FILENAME)
