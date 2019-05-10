from functools import partial

import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense

from utils import get_H, to_sparse_tensor


class MLP(keras.Sequential):
    def __init__(self, in_dim, out_dim, n_hidden, hidden_size, learning_rate=0.001, activation=tf.nn.relu,
                 loss=keras.metrics.mean_absolute_error, landmark_reg=None, landmark_graph=None,
                 landmark_threshold=None, target_reg=None, target_graph=None, target_threshold=None, patience=5,
                 checkpoint_every=0):
        super(MLP, self).__init__(name='mlp')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.loss = loss

        self.target_reg = target_reg
        self.patience = patience
        self.checkpoint_every = checkpoint_every
        self.history = None

        self.name = self._set_name(landmark_threshold, target_threshold)

        self.landmark_reg = None
        if landmark_reg is not None and landmark_graph is not None and landmark_threshold is not None:
            H = to_sparse_tensor(get_H(landmark_graph, landmark_threshold))
            self.landmark_reg = partial(self._landmark_regularization, H, landmark_reg)

        self.target_reg = None
        if target_reg is not None and target_graph is not None and target_threshold is not None:
            H = to_sparse_tensor(get_H(target_graph, target_threshold))
            self.target_reg = partial(self._target_regularization, H, target_reg)

        self.net = self._build_model()

    def _build_model(self):
        model = keras.Sequential()
        model.add(Dense(self.hidden_size, activation=self.activation, input_shape=(self.in_dim,),
                        kernel_regularizer=self.landmark_reg))
        for _ in range(self.n_hidden - 1):
            model.add(Dense(self.hidden_size, activation=self.activation))

        model.add(Dense(self.out_dim, kernel_regularizer=self.target_reg))

        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=[self.loss])
        return model

    def call(self, inputs, mask=None):
        return self.net(inputs)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):

        if not isinstance(callbacks, list):
            callbacks = [callbacks]

        if self.patience > 0:
            es = EarlyStopping(monitor='val_loss', verbose=1, patience=self.patience, restore_best_weights=True)
            callbacks.append(es)

        if self.checkpoint_every > 0:
            mc = ModelCheckpoint('weights/' + str(self.name) + '.h5',
                                 save_weights_only=True, period=self.checkpoint_every)
            callbacks.append(mc)

        self.history = self.net.fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data,
                                    shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch,
                                    validation_steps, **kwargs)

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None):
        loss, error = self.net.evaluate(x, y, batch_size, verbose, sample_weight, steps)
        return error

    def _landmark_regularization(self, H, regularization, weight_matrix):
        return regularization * K.sum(K.abs(tf.sparse.sparse_dense_matmul(H, weight_matrix)))

    def _target_regularization(self, H, regularization, weight_matrix):
        return regularization * K.sum(K.abs(tf.sparse.sparse_dense_matmul(weight_matrix, H)))

    def _set_name(self, lm_threshold, tgt_threshold):
        return 'mlp_l' + str(self.n_hidden) + \
               '_n' + str(self.hidden_size) + \
               '_bs' + str(self.batch_size) + \
               '_lr' + str(self.learning_rate) + \
               '_lreg' + str(self.landmark_reg) + \
               '_lth' + str(lm_threshold) + \
               '_treg' + str(self.target_reg) + \
               '_tth' + str(tgt_threshold)

    def __repr__(self):
        return super(MLP, self).__repr__()


if __name__ == '__main__':
    mlp = MLP(10, 1000, 2, 200)
    mlp.compile(optimizer=keras.optimizers.Adam(), )
    print(mlp.summary())
