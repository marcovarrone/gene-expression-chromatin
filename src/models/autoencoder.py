import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from tensorflow.python.keras.callbacks import TensorBoard

from models.model import ModelNN

# ToDo: optimize GPU memory usage

class Autoencoder(ModelNN):

    def __init__(self, in_dim, embedding_size, encoder_sizes=None, decoder_sizes=None, learning_rate=0.001,
                 activation=tf.nn.relu, regularizer=None, loss=keras.metrics.mean_squared_error,
                 optimizer=keras.optimizers.Adam, batch_norm=False, dropout_in=0.0, dropout=0.0, max_norm=None,
                 initializer='glorot_uniform', patience=10, checkpoint_every=0, save_model=False, run_folder=None,
                 data_representation=''):

        self.in_dim = in_dim
        self.embedding_size = embedding_size
        if encoder_sizes is None:
            encoder_sizes = []
        self.encoder_sizes = encoder_sizes

        if decoder_sizes is None:
            decoder_sizes = []
        self.decoder_sizes = decoder_sizes

        self.learning_rate = learning_rate
        self.batch_size = None
        self.epochs = None
        self.activation = activation
        self.regularizer = regularizer
        self.loss = loss
        self.optimizer = optimizer
        self.batch_norm = batch_norm
        self.dropout_in = dropout_in
        self.dropout = dropout
        self.max_norm = max_norm
        self.initializer = initializer
        self.data_repr = data_representation

        self.encoder = None

        super().__init__(patience, checkpoint_every, save_model, run_folder)

    def _build_model(self):
        input_data = Input(shape=(self.in_dim,))
        x = input_data

        x = Dropout(self.dropout_in)(x)

        for i, encoder_size in enumerate(self.encoder_sizes):
            x = Dense(units=encoder_size, activation=self.activation, kernel_initializer=self.initializer,
                      activity_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)
            x = Dropout(self.dropout)(x)

        x = Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer,
                  kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)
        # x = Dropout(self.dropout)(x)

        self.encoder = Model(inputs=input_data, outputs=x)


        for decoder_size in self.decoder_sizes:
            x = Dense(units=decoder_size, activation=self.activation, kernel_initializer=self.initializer,
                      activity_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)
            x = Dropout(self.dropout)(x)

        x = Dense(units=self.in_dim)(x)
        model = Model(inputs=input_data, outputs=x)
        optimizer = self.optimizer(lr=self.learning_rate)
        model.compile(loss=self.loss, optimizer=optimizer, metrics=[self.loss])

        self.output_model = self.encoder
        return model

    def fit(self, x=None, batch_size=128, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, force_train=False, **kwargs):
        self.batch_size = batch_size
        self.epochs = epochs

        if not force_train and self._load_model():
            return

        print("Start training of model " + str(self))
        callbacks = self._add_callbacks(callbacks)

        self.model.fit(x, x, batch_size, epochs, verbose, callbacks, validation_split,
                       validation_data, shuffle, class_weight, sample_weight,
                       initial_epoch, steps_per_epoch, validation_steps, **kwargs)

        if self.save_model:
            self._save_model(validation_data)

    def __str__(self):
        name = 'autoencoder'
        for encoder in self.encoder_sizes:
            name += '_' + str(encoder)
        name += '_' + str(self.embedding_size)

        for decoder in self.decoder_sizes:
            name += '_' + str(decoder)
        name += '_e' + str(self.epochs) + \
                '_lr' + str(self.learning_rate) + \
                '_bs' + str(self.batch_size)
        if self.dropout_in:
            name += '_dri' + str(self.dropout_in)
        if self.dropout:
            name += '_dr' + str(self.dropout)
        if self.regularizer:
            name += '_l1_' + str(self.regularizer.l1)
            name += '_l2_' + str(self.regularizer.l2)
        if self.max_norm:
            name += '_max' + str(self.max_norm)
        if self.initializer != 'glorot_uniform':
            name += '_' + str(self.initializer)
        if self.batch_norm:
            name += '_bn'
        name += '_' + str(self.data_repr)
        return name
