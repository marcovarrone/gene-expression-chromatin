import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Input, Dropout, Lambda, BatchNormalization
from keras.models import Model

from models.model import ModelNN


# ToDo: optimize GPU memory usage

class VariationalAutoencoder(ModelNN):

    def __init__(self, in_dim, embedding_size, encoder_sizes=None, decoder_sizes=None, learning_rate=0.001,
                 activation=tf.nn.relu, batch_size=64, regularizer=None, loss=keras.metrics.mean_squared_error,
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
        self.batch_size = batch_size
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
            if encoder_size:
                x = Dense(units=encoder_size, activation=self.activation, kernel_initializer=self.initializer,
                          activity_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)
                if self.batch_norm:
                    x = BatchNormalization()(x)
                # x = Dropout(self.dropout)(x)

        mu = Dense(self.embedding_size)(x)
        log_var = Dense(self.embedding_size)(x)
        # x = Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer,
        #          kernel_regularizer=self.regularizer, bias_regularizer=self.regularizer)(x)
        # x = Dropout(self.dropout)(x)
        z = Lambda(self._sampling, output_shape=(self.embedding_size,))([mu, log_var])

        self.encoder = Model(input_data, [mu, log_var, z], name='encoder')

        for decoder_size in self.decoder_sizes:
            if decoder_size:
                z = Dense(units=decoder_size, activation=self.activation, kernel_initializer=self.initializer,
                          activity_regularizer=self.regularizer, bias_regularizer=self.regularizer)(z)
                # x = Dropout(self.dropout)(x)

        z = Dense(units=self.in_dim)(z)


        model = Model(inputs=input_data, outputs=z)
        model.add_loss(self._vae_loss(input_data, z, mu, log_var))
        optimizer = self.optimizer(lr=self.learning_rate)
        print(self._vae_loss(input_data, z, mu, log_var))
        model.compile(optimizer=optimizer)
        model.outputs[0]._uses_learning_phase = True

        self.output_model = self.encoder
        return model

    def fit(self, x=None, epochs=10, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, force_train=False, **kwargs):

        self.epochs = epochs

        if not force_train and self._load_model():
            return

        print("Start training of model " + str(self))
        callbacks = self._add_callbacks(callbacks)
        print(self.batch_size)
        self.model.fit(x, batch_size=self.batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
                       validation_split=validation_split,
                       validation_data=validation_data, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight,
                       initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, **kwargs)
        #self.model.fit(x, shuffle=True,
        #               epochs=self.epochs,
        #               batch_size=self.batch_size,
        #               validation_data=(x, None), verbose=1)

        if self.save_model:
            self._save_model(validation_data)

    def _vae_loss(self, inputs, outputs, mu, log_var):
        reconstruction_loss = self.loss(inputs, outputs) * self.in_dim
        kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis=-1)
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return K.in_train_phase(vae_loss, K.mean(self.loss(inputs, outputs)))
    def _sampling(self, args):
        mu, log_var = args
        eps = K.random_normal(shape=(self.batch_size, self.embedding_size), mean=0., stddev=1.0)
        return mu + K.exp(log_var) * eps

    def __str__(self):
        name = 'variational_ae'
        for encoder in self.encoder_sizes:
            name += '_' + str(encoder)
        name += '_' + str(self.embedding_size)

        for decoder in self.decoder_sizes:
            name += '_' + str(decoder)
        name += '_e' + str(self.epochs) + \
                '_lr' + str(self.learning_rate) + \
                '_bs' + str(self.batch_size)
        if self.activation != 'relu':
            name += '_a' + str(self.activation)
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
