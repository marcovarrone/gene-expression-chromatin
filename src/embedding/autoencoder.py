import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import max_norm
from keras.layers import InputLayer, Dense, BatchNormalization, Dropout
from tensorflow.python.keras.callbacks import TensorBoard


class Autoencoder(keras.Sequential):

    def __init__(self, in_dim, encoder_sizes, decoder_sizes, learning_rate=0.001, activation=tf.nn.relu,
                 loss=keras.metrics.mean_squared_error, optimizer=keras.optimizers.Adam, batch_norm=False,
                 dropout_in=None,
                 dropout=None, max_norm=None, patience=10, checkpoint_every=0, run_folder=None):
        super().__init__()
        self.in_dim = in_dim
        if encoder_sizes is None:
            encoder_sizes = []
        self.encoder_sizes = encoder_sizes

        if decoder_sizes is None:
            decoder_sizes = []
        self.decoder_sizes = decoder_sizes

        self.learning_rate = learning_rate
        self.batch_size = None
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.batch_norm = batch_norm
        self.dropout_in = dropout_in
        self.dropout = dropout
        self.max_norm = max_norm
        self.patience = patience
        self.checkpoint_every = checkpoint_every
        self.run_folder = run_folder

        self.history = None

        self._build_model()

    def _build_model(self):

        self.add(InputLayer(input_shape=(self.in_dim,)))
        if self.dropout_in:
            self.add(Dropout(self.dropout_in))

        for encoder_size in self.encoder_sizes:
            if self.batch_norm:
                self.add(BatchNormalization())

            self.add(Dense(units=encoder_size, activation=self.activation, kernel_constraint=max_norm(self.max_norm),
                           bias_constraint=max_norm(self.max_norm)))

            if self.dropout:
                self.add(Dropout(self.dropout))

        for decoder_size in self.decoder_sizes:
            self.add(Dense(units=decoder_size, activation=self.activation, kernel_constraint=max_norm(self.max_norm),
                           bias_constraint=max_norm(self.max_norm)))

            if self.dropout:
                self.add(Dropout(self.dropout))

        self.add(Dense(units=self.in_dim))

        optimizer = self.optimizer(lr=self.learning_rate)
        self.compile(loss=self.loss, optimizer=optimizer, metrics=[self.loss])

    def fit(self, x=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, **kwargs):

        self.batch_size = batch_size

        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks, list):
            callbacks = [callbacks]

        if self.patience > 0:
            es = EarlyStopping(monitor='val_loss', verbose=1, patience=self.patience, restore_best_weights=True)
            callbacks.append(es)

        if self.checkpoint_every > 0:
            mc = ModelCheckpoint('weights/' + str(self) + '.h5',
                                 save_weights_only=True, period=self.checkpoint_every)
            callbacks.append(mc)

        if self.run_folder:
            tb = TensorBoard(log_dir=str(self.run_folder) + '/' + str(self))
            callbacks.append(tb)

        self.history = super().fit(x, x, batch_size, epochs, verbose, callbacks, validation_split,
                                   validation_data, shuffle, class_weight, sample_weight,
                                   initial_epoch, steps_per_epoch, validation_steps, **kwargs)
        return self.history

    def __str__(self):
        name = 'autoencoder'
        for encoder in self.encoder_sizes:
            name += '_' + str(encoder)

        for decoder in self.decoder_sizes:
            name += '_' + str(decoder)

        name += '_lr' + str(self.learning_rate) + \
                '_bs' + str(self.batch_size)
        if self.max_norm:
            name += '_max'+str(self.max_norm)
        if self.batch_norm:
            name += '_bn'
        return name
