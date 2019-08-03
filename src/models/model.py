import os

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


class ModelNN(object):
    def __init__(self, patience=10, checkpoint_every=0, save_model=False, run_folder=None):

        self.output_model = None
        self.patience = patience
        self.checkpoint_every = checkpoint_every
        self.save_model = save_model
        self.run_folder = run_folder
        self.history = None
        print("before build model")
        self.model = self._build_model()

    def _build_model(self):

        return NotImplementedError

    def _add_callbacks(self, callbacks):
        if callbacks is None:
            callbacks = []
        if not isinstance(callbacks, list):
            callbacks = [callbacks]

        if self.patience > 0:
            es = EarlyStopping(monitor='val_loss', verbose=1, patience=self.patience, restore_best_weights=True)
            callbacks.append(es)

        if self.checkpoint_every > 0:
            mc = ModelCheckpoint('weights/' + str(self) + '.h5',
                                 save_weights_only=False, period=self.checkpoint_every)
            callbacks.append(mc)

        if self.run_folder:
            tb = TensorBoard(log_dir=str(self.run_folder) + '/' + str(self))
            callbacks.append(tb)
        return callbacks

    def _load_model(self):
        if os.path.isfile('models/' + str(self) + '_weights.h5'):
            print("The model " + str(self) + " has already been trained. Loading from file")
            self.output_model.load_weights('models/' + str(self) + '_weights.h5')
            return True
        return False

    def _save_model(self, validation_data=None):
        if validation_data:
            print("Warning: saving a model which has not been trained on all the genes.")

        if os.path.isfile('models/' + str(self) + '_weights.h5'):
            os.remove('models/' + str(self) + '_weights.h5')
        self.output_model.save_weights('models/' + str(self) + '_weights.h5')

    def evaluate(self, x=None, y=None, batch_size=64, verbose=1, sample_weight=None, steps=None):
        return self.model.evaluate(x, y, batch_size, verbose, sample_weight, steps)

    def predict(self, x, batch_size=64, verbose=1):
        return self.model.predict(x, batch_size=batch_size, verbose=verbose)

    def __str__(self):
        return NotImplementedError
