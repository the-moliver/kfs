from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import warnings
import pprint
range2 = range
from six.moves import range
import six
import time
import threading
try:
    import queue
except ImportError:
    import Queue as queue

from keras import backend as K
from keras import optimizers
from keras import objectives
from keras import callbacks as cbks
from keras.utils.layer_utils import model_summary
from keras.utils.generic_utils import Progbar
from keras.layers import containers
from keras.models import standardize_y, weighted_objective, standardize_weights, batch_shuffle, make_batches, standardize_X, slice_X, get_function_name


def slice_X_delay(X, start=None):
    if type(X) == list:
        tlist = range2(len(X[0].shape) + 1)
        tlist[:2] = [1, 0]
        if len(X) > 1:
            return [x[start, :].transpose(tlist) for x in X[:-2]] + [X[-2][start[0], :]] + [X[-1][start[0], None][:, 0]]
        else:
            return [x[start, :].transpose(tlist) for x in X]
    else:
        tlist = range2(len(X.shape) + 1)
        tlist[:2] = [1, 0]
        return X[start, :].transpose(tlist)


class Model(object):
    '''Abstract base model class.
    '''
    def _fit(self, f, ins, out_labels=[], batch_size=128,
             nb_epoch=100, verbose=1, callbacks=[],
             val_f=None, val_ins=None, shuffle=True, metrics=[]):
        '''
            Abstract fit function for f(ins).
            Assume that f returns a list, labelled by out_labels.
        '''
        self.training_data = ins
        self.validation_data = val_ins
        do_validation = False
        if val_f and val_ins:
            do_validation = True
            if verbose:
                print('Train on %d samples, validate on %d samples' %
                      (len(ins[0]), len(val_ins[0])))

        nb_train_sample = len(ins[0])
        index_array = np.arange(nb_train_sample)

        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + callbacks + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)

        callbacks._set_model(self)
        callbacks._set_params({
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': metrics,
        })
        callbacks.on_train_begin()

        self.stop_training = False
        for epoch in range(nb_epoch):
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    batch_ids = [np.maximum(0, batch_ids-d) for d in range(0, self.delays + 1)]
                    ins_batch = slice_X_delay(ins, batch_ids)
                except TypeError:
                    print(ins)
                    raise Exception('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids[0])
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(ins_batch)
                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                epoch_logs = {}
                if batch_index == len(batches) - 1:  # last batch
                    # validation
                    if do_validation:
                        # replace with self._evaluate
                        val_outs = self._test_loop(val_f, val_ins,
                                                   batch_size=batch_size,
                                                   verbose=0)
                        if type(val_outs) != list:
                            val_outs = [val_outs]
                        # same labels assumed
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return self.history

    def _predict_loop(self, f, ins, batch_size=128, verbose=0):
        '''Abstract method to loop over some data in batches.
        '''
        nb_sample = len(ins[0])
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            batch_ids = [np.maximum(0, batch_ids-d) for d in range(0, self.delays + 1)]
            ins_batch = slice_X_delay(ins, batch_ids)

            batch_outs = f(ins_batch)
            if type(batch_outs) != list:
                batch_outs = [batch_outs]
            if batch_index == 0:
                for batch_out in batch_outs:
                    shape = (nb_sample,) + batch_out.shape[1:]
                    outs.append(np.zeros(shape))

            for i, batch_out in enumerate(batch_outs):
                outs[i][batch_start:batch_end] = batch_out
            if verbose == 1:
                progbar.update(batch_end)
        return outs

    def _test_loop(self, f, ins, batch_size=128, verbose=0):
        '''Abstract method to loop over some data in batches.
        '''
        nb_sample = len(ins[0])
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            batch_ids = [np.maximum(0, batch_ids-d) for d in range(0, self.delays + 1)]
            ins_batch = slice_X_delay(ins, batch_ids)

            batch_outs = f(ins_batch)
            if type(batch_outs) == list:
                if batch_index == 0:
                    for batch_out in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    outs[i] += batch_out * len(batch_ids[0])
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += batch_outs * len(batch_ids[0])

            if verbose == 1:
                progbar.update(batch_end)
        for i, out in enumerate(outs):
            outs[i] /= nb_sample
        return outs

    def get_config(self, verbose=0):
        '''Return the configuration of the model
        as a dictionary.

        To load a model from its configuration, use
        `keras.models.model_from_config(config, custom_objects={})`.
        '''
        config = super(Model, self).get_config()
        for p in ['sample_weight_mode', 'sample_weight_modes', 'loss_weights']:
            if hasattr(self, p):
                config[p] = getattr(self, p)
        if hasattr(self, 'optimizer'):
            config['optimizer'] = self.optimizer.get_config()
        if hasattr(self, 'loss'):
            if type(self.loss) == dict:
                config['loss'] = dict([(k, get_function_name(v)) for k, v in self.loss.items()])
            else:
                config['loss'] = get_function_name(self.loss)
        if verbose:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(config)
        return config

    def to_yaml(self, **kwargs):
        '''Return a yaml string containing the model configuration.

        To load a model from a yaml save file, use
        `keras.models.from_yaml(yaml_string, custom_objects={})`.

        `custom_objects` should be a dictionary mapping
        the names of custom losses / layers / etc to the corresponding
        functions / classes.
        '''
        import yaml
        config = self.get_config()
        return yaml.dump(config, **kwargs)

    def to_json(self, **kwargs):
        '''Return a JSON string containing the model configuration.

        To load a model from a JSON save file, use
        `keras.models.from_json(json_string, custom_objects={})`.
        '''
        import json

        def get_json_type(obj):
            # if obj is any numpy type
            if type(obj).__module__ == np.__name__:
                return obj.item()

            # if obj is a python 'type'
            if type(obj).__name__ == type.__name__:
                return obj.__name__

            raise TypeError('Not JSON Serializable')

        config = self.get_config()
        return json.dumps(config, default=get_json_type, **kwargs)

    def summary(self):
        '''Print out a summary of the model architecture,
        include parameter count information.
        '''
        model_summary(self)


class Sequential(Model, containers.Sequential):
    '''Linear stack of layers.

    Inherits from containers.Sequential.
    '''

    def __init__(self, delays, **kwargs):
        super(Sequential, self).__init__(**kwargs)
        self.delays = delays

    def compile(self, optimizer, loss,
                class_mode=None,
                sample_weight_mode=None,
                **kwargs):
        '''Configure the learning process.

        # Arguments
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](optimizers.md).
            loss: str (name of objective function) or objective function.
                See [objectives](objectives.md).
            class_mode: deprecated argument,
                it is set automatically starting with Keras 0.3.3.
            sample_weight_mode: if you need to do timestep-wise
                sample weighting (2D weights), set this to "temporal".
                "None" defaults to sample-wise weights (1D).
            kwargs: for Theano backend, these are passed into K.function.
                Ignored for Tensorflow backend.
        '''
        if class_mode is not None:
            warnings.warn('The "class_mode" argument is deprecated, please remove it from your code.')

        self.optimizer = optimizers.get(optimizer)
        self.sample_weight_mode = sample_weight_mode

        self.loss = objectives.get(loss)
        weighted_loss = weighted_objective(self.loss)

        # input of model
        self.X_train = self.get_input(train=True)
        self.X_test = self.get_input(train=False)

        self.y_train = self.get_output(train=True)
        self.y_test = self.get_output(train=False)

        # target of model
        self.y = K.placeholder(ndim=K.ndim(self.y_train))

        if self.sample_weight_mode == 'temporal':
            self.weights = K.placeholder(ndim=2)
        else:
            self.weights = K.placeholder(ndim=1)

        if hasattr(self.layers[-1], 'get_output_mask'):
            mask = self.layers[-1].get_output_mask()
        else:
            mask = None
        train_loss = weighted_loss(self.y, self.y_train, self.weights, mask)
        test_loss = weighted_loss(self.y, self.y_test, self.weights, mask)

        # set class_mode, for accuracy computation:
        if self.output_shape[-1] == 1:
            class_mode = 'binary'
        else:
            class_mode = 'categorical'
        self.class_mode = class_mode

        if class_mode == 'categorical':
            train_accuracy = K.mean(K.equal(K.argmax(self.y, axis=-1),
                                            K.argmax(self.y_train, axis=-1)))
            test_accuracy = K.mean(K.equal(K.argmax(self.y, axis=-1),
                                           K.argmax(self.y_test, axis=-1)))
        elif class_mode == 'binary':
            if self.loss.__name__ == 'categorical_crossentropy':
                warnings.warn('Your model output has shape ' + str(self.output_shape) +
                              ' (1-dimensional features), but you are using ' +
                              ' the `categorical_crossentropy` loss. You ' +
                              'almost certainly want to use `binary_crossentropy` instead.')
            train_accuracy = K.mean(K.equal(self.y, K.round(self.y_train)))
            test_accuracy = K.mean(K.equal(self.y, K.round(self.y_test)))

        for r in self.regularizers:
            train_loss = r(train_loss)
        updates = self.optimizer.get_updates(self.trainable_weights,
                                             self.constraints,
                                             train_loss)
        updates += self.updates

        if type(self.X_train) == list:
            train_ins = self.X_train + [self.y, self.weights]
            test_ins = self.X_test + [self.y, self.weights]
            assert type(self.X_test) == list
            predict_ins = self.X_test
        else:
            train_ins = [self.X_train, self.y, self.weights]
            test_ins = [self.X_test, self.y, self.weights]
            predict_ins = [self.X_test]

        self._train = K.function(train_ins, [train_loss],
                                 updates=updates, **kwargs)
        self._train_with_acc = K.function(train_ins,
                                          [train_loss, train_accuracy],
                                          updates=updates, **kwargs)
        self._predict = K.function(predict_ins, [self.y_test],
                                   updates=self.state_updates, **kwargs)
        self._test = K.function(test_ins, [test_loss],
                                updates=self.state_updates, **kwargs)
        self._test_with_acc = K.function(test_ins,
                                         [test_loss, test_accuracy],
                                         updates=self.state_updates, **kwargs)

    def fit(self, X, y, batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            show_accuracy=False, class_weight=None, sample_weight=None):
        '''Train the model for a fixed number of epochs.

        Returns a history object. Its `history` attribute is a record of
        training loss values at successive epochs,
        as well as validation loss values (if applicable).

        # Arguments
            X: data, as a numpy array.
            y: labels, as a numpy array.
            batch_size: int. Number of samples per gradient update.
            nb_epoch: int.
            verbose: 0 for no logging to stdout,
                1 for progress bar logging, 2 for one log line per epoch.
            callbacks: `keras.callbacks.Callback` list.
                List of callbacks to apply during training.
                See [callbacks](callbacks.md).
            validation_split: float (0. < x < 1).
                Fraction of the data to use as held-out validation data.
            validation_data: tuple (X, y) to be used as held-out
                validation data. Will override validation_split.
            shuffle: boolean or str (for 'batch').
                Whether to shuffle the samples at each epoch.
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
            show_accuracy: boolean. Whether to display
                class accuracy in the logs to stdout at each epoch.
            class_weight: dictionary mapping classes to a weight value,
                used for scaling the loss function (during training only).
            sample_weight: list or numpy array of weights for
                the training samples, used for scaling the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                sample_weight_mode="temporal" in compile().
        '''
        if type(X) == list:
            if len(set([len(a) for a in X] + [len(y)])) != 1:
                raise Exception('All input arrays and the target array must '
                                'have the same number of samples.')
        else:
            if len(X) != len(y):
                raise Exception('The input data tensor (X) and '
                                'the target tensor (y) must have '
                                'the same number of samples. Found: '
                                'len(X) = {}, len(y) = {}'.format(len(X), len(y)))
        if sample_weight is not None:
            assert len(sample_weight) == len(y), ('"sample_weight" must have '
                                                  'the same number of samples '
                                                  'as X and y.')
        X = standardize_X(X)
        y = standardize_y(y)

        val_f = None
        val_ins = None
        if validation_data or validation_split:
            if show_accuracy:
                val_f = self._test_with_acc
            else:
                val_f = self._test
        if validation_data:
            if len(validation_data) == 2:
                X_val, y_val = validation_data
                if type(X_val) == list:
                    assert len(set([len(a) for a in X_val] + [len(y_val)])) == 1
                else:
                    assert len(X_val) == len(y_val)
                X_val = standardize_X(X_val)
                y_val = standardize_y(y_val)
                sample_weight_val = standardize_weights(y_val)
            elif len(validation_data) == 3:
                X_val, y_val, sample_weight_val = validation_data
                if type(X_val) == list:
                    assert len(set([len(a) for a in X_val] +
                                   [len(y_val), len(sample_weight_val)])) == 1
                else:
                    assert len(X_val) == len(y_val) == len(sample_weight_val)
                X_val = standardize_X(X_val)
                y_val = standardize_y(y_val)
                sample_weight_val = standardize_weights(y_val,
                                                        sample_weight=sample_weight_val,
                                                        sample_weight_mode=self.sample_weight_mode)
            else:
                raise Exception('Invalid format for validation data; '
                                'provide a tuple (X_val, y_val) or '
                                '(X_val, y_val, sample_weight). '
                                'X_val may be a numpy array or a list of '
                                'numpy arrays depending on your model input.')
            val_ins = X_val + [y_val, sample_weight_val]

        elif 0 < validation_split < 1:
            split_at = int(len(X[0]) * (1 - validation_split))
            X, X_val = (slice_X(X, 0, split_at), slice_X(X, split_at))
            y, y_val = (slice_X(y, 0, split_at), slice_X(y, split_at))
            if sample_weight is not None:
                sample_weight, sample_weight_val = (slice_X(sample_weight, 0, split_at), slice_X(sample_weight, split_at))
                sample_weight_val = standardize_weights(y_val,
                                                        sample_weight=sample_weight_val,
                                                        sample_weight_mode=self.sample_weight_mode)
            else:
                sample_weight_val = standardize_weights(y_val)
            val_ins = X_val + [y_val, sample_weight_val]

        if show_accuracy:
            f = self._train_with_acc
            out_labels = ['loss', 'acc']
        else:
            f = self._train
            out_labels = ['loss']

        sample_weight = standardize_weights(y, class_weight=class_weight,
                                            sample_weight=sample_weight,
                                            sample_weight_mode=self.sample_weight_mode)
        ins = X + [y, sample_weight]
        metrics = ['loss', 'acc', 'val_loss', 'val_acc']
        return self._fit(f, ins, out_labels=out_labels,
                         batch_size=batch_size, nb_epoch=nb_epoch,
                         verbose=verbose, callbacks=callbacks,
                         val_f=val_f, val_ins=val_ins,
                         shuffle=shuffle, metrics=metrics)

    def predict(self, X, batch_size=128, verbose=0):
        '''Generate output predictions for the input samples
        batch by batch.

        # Arguments
            X: the input data, as a numpy array.
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A numpy array of predictions.
        '''
        X = standardize_X(X)
        return self._predict_loop(self._predict, X, batch_size, verbose)[0]

    def predict_proba(self, X, batch_size=128, verbose=1):
        '''Generate class probability predictions for the input samples
        batch by batch.

        # Arguments
            X: the input data, as a numpy array.
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A numpy array of probability predictions.
        '''
        preds = self.predict(X, batch_size, verbose)
        if preds.min() < 0 or preds.max() > 1:
            warnings.warn('Network returning invalid probability values.')
        return preds

    def predict_classes(self, X, batch_size=128, verbose=1):
        '''Generate class predictions for the input samples
        batch by batch.

        # Arguments
            X: the input data, as a numpy array.
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A numpy array of class predictions.
        '''
        proba = self.predict(X, batch_size=batch_size, verbose=verbose)
        if self.class_mode == 'categorical':
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    def evaluate(self, X, y, batch_size=128, show_accuracy=False,
                 verbose=1, sample_weight=None):
        '''Compute the loss on some input data, batch by batch.

        # Arguments
            X: input data, as a numpy array.
            y: labels, as a numpy array.
            batch_size: integer.
            show_accuracy: boolean.
            verbose: verbosity mode, 0 or 1.
            sample_weight: sample weights, as a numpy array.
        '''
        if type(X) == list:
            if len(set([len(a) for a in X] + [len(y)])) != 1:
                raise Exception('All input arrays and the target array must '
                                'have the same number of samples.')
        else:
            if len(X) != len(y):
                raise Exception('The input data tensor (X) and '
                                'the target tensor (y) must have '
                                'the same number of samples. Found: '
                                'len(X) = {}, len(y) = {}'.format(len(X), len(y)))
        if sample_weight is not None:
            assert len(sample_weight) == len(y), ('"sample_weight" must have '
                                                  'the same number of samples '
                                                  'as X and y.')
        X = standardize_X(X)
        y = standardize_y(y)
        sample_weight = standardize_weights(y, sample_weight=sample_weight,
                                            sample_weight_mode=self.sample_weight_mode)

        ins = X + [y, sample_weight]
        if show_accuracy:
            f = self._test_with_acc
        else:
            f = self._test
        outs = self._test_loop(f, ins, batch_size, verbose)
        if show_accuracy:
            return outs
        else:
            return outs[0]

    def train_on_batch(self, X, y, accuracy=False,
                       class_weight=None, sample_weight=None):
        '''Single gradient update over one batch of samples.

        Returns the loss over the data,
        or a tuple `(loss, accuracy)` if `accuracy=True`.

        Arguments: see `fit` method.
        '''
        if type(X) == list:
            if len(set([len(a) for a in X] + [len(y)])) != 1:
                raise Exception('All input arrays and the target array must '
                                'have the same number of samples.')
        else:
            if len(X) != len(y):
                raise Exception('The input data tensor (X) and '
                                'the target tensor (y) must have '
                                'the same number of samples. Found: '
                                'len(X) = {}, len(y) = {}'.format(len(X), len(y)))
        if sample_weight is not None:
            assert len(sample_weight) == len(y), ('"sample_weight" must have '
                                                  'the same number of samples '
                                                  'as X and y.')
        X = standardize_X(X)
        y = standardize_y(y)
        sample_weight = standardize_weights(y, class_weight=class_weight,
                                            sample_weight=sample_weight,
                                            sample_weight_mode=self.sample_weight_mode)
        ins = X + [y, sample_weight]
        if accuracy:
            return self._train_with_acc(ins)
        else:
            return self._train(ins)

    def test_on_batch(self, X, y, accuracy=False, sample_weight=None):
        '''Returns the loss over a single batch of samples,
        or a tuple `(loss, accuracy)` if `accuracy=True`.

        Arguments: see `fit` method.
        '''
        if type(X) == list:
            if len(set([len(a) for a in X] + [len(y)])) != 1:
                raise Exception('All input arrays and the target array must '
                                'have the same number of samples.')
        else:
            if len(X) != len(y):
                raise Exception('The input data tensor (X) and '
                                'the target tensor (y) must have '
                                'the same number of samples. Found: '
                                'len(X) = {}, len(y) = {}'.format(len(X), len(y)))
        if sample_weight is not None:
            assert len(sample_weight) == len(y), ('"sample_weight" must have '
                                                  'the same number of samples '
                                                  'as X and y.')
        X = standardize_X(X)
        y = standardize_y(y)
        sample_weight = standardize_weights(y, sample_weight=sample_weight,
                                            sample_weight_mode=self.sample_weight_mode)

        ins = X + [y, sample_weight]
        if accuracy:
            return self._test_with_acc(ins)
        else:
            return self._test(ins)

    def predict_on_batch(self, X):
        '''Returns predictions for a single batch of samples.
        '''
        ins = standardize_X(X)
        return self._predict(ins)

    def save_weights(self, filepath, overwrite=False):
        '''Dump all layer weights to a HDF5 file.
        '''
        import h5py
        import os.path
        # if file exists and should not be overwritten
        if not overwrite and os.path.isfile(filepath):
            import sys
            get_input = input
            if sys.version_info[:2] <= (2, 7):
                get_input = raw_input
            overwrite = get_input('[WARNING] %s already exists - overwrite? '
                                  '[y/n]' % (filepath))
            while overwrite not in ['y', 'n']:
                overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
            if overwrite == 'n':
                return
            print('[TIP] Next time specify overwrite=True in save_weights!')

        f = h5py.File(filepath, 'w')
        f.attrs['nb_layers'] = len(self.layers)
        for k, l in enumerate(self.layers):
            g = f.create_group('layer_{}'.format(k))
            weights = l.get_weights()
            g.attrs['nb_params'] = len(weights)
            for n, param in enumerate(weights):
                param_name = 'param_{}'.format(n)
                param_dset = g.create_dataset(param_name, param.shape,
                                              dtype=param.dtype)
                param_dset[:] = param
        f.flush()
        f.close()

    def load_weights(self, filepath):
        '''Load all layer weights from a HDF5 save file.
        '''
        import h5py
        f = h5py.File(filepath, mode='r')
        for k in range(f.attrs['nb_layers']):
            # This method does not make use of Sequential.set_weights()
            # for backwards compatibility.
            g = f['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            self.layers[k].set_weights(weights)
        f.close()

    def _check_generator_output(self, generator_output, stop):
        '''Validates the output of a generator. On error, calls
        stop.set().

        # Arguments
            generator_output: output of a data generator.
            stop: threading event to be called to
                interrupt training/evaluation.
        '''
        if not hasattr(generator_output, '__len__'):
            stop.set()
            raise Exception('The generator output must be a tuple. Found: ' +
                            str(type(generator_output)))
        if len(generator_output) == 2:
            X, y = generator_output
            if type(X) == list:
                assert len(set([len(a) for a in X] + [len(y)])) == 1
            else:
                assert len(X) == len(y)
                X = [X]
            sample_weight = None
        elif len(generator_output) == 3:
            X, y, sample_weight = generator_output
            if type(X) == list:
                assert len(set([len(a) for a in X] +
                               [len(y), len(sample_weight)])) == 1
            else:
                assert len(X) == len(y) == len(sample_weight)
                X = [X]
        else:
            stop.set()
            raise Exception('The generator output tuple must have '
                            '2 or 3 elements.')

        sample_weight = standardize_weights(y, sample_weight=sample_weight,
                                            sample_weight_mode=self.sample_weight_mode)
        return X, y, sample_weight

    def evaluate_generator(self, generator, val_samples, show_accuracy=False,
                           verbose=1, **kwargs):
        '''Evaluates the model on a generator. The generator should
        return the same kind of data with every yield as accepted
        by `evaluate`

        Arguments:
            generator:
                generator yielding dictionaries of the kind accepted
                by `evaluate`, or tuples of such dictionaries and
                associated dictionaries of sample weights.
            val_samples:
                total number of samples to generate from `generator`
                to use in validation.
            show_accuracy: whether to display accuracy in logs.
            verbose: verbosity mode, 0 (silent), 1 (per-batch logs),
                or 2 (per-epoch logs).
        '''
        done_samples = 0
        all_outs = None
        weights = []
        q, _stop = generator_queue(generator, **kwargs)

        while done_samples < val_samples:
            X, y, sample_weight = self._check_generator_output(q.get(), _stop)
            do_samples = len(X[0])
            outs = self.evaluate(X, y, batch_size=do_samples,
                                 sample_weight=sample_weight,
                                 show_accuracy=show_accuracy,
                                 verbose=verbose)
            if show_accuracy:
                if all_outs is None:
                    all_outs = [[] for _ in outs]
                for ox, out in enumerate(outs):
                    all_outs[ox].append(out)
            else:
                if all_outs is None:
                    all_outs = []
                all_outs.append(outs)

            done_samples += do_samples
            weights.append(do_samples)

        _stop.set()
        if show_accuracy:
            return [np.average(outx, weights=weights)
                    for outx in all_outs]
        else:
            return np.average(np.asarray(all_outs),
                              weights=weights)

    def fit_generator(self, generator, samples_per_epoch, nb_epoch,
                      verbose=1, show_accuracy=False, callbacks=[],
                      validation_data=None, nb_val_samples=None,
                      class_weight=None,
                      nb_worker=1, nb_val_worker=None):
        '''Fit a model on data generated batch-by-batch by a Python generator.
        The generator is run in parallel to the model, for efficiency,
        and can be run by multiple workers at the same time.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.

        # Arguments
            generator: a Python generator,
                yielding either (X, y) or (X, y, sample_weight).
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `samples_per_epoch`
                samples have been seen by the model.
                The output of the generator must be a tuple of either 2 or 3
                numpy arrays.
                If the output tuple has two elements, they are assumed to be
                (input_data, target_data).
                If it has three elements, they are assumed to be
                (input_data, target_data, sample_weight).
                All arrays should contain the same number of samples.
            samples_per_epoch: integer, number of samples to process before
                starting a new epoch.
            nb_epoch: integer, total number of iterations on the data.
            verbose: verbosity mode, 0, 1, or 2.
            show_accuracy: boolean. Whether to display accuracy (only relevant
                for classification problems).
            callbacks: list of callbacks to be called during training.
            validation_data: tuple of 2 or 3 numpy arrays, or a generator.
                If 2 elements, they are assumed to be (input_data, target_data);
                if 3 elements, they are assumed to be
                (input_data, target_data, sample weights). If generator,
                it is assumed to yield tuples of 2 or 3 elements as above.
                The generator will be called at the end of every epoch until
                at least `nb_val_samples` examples have been obtained,
                with these examples used for validation.
            nb_val_samples: number of samples to use from validation
                generator at the end of every epoch.
            class_weight: dictionary mapping class indices to a weight
                for the class.
            nb_worker: integer, number of workers to use for running
                the generator (in parallel to model training).
                If using multiple workers, the processing order of batches
                generated by the model will be non-deterministic.
                If using multiple workers, make sure to protect
                any thread-unsafe operation done by the generator
                using a Python mutex.
            nb_val_worker: same as `nb_worker`, except for validation data.
                Has no effect if no validation data or validation data is
                not a generator. If `nb_val_worker` is None, defaults to
                `nb_worker`.

        # Returns
            A `History` object.

        # Examples

        ```python
            def generate_arrays_from_file(path):
                while 1:
                    f = open(path)
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x, y = process_line(line)
                        yield x, y
                    f.close()

            model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                                samples_per_epoch=10000, nb_epoch=10)
        ```
        '''
        # TODO: make into kwargs?
        max_data_q_size = 10  # maximum number of batches in queue
        wait_time = 0.05  # in seconds
        epoch = 0

        do_validation = bool(validation_data)
        # python 2 has 'next', 3 has '__next__'
        # avoid any explicit version checks
        val_gen = (hasattr(validation_data, 'next') or
                   hasattr(validation_data, '__next__'))
        if val_gen and not nb_val_samples:
            raise Exception('When using a generator for validation data, '
                            'you must specify a value for "nb_val_samples".')
        if nb_val_worker is None:
            nb_val_worker = nb_worker

        if show_accuracy:
            out_labels = ['loss', 'acc']
        else:
            out_labels = ['loss']
        metrics = ['loss', 'acc', 'val_loss', 'val_acc']

        # prepare callbacks
        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + callbacks + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)

        callbacks._set_model(self)
        callbacks._set_params({
            'nb_epoch': nb_epoch,
            'nb_sample': samples_per_epoch,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': metrics,
        })
        callbacks.on_train_begin()

        # start generator thread storing batches into a queue
        data_gen_queue, _data_stop = generator_queue(generator, max_q_size=max_data_q_size,
                                                     wait_time=wait_time, nb_worker=nb_worker)
        if do_validation and not val_gen:
            X_val, y_val, sample_weight_val = self._check_generator_output(validation_data,
                                                                           _data_stop)
            self.validation_data = X_val + [y_val, sample_weight_val]
        else:
            self.validation_data = None

        self.stop_training = False
        while epoch < nb_epoch:
            callbacks.on_epoch_begin(epoch)
            samples_seen = 0
            batch_index = 0
            while samples_seen < samples_per_epoch:
                generator_output = None
                while not _data_stop.is_set():
                    if not data_gen_queue.empty():
                        generator_output = data_gen_queue.get()
                        break
                    else:
                        time.sleep(wait_time)

                X, y, sample_weight = self._check_generator_output(generator_output,
                                                                   _data_stop)
                batch_logs = {}
                batch_size = len(X[0])
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = self.train_on_batch(X, y,
                                           accuracy=show_accuracy,
                                           sample_weight=sample_weight,
                                           class_weight=class_weight)
                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                # construct epoch logs
                epoch_logs = {}
                batch_index += 1
                samples_seen += batch_size

                # epoch finished
                if samples_seen >= samples_per_epoch and do_validation:
                    if val_gen:
                        val_outs = self.evaluate_generator(validation_data,
                                                           nb_val_samples,
                                                           show_accuracy=show_accuracy,
                                                           verbose=0, nb_worker=nb_val_worker,
                                                           wait_time=wait_time)
                    else:
                        val_outs = self.evaluate(X_val, y_val,
                                                 show_accuracy=show_accuracy,
                                                 sample_weight=sample_weight_val,
                                                 verbose=0)
                    if type(val_outs) != list:
                        val_outs = [val_outs]
                    # same labels assumed
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            if self.stop_training:
                break
        _data_stop.set()
        callbacks.on_train_end()
        return self.history

