from abc import ABCMeta
import logging

import h5py
from keras.callbacks import CallBack, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD
import numpy as np


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_WINDOW_SIZE = 8
DEFAULT_EMBEDDING_SIZE = 300
DEFAULT_NUM_EPOCHS = 250
DEFAULT_STEPS_PER_EPOCH = 10000

DOC_EMBEDDINGS_LAYER_NAME = 'doc_embeddings'


class Doc2VecModel(object):

    __metaclass__ = ABCMeta

    def __init__(self, window_size, vocab_size, num_docs,
                 embedding_size=DEFAULT_EMBEDDING_SIZE):
        self._window_size = window_size
        self._vocab_size = vocab_size
        self._num_docs = num_docs

        self._embedding_size = embedding_size

        self._model = None

    @property
    def doc_embeddings(self):
        return _doc_embeddings_from_model(self._model)

    def build(self):
        raise NotImplementedError()

    def compile(self, optimizer=None):
        if not optimizer:
            optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)

        self._model.compile(optimizer=optimizer,
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])

    def train(self, generator,
              steps_per_epoch=DEFAULT_STEPS_PER_EPOCH,
              epochs=DEFAULT_NUM_EPOCHS,
              early_stopping_patience=None,
              save_path=None, save_period=None,
              save_doc_embeddings_path=None, save_doc_embeddings_period=None):

        callbacks=[]
        if early_stopping_patience:
            callbacks.append(EarlyStopping(monitor='loss',
                                           patience=early_stopping_patience))
        if save_path and save_period:
            callbacks.append(ModelCheckpoint(save_path,
                                             period=save_period))
        if save_doc_embeddings_path and save_doc_embeddings_period:
            callbacks.append(_SaveDocEmbeddings(save_doc_embeddings_path,
                                                period=save_doc_embeddings_period))

        history = self._model.fit_generator(
	    generator,
	    callbacks=callbacks,
	    steps_per_epoch=steps_per_epoch,
	    epochs=epochs)
  
        return history

    def save(self, path):
        logger.info('Saving model to %s', path)
        self._model.save(path)

    def save_doc_embeddings(self, path):
        _write_doc_embeddings(self.doc_embeddings, path)

    def load(self, path):
        logger.info('Loading model from %s', path)
        self._model = load_model(path)


class _SaveDocEmbeddings(Callback):

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.params['period'] != 0:
            return

        path = self.params['path'].format({'epoch': epoch})
        embeddings = _doc_embeddings_from_model(self.model)
        _write_doc_embeddings(embeddings, path)


def _doc_embeddings_from_model(keras_model):
    for layer in keras_model.layers:
        if layer.get_config()['name'] == _DOC_EMBEDDINGS_LAYER_NAME:
            return layer.get_weights()[0]


def _write_doc_embeddings(doc_embeddings, path):
    logger.info('Saving doc embeddings to %s', path)
    with h5py.File(path, 'w') as f:
        f.create_dataset(_DOC_EMBEDDINGS_LAYER_NAME, data=doc_embeddings)
