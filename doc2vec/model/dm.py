import logging

import h5py
from keras.callbacks import EarlyStopping
from keras.layers import Average, Concatenate, Dense, Embedding, Input, Lambda
from keras.models import Model, load_model
from keras.optimizers import SGD
import tensorflow as tf


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _split(window_size):
    def _lambda(tensor):
        import tensorflow as tf
        return tf.split(tensor, window_size + 1, axis=1)
    return _lambda


def _squeeze():
    def _lambda(tensor):
        import tensorflow as tf
        return tf.squeeze(tensor, axis=1)
    return _lambda


class DM(object):

    def __init__(self, window_size, vocab_size, num_docs):
        self._window_size = window_size
        self._vocab_size = vocab_size
        self._num_docs = num_docs

        self._model = None

    @property
    def doc_embeddings(self):
        # TODO: Name the embedding layer and then look it up by key instead
        # of relying on ordering.
        return np.array(self._model.layers[2].get_weights()[0])

    def build(self):
        sequence_input = Input(shape=(self._window_size,))
        doc_input = Input(shape=(1,))
      
        embedded_sequence = Embedding(input_dim=self._vocab_size, output_dim=300, input_length=self._window_size)(sequence_input)
        embedded_doc = Embedding(input_dim=self._num_docs, output_dim=300, input_length=1)(doc_input)
      
        embedded = Concatenate(axis=1)([embedded_doc, embedded_sequence])
        split = Lambda(_split(self._window_size))(embedded)
        averaged = Average()(split)
        squeezed = Lambda(_squeeze())(averaged)
      
        softmax = Dense(self._vocab_size, activation='softmax')(squeezed)
      
        self._model = Model(inputs=[doc_input, sequence_input], outputs=softmax)

    def compile(self, optimizer=None):
        if not optimizer:
            optimizer = SGD(lr=0.001, momentum=0.9, nesterov=True)

        self._model.compile(optimizer=optimizer,
                            loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'])

    def train(self, generator, steps_per_epoch=10000, epochs=250):
        history = self._model.fit_generator(
	    generator,
	    callbacks=[
		EarlyStopping(monitor='loss', patience=10)
	      ],
	    steps_per_epoch=steps_per_epoch,
	    epochs=epochs)
  
        return history

    def save(self, path):
        logger.info('Saving model to %s', path)
        self._model.save(path)

    def save_doc_embeddings(self, path):
        logger.info('Saving doc embeddings to %s', path)
        with h5py.File(path, 'w') as f:
            f.create_dataset("doc_embeddings", data=self.doc_embeddings)

    def load(self, path):
        logger.info('Loading model from %s', path)
        self._model = load_model(path)
