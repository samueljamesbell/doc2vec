from keras.layers import Average, Concatenate, Dense, Embedding, Input, Lambda
from keras.models import Model
import tensorflow as tf

from doc2vec.model import lambdas, model


class DM(model.Doc2VecModel):

    def build(self):
        sequence_input = Input(shape=(self._window_size,))
        doc_input = Input(shape=(1,))

        embedded_sequence = Embedding(input_dim=self._vocab_size,
                                      output_dim=self._embedding_size,
                                      input_length=self._window_size,
                                      name=model.DOC_EMBEDDINGS_LAYER_NAME)(sequence_input)
        embedded_doc = Embedding(input_dim=self._num_docs,
                                 output_dim=self._embedding_size,
                                 input_length=1)(doc_input)
      
        embedded = Concatenate(axis=1)([embedded_doc, embedded_sequence])
        split = Lambda(lambdas.split(self._window_size))(embedded)
        averaged = Average()(split)
        squeezed = Lambda(lambdas.squeeze(axis=1))(averaged)
      
        softmax = Dense(self._vocab_size, activation='softmax')(squeezed)
      
        self._model = Model(inputs=[doc_input, sequence_input], outputs=softmax)
