from keras.layers import Dense, Embedding, Input, Lambda
from keras.models import Model
import tensorflow as tf

from doc2vec.model import model


def _stack(window_size):
    def _lambda(tensor):
        import tensorflow as tf
        return tf.stack([tensor] * window_size, axis=1)
    return _lambda


def _squeeze(axis=-1):
    def _lambda(tensor):
        import tensorflow as tf
        return tf.squeeze(tensor, axis=axis)
    return _lambda


class DBOW(model.Doc2VecModel):

    def build(self):
        doc_input = Input(shape=(1,))
      
        embedded_doc = Embedding(input_dim=self._num_docs,
                                 output_dim=self._embedding_size,
                                 input_length=1,
                                 name=model.DOC_EMBEDDINGS_LAYER_NAME)(doc_input)

        embedded_doc = Lambda(_squeeze(axis=1))(embedded_doc)

        stack = Lambda(_stack(self._window_size))(embedded_doc)
      
        softmax = Dense(self._vocab_size, activation='softmax')(stack)
      
        self._model = Model(inputs=doc_input, outputs=softmax)
