from keras.layers import Dense, Embedding, Input

from doc2vec.model import model


class DBOW(model.Doc2VecModel):

    def build(self):
        doc_input = Input(shape=(1,))
      
        embedded_doc = Embedding(input_dim=self._num_docs,
                                 output_dim=self._embedding_size,
                                 input_length=1,
                                 name=model.DOC_EMBEDDINGS_LAYER_NAME)(doc_input)
      
        softmax = Dense((self._window_size, self._vocab_size),
                        activation='softmax')(embedded_doc)
      
        self._model = Model(inputs=doc_input, outputs=softmax)
