# doc2vec

A simple and readable implementation of doc2vec [1], using Python 3, Keras and TensorFlow.

## Installation
```
pip install -r requirements.txt
python setup.py install
```

## Assumptions
This implementation assumes that your documents are all in the same directory,
and named with monotonically incrementing integer IDs, e.g. `0.txt`, `1.txt`.
Each file should contain an ordinary text document, i.e. without any special
preprocessing.

## Usage
```
doc2vec path/to/docs/ \
    --save path/to/save/model.hdf5 \
    --save_vocab path/to/save/vocab.vocab \
    --train
```

By default, this will use the Distributed Memory (DM) model. You can also use 
the Distributed Bag-Of-Words (DBOW) model with:
```
doc2vec path/to/docs/ --model dbow
```

Previously trained models can be loaded with:
```
doc2vec path/to/docs/ \
    --load path/to/load/model.hdf5 \
    --load_vocab path/to/load/vocab.vocab
```

And document embeddings can be written to file as follows:
```
doc2vec path/to/docs/ \
    --load path/to/load/model.hdf5 \
    --load_vocab path/to/load/vocab.vocab \
    --save_doc_embeddings path/to/save/embeddings.hdf5
```

Finally, you can see all available options and model parameters with:
```
doc2vec -h
```

## References
1. Le, Quoc, and Mikolov, Tomas. "Distributed representations of sentences and
   documents." International Conference on Machine Learning. 2014.
2. Bird, Steven, Loper, Edward and Klein, Ewan. "Natural Language
   Processing with Python." O’Reilly Media Inc. 2009.
