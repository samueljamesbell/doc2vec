import argparse
import itertools

import nltk

from data import batch, doc
from model import dm
import vocab


nltk.download('punkt')


_DEFAULT_WINDOW_SIZE = 8
_DEFAULT_VOCAB_SIZE = 10000


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='Path to documents directory')
                        
    parser.add_argument('--save', help='Path to save model')
    parser.add_argument('--save_vocab', help='Path to save vocab file')
    parser.add_argument('--save_doc_embeddings', help='Path to save doc embeddings file')
    parser.add_argument('--load', help='Path to load model')
    parser.add_argument('--load_vocab', help='Path to load vocab file')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', dest='train', action='store_true')
    group.add_argument('--no-train', dest='train', action='store_false')
    group.set_defaults(train=False)

    return parser.parse_args()


def _main():
    args = _parse_args()

    tokens_by_doc_id = doc.tokens_by_doc_id(args.path)

    num_docs = len(tokens_by_doc_id)

    v = vocab.Vocabulary()
    if args.load_vocab:
        v.load(args.load_vocab)
    else:
        all_tokens = list(itertools.chain.from_iterable(tokens_by_doc_id.values()))
        v.build(all_tokens, max_size=_DEFAULT_VOCAB_SIZE)
        if args.save_vocab:
            v.save(args.save_vocab)

    token_ids_by_doc_id = {d: v.to_ids(t) for d, t in tokens_by_doc_id.items()}

    m = dm.DM(_DEFAULT_WINDOW_SIZE, v.size, num_docs)

    if args.load:
        m.load(args.load) 
    else:
        m.build()
        m.compile()

    if args.train:
        all_data = batch.batch(
                batch.data_generator(
                    token_ids_by_doc_id,
                    _DEFAULT_WINDOW_SIZE,
                    v.size))

        m.train(all_data)

    if args.save:
        m.save(args.save)

    if args.save_doc_embeddings:
        m.save_doc_embeddings(args.save_doc_embeddings)


if __name__ == '__main__':
    _main()
