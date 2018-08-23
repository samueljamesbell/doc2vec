import argparse
import itertools

from doc2vec.data import batch, doc
from doc2vec.model import dm
from doc2vec import vocab


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='Path to documents directory')
                        
    parser.add_argument('--save', help='Path to save model')
    parser.add_argument('--save_period', help='Save model every n epochs')
    parser.add_argument('--save_vocab', help='Path to save vocab file')
    parser.add_argument('--save_doc_embeddings',
                        help='Path to save doc embeddings file')
    parser.add_argument('--save_doc_embeddings_period',
                        help='Save doc embeddings every n epochs')

    parser.add_argument('--load', help='Path to load model')
    parser.add_argument('--load_vocab', help='Path to load vocab file')

    parser.add_argument('--early_stopping_patience',
                        help='Stop after no loss decrease for n epochs')

    parser.add_argument('--vocab_size', default=vocab.DEFAULT_SIZE,
                        help='Max vocabulary size; ignored if loading from file')
    parser.add_argument('--vocab_rare_threshold',
                        default=vocab.DEFAULT_RARE_THRESHOLD,
                        help=('Words less frequent than this threshold '
                              'will be considered unknown'))

    parser.add_argument('--window_size',
                        default=dm.DEFAULT_WINDOW_SIZE,
                        help='Context window size')
    parser.add_argument('--embedding_size',
                        default=dm.DEFAULT_EMBEDDING_SIZE,
                        help='Word and document embedding size')

    parser.add_argument('--num_epochs',
                        default=dm.DEFAULT_NUM_EPOCHS,
                        help='Number of epochs to train for')
    parser.add_argument('--steps_per_epoch',
                        default=dm.DEFAULT_STEPS_PER_EPOCH,
                        help='Number of samples per epoch')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', dest='train', action='store_true')
    group.add_argument('--no-train', dest='train', action='store_false')
    group.set_defaults(train=False)

    return parser.parse_args()


def main():
    args = _parse_args()

    tokens_by_doc_id = doc.tokens_by_doc_id(args.path)

    num_docs = len(tokens_by_doc_id)

    v = vocab.Vocabulary()
    if args.load_vocab:
        v.load(args.load_vocab)
    else:
        all_tokens = list(itertools.chain.from_iterable(tokens_by_doc_id.values()))
        v.build(all_tokens, max_size=args.vocab_size)
        if args.save_vocab:
            v.save(args.save_vocab)

    token_ids_by_doc_id = {d: v.to_ids(t) for d, t in tokens_by_doc_id.items()}

    m = dm.DM(args.window_size, v.size, num_docs,
              embedding_size=args.embedding_size)

    if args.load:
        m.load(args.load) 
    else:
        m.build()
        m.compile()

    if args.train:
        all_data = batch.batch(
                batch.data_generator(
                    token_ids_by_doc_id,
                    args.window_size,
                    v.size))

        history = m.train(
                all_data,
                epochs=args.num_epochs,
                steps_per_epoch=args.steps_per_epoch,
                early_stopping_patience=args.early_stopping_patience,
                save_path=args.save,
                save_period=args.save_period,
                save_doc_embeddings=args.save_doc_embeddings,
                save_doc_embeddings_period=args.save_doc_embeddings_period)

    if args.save:
        m.save(
            args.save.format({'epoch': len(history)}))

    if args.save_doc_embeddings:
        m.save_doc_embeddings(
            args.save_doc_embeddings.format({'epoch': len(history)}))
