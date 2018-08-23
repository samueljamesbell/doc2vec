import collections
import logging
import pickle


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_SIZE = 10000
DEFAULT_RARE_THRESHOLD = 0


_UNKNOWN = '<unk>'


class Vocabulary(object):

    def __init__(self):
        self._vocab = {}
        self._vocab_inverse = {}

    @property
    def size(self):
        return len(self._vocab)

    def __contains__(self, token):
        return token in self._vocab

    def build(self, tokens,
              max_size=DEFAULT_SIZE,
              rare_threshold=DEFAULT_RARE_THRESHOLD):
        logger.info('Building vocab')

        def _unk(token, count):
            """Replace a token by unk if count is below threshold."""
            return token if count > rare_threshold else _UNKNOWN

        c = collections.Counter(tokens)
        self._vocab = {_unk(t, c): i for i, (t, c) in enumerate(c.most_common(max_size))}
        # Force unk to be in vocab if not already
        if _UNKNOWN not in self._vocab:
            self._vocab[_UNKNOWN] = len(self._vocab)

        self._vocab_inverse = _inverse(self._vocab)

        logger.info('Vocab size: %d', self.size)

    def load(self, path):
        logger.info('Loading vocab from %s', path)

        with open(path, 'rb') as f:
            self._vocab = pickle.load(f) or {}
        self._vocab_inverse = _inverse(self._vocab)

        logger.info('Vocab size: %d', self.size)

    def save(self, path):
        logger.info('Saving vocab to %s', path)

        with open(path, 'wb') as f:
            pickle.dump(self._vocab, f)

    def to_id(self, token): 
        return self._vocab.get(token, self._vocab[_UNKNOWN])

    def to_token(self, id):
        return self._vocab_inverse[id]

    def to_ids(self, tokens):
        if not tokens:
            return []

        return [self.to_id(t) for t in tokens]

    def to_tokens(self, ids):
        return [self.to_token(id) for id in ids]


def _inverse(d):
    return {v: k for k, v in d.items()}
