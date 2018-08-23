import itertools 
from progressbar import progressbar
import random

from keras.utils import to_categorical
import numpy as np


def data_generator(token_ids_by_doc_id, window_size, vocab_size):
    doc_ids = list(token_ids_by_doc_id.keys())
  
    for doc_id in progressbar(itertools.cycle(doc_ids)):
        token_ids = token_ids_by_doc_id[doc_id]
        num_tokens = len(token_ids)
    
        if num_tokens <= window_size:
            continue
    
        target_idx = random.randint((num_tokens - offset) - 1)
        target_id = token_ids[target_idx]
      
        context_window = token_ids[target_idx:target_idx+offset]
    
        yield (doc_id, to_categorical(context_window, num_classes=vocab_size))


def batch(data, batch_size=32):
    while True:
        batch = itertools.islice(data, batch_size)
    
        x = []
        y = []
    
        for item in batch:
            doc_id, context_window = item
      
            x.append(doc_id)
            y.append(context_window)
      
        yield np.array(x), np.array(y)
