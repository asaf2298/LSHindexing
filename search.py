import os
import warnings
import json
import heapq
import math
from collections import defaultdict
from itertools import chain

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings('ignore')
hf_logging.set_verbosity_error()

# consine_similarity
# LSHIndex
#search
class cosine_similarty:
  def __init__(self, a, b):
    self.a = a
    self.b = b
  def 

def cosine_similarity(a, b):
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise ValueError("a and b must be numpy arrays")
    if a.dtype != np.float32 or b.dtype != np.float32:
        raise ValueError("a and b must be with the same dtype")
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float32(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))

class LSHIndex:
    def __init__(self, vectors, num_tables, num_bits, **kwargs):
        self.vectors = vectors
        self.num_tables = num_tables
        self.num_bits = num_bits


