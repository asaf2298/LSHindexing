import os
import warnings
import json
import heapq
import math
from collections import defaultdict
from itertools import chain
import numpy as np
import heapq

from sentence_transformers import SentenceTransformer
from transformers import logging as hf_logging

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
warnings.filterwarnings('ignore')
hf_logging.set_verbosity_error()

# consine_similarity
# LSHIndex
#search

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
        if not isinstance(vectors, np.ndarray) or not isinstance(num_table, int) or not isinstance(num_bit, int):
          raise ValueError("vectors must be numpy arrays and num_bits and num_tablemust be int type")
        if vectors.dtype != np.float32:
          raise ValueError("vectors must be with the same dtype")
        self.vectors = vectors
        self.num_tables = num_tables
        self.num_bits = num_bits
        seed = kwargs.get("seed", 42)
        RNG = np.random.default_rng(seed)
        self.planes = rng.normal(size=(num_tables, num_bits, vectors.shape[1])).astype(np.float32)
        self.tables = self.build_tables
    def _build_tables(): 
      tables = []
      for idx in range(self.num_tables): # ריצה על כל INDEPANDET HASH
        table = {}
        planes = self.planes[idx]
        for n, d in enumerate(self.vectors): 
          dot_product = (np.dot(planes, d) >= 0).astype(int) # באיזה צד של החציצה הויקטור
          key = 0
          for i in range(self.num_bits):
            key =+ dot_product[i]*(2**i)
            if key not in table:
              table[key] = []
            table[key].append(n)
            tables.append(dict(table))
    return tables
  
    def query(self, q_vec, k):
      matched_hashes = set()
      heap = []
      for idx in range(self.num_tables):
        dot_product_query = (np.dot(self.planes[idx], q_vec) >= 0).astype(int) 
        key_query = sum(dot_product_query[i] * (2 ** i) for i in range(self.num_bits))
        matched_hashes.update(self.tables[idx].get(key_query, []))          
      for n in matched_hashes: # כיוון שמחפשים קי אייברים שזה פחות שווה לגודל המערכת, יעיל יותר למיין קי ערכים תוך כדי שלפיה
          score = cosine_similarity(self.vectors[n], q_vec)
          item = (score, n)   
          if len(heap) < k:
              heapq.heappush(heap, item)
          else:
              if score > heap[0][0]:
                  heapq.heapreplace(heap, item)      
      scoretables = sorted(heap, reverse=True) #O(k log k)
      return scoretables

def search(query, index, encoder, movies, k=5):
    encoded = encoder.encode(query)
    results = index.query(encoded, k)
    res = []
    for score, idx in results:
        movie = dict(movies[idx])
        movie["score"] = float(score)
        res.append(movie)
  return res;
