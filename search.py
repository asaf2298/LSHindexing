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