import numpy as np
import sys
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.preprocessing import normalize
import pandas as pd
from memory_profiler import memory_usage
from sklearn.feature_extraction.text import TfidfTransformer
from time import time
#file = "data/rep.1000ec.pgf.seqs.filter"
file = "data/coreseed.train.tsv"
data = pd.read_csv(file, names=["label", "aa", "dna"], usecols=[1, 6, 7], delimiter='\t', header=0)



c = Counter(data.label)
print c.most_common()[:10]
print c.most_common()[-10:]