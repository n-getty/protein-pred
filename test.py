import numpy as np
import sys
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.preprocessing import normalize
import pandas as pd


file = "data/coreseed.train.tsv"
data = pd.read_csv(file, names=["label", "dna"], usecols=[1, 5], delimiter='\t', header=0)

print data.shape
print len(Counter(data.label))