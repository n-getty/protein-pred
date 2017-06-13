import numpy as np
import sys
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.preprocessing import normalize
import pandas as pd
from memory_profiler import memory_usage


mem_usage = memory_usage(-1, interval=.2)
x = np.zeros((50000,50000))
print(mem_usage)