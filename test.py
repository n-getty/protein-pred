import numpy as np
import sys
from scipy.sparse import csr_matrix
from collections import Counter

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape']), loader['labels']


'''file="data/feature_matrix.sm.10.csr.npz"
M, _ = load_sparse_csr(file)
ocols = M.shape[1]
M = M[:,M.getnnz(0)>0]
print M.getnnz(0)>0
print ocols - M.shape[1]'''

x = [1,2,3,4,5]
y = [6,7,8,9,10]
z=np.array([x,y])

print z[:,:-2]