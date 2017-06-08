import numpy as np
import sys,os
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA, TruncatedSVD




def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape']), loader['labels']


file="data/feature_matrix.sm.10.csr.npz"
X, _ = load_sparse_csr(file)
print X.shape
svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
svd.fit(X)
X = svd.transform(X)
#pca = PCA(n_components="mle")
#pca.fit(X)
#X = pca.transform(X)
print X.shape