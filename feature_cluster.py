import numpy as np
import seaborn as sns; sns.set(color_codes=True)
from scipy.sparse import csr_matrix, hstack
import pandas as pd

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype="float32"), loader['labels']


def load_data(size, dna1, dna3, dna5, dna10, aa1, aa2, aa3, aa4):
    path = "data/" + size + '/'

    files = []
    if dna1:
        features, labels = load_sparse_csr(path + "feature_matrix.1.csr.npz")
        files.append(features)
    if dna3:
        features, labels = load_sparse_csr(path + "feature_matrix.3.csr.npz")
        files.append(features)
    if dna5:
        features, labels = load_sparse_csr(path + "feature_matrix.5.csr.npz")
        files.append(features)
    if dna10:
        features, labels = load_sparse_csr(path + "feature_matrix.10.csr.npz")
        files.append(features)
    if aa1:
        features, labels = load_sparse_csr(path + "feature_matrix.aa1.csr.npz")
        files.append(features)
    if aa2:
        features, labels = load_sparse_csr(path + "feature_matrix.aa2.csr.npz")
        files.append(features)
    if aa3:
        features, labels = load_sparse_csr(path + "feature_matrix.aa3.csr.npz")
        files.append(features)
    if aa4:
        features, labels = load_sparse_csr(path + "feature_matrix.aa4.csr.npz")
        files.append(features)

    if not files:
        print "No dataset provided"
        exit(0)

    features = hstack(files, format='csr')

    return features, labels

features, _ = load_data("sm", False, False, False, False, False, True, True, False)

features = features.toarray()
g = sns.clustermap(features, metric="correlation")