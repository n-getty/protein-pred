from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


def save_sparse_csr(filename,array, labels):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape, labels=labels)


def convert_labels(labels):
    """ 
    Convert labels to indexes
    Params:
        labels...Original k class string labels
    """
    label_idxs = {}
    new_labels = np.empty(len(labels))
    for x in range(len(labels)):
        new_labels[x] = label_idxs.setdefault(labels[x], len(label_idxs))
    return new_labels


file = "data/rep.1000ec.pgf.seqs.filter"
data = pd.read_csv(file, names=["label"], usecols=[0], delimiter = '\t', header=0)
labels = convert_labels(data.label)
csr = "data/feature_matrix.lg.3.csr.npz"
M = load_sparse_csr(csr)
save_sparse_csr(csr, M, labels)