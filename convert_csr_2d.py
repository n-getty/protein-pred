import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

def save_sparse_csr(filename,array, labels):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape, labels=labels)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape']), loader['labels']


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


def convert_csr_2d(csr_matrix):
    csr_2d = []
    max_len = 0
    for x in range(csr_matrix.shape[0]):
        l = len(csr_matrix.getrow(x).data)
        if l > 0: max_len = l

    for x in range(csr_matrix.shape[0]):
        row = csr_matrix.getrow(x)
        data = row.data
        indices = row.indices
        l = len(data)
        if l < max_len:
            pad = [0]*(max_len-l)
            data = np.append(data, pad)
            indices = np.append(indices, pad)

        csr_2d.append(np.column_stack((data,indices)))

    return np.array(csr_2d)



M, labels = load_sparse_csr("data/feature_matrix.3.csr.npz")
csr_2d = convert_csr_2d(M)
np.savez("data/feature_matrix.sm.3.csr_2d", data = csr_2d, labels = labels)
