from scipy.sparse import csr_matrix, hstack
import numpy as np


def convert_labels(labels):
    """ 
    Convert labels to indexes
    Params:
        labels...Original k class string labels
    Returns:
        Categorical label vector
    """
    label_idxs = {}
    new_labels = np.empty(len(labels))
    for x in range(len(labels)):
        new_labels[x] = label_idxs.setdefault(labels[x], len(label_idxs))
    return new_labels


def save_sparse_csr(filename,array, labels, vocab):
    """ 
    Save csr matrix in loadable format
    Params:
        filename...save path
        array......csr matrix
        labels.....ordered true labels
        vocab......maps kmer to feature vector index
    """
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape, labels=labels, vocab=vocab)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape']), loader['labels'], loader['vocab']


path = "data/lg/"
features, labels, vocab = load_sparse_csr(path + "feature_matrix.3.csr.npz")
vocab = dict(vocab.tolist())

labels = convert_labels(labels)

features2, _, vocab2 = load_sparse_csr(path + "feature_matrix.5.csr.npz")
vocab2 = dict(vocab2.tolist())
features2 = features2[:,:-5]
features = hstack([features, features2],format='csr')

for key, value in vocab2.iteritems():
    vocab2[key] = value + features.shape[1]

features3, _, vocab3 = load_sparse_csr(path + "feature_matrix.10.csr.npz")
vocab3 = dict(vocab3.tolist())
features3 = features3[:,:-5]
features = hstack([features, features3],format='csr', dtype="Float32")

for key, value in vocab3.iteritems():
    vocab3[key] = value + features.shape[1] + features2.shape[1]

vocab = vocab + vocab2 + vocab3

save_sparse_csr("data/lg/feature_matrix.3.5.10.csr", features, labels, vocab)