from scipy.sparse import csr_matrix, hstack
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfTransformer


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


def main(size="sm"):
    path = "data/" + size
    features, labels, vocab = load_sparse_csr(path + "/feature_matrix.3.csr.npz")
    print features.shape
    vocab = dict(vocab.tolist())

    tfer = TfidfTransformer()
    tfer.fit(features[:,:32],labels)
    features_tf = tfer.transform(features[:,:32])
    features = hstack([features_tf, features[:,32:]], format='csr')

    labels = convert_labels(labels)

    features2, _, vocab2 = load_sparse_csr(path + "/feature_matrix.5.csr.npz")
    print features2.shape
    vocab2 = dict(vocab2.tolist())
    features2 = features2[:,:-5]

    tfer.fit(features2)
    features2 = tfer.transform(features2)

    features = hstack([features, features2],format='csr')

    for key, value in vocab2.iteritems():
        vocab2[key] = value + features.shape[1]

    features3, _, vocab3 = load_sparse_csr(path + "/feature_matrix.10.csr.npz")
    print features3.shape
    vocab3 = dict(vocab3.tolist())
    features3 = features3[:,:-5]

    tfer.fit(features3)
    features3 = tfer.transform(features3)

    features = hstack([features, features3],format='csr', dtype="Float32")

    for key, value in vocab3.iteritems():
        vocab3[key] = value + features.shape[1] + features2.shape[1]

    vocab = dict(vocab.items() + vocab2.items() + vocab3.items())

    save_sparse_csr("data/" + size + "/feature_matrix.3.5.10.csr", features, labels, vocab)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        main(args[0])
    else:
        main()