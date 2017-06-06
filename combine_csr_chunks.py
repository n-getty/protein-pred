from os import listdir
from os.path import isfile, join
import numpy as np
from scipy.sparse import csr_matrix, vstack
import os, sys


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape']), loader['labels'], loader['vocab']


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


def main(path = "data/feature_matrix.lg.10"):

    files = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    csrs = []
    labels = []
    for f in files:
        csr, l, vocab = load_sparse_csr(f)
        csrs.append(csr)
        labels.extend(l)

    print "Stacking csrs"
    csr_matrix = vstack(csrs)

    save_sparse_csr(path + ".csr", csr_matrix, labels, vocab)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        os.chdir("/home/ngetty/examples/protein-pred")
        args = sys.argv[1:]
        main(args[0])
    else:
        main()