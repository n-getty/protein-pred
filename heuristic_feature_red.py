import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape']), loader['labels']


def red_features(features, labels):
    feature_count = features.shape[1]
    rows = features.shape[0]

    nonzero_counts = features.getnnz(0)
    nonz = nonzero_counts > 0
    print "Number of columns with no nonzero counts", (features.shape[1] - np.sum(nonz))

    nonz = nonzero_counts > 50
    print "Number of columns with more than 50 nonzero counts", np.sum(nonz)

    nonz = nonzero_counts > 100
    print "Number of columns with more than 100 nonzero counts", np.sum(nonz)

    nonz = nonzero_counts > 150
    print "Number of columns with more than 150 nonzero counts", np.sum(nonz)

    nonz = nonzero_counts > 200
    print "Number of columns with more than 200 nonzero counts", np.sum(nonz)

    feature_counts = Counter(nonzero_counts)

    print feature_counts[1]

    df = pd.DataFrame.from_dict(feature_counts, orient='index')
    df.plot(kind='bar')
    plt.show()

    #features = features[:, nonz]


def main():
    print "Using small dataset 10mer features"
    file = "feature_matrix.sm.10.csr.npz"
    features, labels = load_sparse_csr("data/" + file)
    red_features(features, labels)

main()