import numpy as np
import seaborn as sns; sns.set(color_codes=True)
from scipy.sparse import csr_matrix, hstack
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype="float32"), loader['labels']


def unique_class_names(names):
    """ 
    Generate ordered unique class names
    Params:
        names....Label for every data point
    Returns:
        Name for each class in the set
    """
    cns = set()
    unique = []
    for c in names:
        if c not in cns:
            unique.append(c)
            cns.add(c)

    return unique


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

core=False
if core:
    print "Loading coreseed functions"
    file = "data/coreseed.train.tsv"
    functions = pd.read_csv(file, names=["peg", "function"], usecols=[0, 2], delimiter='\t', header=0)
else:
    print "Loading sm functions"
    file = "data/ref.100ec.pgf.seqs.filter"
    functions = pd.read_csv(file, names=["peg", "function"], usecols=[2, 5], delimiter='\t')


features, labels = load_data("sm", True, True, False, False, True, True, False, False)
names = unique_class_names(functions.function)
labels = convert_labels(labels)
#uni = range(10)
uni = [19, 83, 36, 9, 61, 99, 22, 2, 7, 17]
#uni = [77, 78, 82, 85, 86, 90, 92, 93, 94, 97]

sub = np.in1d(labels, uni)
features = features[sub]

features = features.toarray()

feat_scores = np.genfromtxt("results/LightGBM07-31_20_42.feat_scores",delimiter=',', dtype="float32")

for x in range(len(features)):
    features[x] = np.multiply(features[x],feat_scores)

normalize(features, copy=False, axis=0)

features = pd.DataFrame(features)

labels = labels[sub]

network_pal = sns.cubehelix_palette(len(uni),
                                    light=.9, dark=.1, reverse=True,
                                    start=1, rot=-2)
network_lut = dict(zip(uni, network_pal))

network_colors = pd.Series(labels).map(network_lut)

g = sns.clustermap(features, method="average", metric="matching", center=0,

                  # Turn off the clustering
                  row_cluster=True, col_cluster=False,

                  # Add colored class labels
                  row_colors=network_colors,

                  # Make the plot look better when many rows/cols
                  linewidths=0, xticklabels=False, yticklabels=False)

for x in uni:
    g.ax_col_dendrogram.bar(0, 0, color=network_lut[x],
                            label=names[x], linewidth=0)

g.ax_col_dendrogram.legend(loc="center", ncol=2)

g.cax.set_position([-.10, .2, .03, .45])
g.savefig("results/clustermaps/clustermap_worst.png")

#g = sns.clustermap(features, metric="correlation", center=0)