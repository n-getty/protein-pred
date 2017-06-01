#/home/ngetty/dev/anaconda2/bin/python
import pandas as pd
import math
from itertools import islice, product
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
import sys
from time import time


def window(seq, k=3):
    """ 
    Generate generate all kmers for a sequence
    Params:
        seq....dna sequence
        k......length of kmer
    """
    it = iter(seq)
    result = tuple(islice(it, k))
    if len(result) == k:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def reverse_complement(kmer):
    """
    Generate a kmers complement
    Params:
        kmer....The kmer to generate the complement for
    """
    comp = {'a': 't', 'c': 'g', 't': 'a', 'g': 'c'}
    rc = ()
    for x in range(len(kmer)):
        rc = rc + (comp[kmer[len(kmer)-x-1]],)
    return rc


def gen_comp_dict(all_kmers):
    comps = {}
    for kmer in all_kmers:
        comps[kmer] = reverse_complement(kmer)
    return comps


def gen_vocab(k=3):
    """
    Generate index kmer pairs for all possible kmers, binning complements together
    Params:
        k....length of kmer
    """
    all_kmers = list(product('acgt', repeat=k))
    vocab = {}
    unique = 0
    comps = gen_comp_dict(all_kmers)
    for mer in all_kmers:
        rc = comps[mer]
        if rc in vocab:
            vocab[mer] = vocab[rc]
        else:
            vocab[mer] = unique
            unique += 1
    return vocab, comps


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


def normalize_tfidf(vocab, frequencies):
    N = len(frequencies)
    df = Counter()

    for freq in frequencies:
        df.update(freq.keys())

    features = []
    for freq in frequencies:
        tokens = freq.keys()
        feature_vector = np.zeros((len(vocab)/2))
        max_k = freq.most_common(1)[0][1]
        for token in tokens:
            tfidf = freq[token] / (max_k * math.log10(float(N) / df[token]))
            feature_vector[vocab[token]] = tfidf
        features.append(feature_vector)

    return np.array(features)


def combine_complements(kmer_counters, comps):
    new_kmer_counters = []
    for kmers in kmer_counters:
        new_counts = Counter()
        for kmer, v in kmers.items():
            comp = comps[kmer]
            if comp in new_counts:
                new_counts[comp] += v
            else:
                new_counts[kmer] = v
        new_kmer_counters.append(new_counts)

    return new_kmer_counters


#def gen_meta_features(data):


def featurize_data(file, k=3):
    """ 
    Featurize sequences and index labels
    Params:
        file....Delimited data file
    """
    data = pd.read_csv(file, names=["label", "dna"], usecols=[0,7], delimiter = '\t', header=0)
    labels = convert_labels(data.label)
    # labels = data.label
    start = time()
    kmers = [Counter(list(window(x.lower(), k))) for x in data.dna]
    print "Counted kmers for %d sequences in %d seconds" % (len(kmers), time()-start)
    vocab, comps = gen_vocab(k)
    start = time()
    print "Generated vocab for complements in %d seconds" % (time() - start)
    # comb_kmers = combine_complements(kmers, comps)

    # features = normalize_tfidf(vocab, comb_kmers)

    nrows = len(data.label)
    ncols = len(vocab)/2 if k % 2 == 1 else (len(vocab) + 2**k)/2
    row = []
    col = []
    csr_data = []
    print "Bulding feature matrix"
    for x in range(nrows):
        counts = kmers[x]
        cols = [vocab[kmer] for kmer in counts.keys()]
        csr_data.extend(counts.values())
        col.extend(cols)
        row.extend([x]*len(counts))
        '''
        for kmer in kmers[x].keys():
            features[x][vocab[kmer]] += kmers[x][kmer]'''

    print("Constructing sparse matrix")
    features = csr_matrix((csr_data, (row, col)), shape=(nrows, ncols))
    # normalize(features, copy=True)
    return labels, features, vocab


def save_sparse_csr(filename,array, labels, vocab):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape, labels=labels, vocab=vocab)


def main():
    print "Generating labels and features"
    sm_file = "data/ref.100ec.pgf.seqs.filter"
    lg_file = "data/rep.1000ec.pgf.seqs.filter"
    k = 3
    start = time()
    labels, features, vocab = featurize_data(sm_file, k)
    print "Time elapsed to build %d mers is %f" % (k, time()-start)
    #print "There are %d unique kmers" % len(features[0])
    print "Size of sparse matrix is %f (mbs)" % (float(sys.getsizeof(features))/1024**2)
    save_sparse_csr("data/feature_matrix." + str(k) + ".csr", features, labels, vocab)


if __name__ == '__main__':
    #os.chdir("/home/ngetty/examples/protein-pred")
    main()

