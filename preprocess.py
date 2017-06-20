#/home/ngetty/dev/anaconda2/bin/python
import pandas as pd
import math
from itertools import islice, product
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix, hstack
import sys
from time import time
from multiprocessing import Pool
import os


def window(seq, k=3):
    """ 
    Generate generate all kmers for a sequence
    Params:
        seq....dna sequence
        k......length of kmer
    Returns:
        Returns kmer generator
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
    Returns:
        A single kmer complement
    """
    comp = {'a': 't', 'c': 'g', 't': 'a', 'g': 'c'}
    rc = ()
    for x in range(len(kmer)):
        rc = rc + (comp[kmer[len(kmer)-x-1]],)
    return rc


def gen_comp_dict(all_kmers):
    """ 
    Generates mapping of kmers to their complements
    Params:
        all_kmers...All possible kmers for a given k
    Returns:
        Dictionary mapping kmer to their complement and vice-versa
    """
    comps = {}
    for kmer in all_kmers:
        comps[kmer] = reverse_complement(kmer)
    return comps


def gen_vocab(k=3):
    """
    Generate index kmer pairs for all possible kmers, binning complements together
    Params:
        k....length of kmer
    Returns:
        Dictionary mapping kmers and their complements to an index (column) in feature matrix
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
    Returns:
        Categorical label vector
    """
    label_idxs = {}
    new_labels = np.empty(len(labels))
    for x in range(len(labels)):
        new_labels[x] = label_idxs.setdefault(labels[x], len(label_idxs))
    return new_labels


def normalize_tfidf(vocab, frequencies):
    """ 
    Convert labels to indexes
    Params:
        vocab.........mapping of kmers to their feature vector index
        frequencies...kmer counts (complements combined)
    Returns:
        complete feature matrix
    """
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
    """ 
    Convert labels to indexes
    Params:
        kmer_counters...kmer counts for all sequences
        comps...........dict mapping kmers to their complements
    Returns:
        kmer counters with complements combined
    """
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

def work(seqnum_seq_k):
    """ 
    Function for workers to count kmers in parallel
    Params:
        seqnum_seq_k...Sequence number, dna sequence and k
    Returns:
        sequence number paired with that sequences kmer counts
    """
    seq = seqnum_seq_k[1]
    seqnum = seqnum_seq_k[0]
    k = seqnum_seq_k[2]
    kmers = window(seq.lower(), k)
    counts = Counter(kmers)

    return [seqnum, counts]


def get_kmer_counts(data, k):
    """ 
    Pools workers and resequences results to match original order
    Params:
        data...Dna sequences
        k......kmer length
    Returns:
        kmer counts for all sequences in dataset
    """
    data = zip(range(len(data)), data, [k]*len(data))
    pool = Pool(processes=100)

    res = [None] * len(data)

    for i, r in enumerate(pool.imap_unordered(work, data)):
        res[i] = r
        sys.stderr.write('\rdone {0:%}'.format(float(i+1) / len(data)))

    res = np.array(res)
    #print res.dtype
    indices = np.array(res[:,0],dtype=int)
    data = np.array(res[:,1])
    counts = data[indices]

    return counts


def featurize_data(data, k=3):
    """ 
    Featurize sequences and index labels
    Params:
        file....Delimited data file
    """

    # labels = convert_labels(data.label)
    # labels = data.label
    start = time()
    #kmers = [Counter(list(window(x.lower(), k))) for x in data.dna]
    kmers = get_kmer_counts(data, k)

    print "\nCounted kmers for %d sequences in %d seconds" % (len(kmers), time()-start)
    nrows = data.shape[0]

    vocab, _ = gen_vocab(k)

    ncols = len(vocab) / 2 if k % 2 == 1 else (len(vocab) + 2 ** k) / 2
    start = time()
    print "Generated vocab for complements in %d seconds" % (time() - start)
    # comb_kmers = combine_complements(kmers, comps)

    # features = normalize_tfidf(vocab, comb_kmers)
    nonzero_data = 0
    print "Counting nonzero data"
    for kmer in kmers:
        nonzero_data += len(kmer)



    indptr = np.zeros(nrows+1, dtype="int32")
    col = np.empty(nonzero_data, dtype="int32")
    csr_data = np.empty(nonzero_data, dtype="int8")
    print "Bulding feature matrix"
    #features = csr_matrix((nrows, ncols))
    data_counter = 0
    for x in range(nrows):
        sys.stderr.write('\rdone {0:%}'.format(float(x + 1) / nrows))
        counts = kmers[x]
        for k, v in counts.items():
            col[data_counter] = vocab[k]
            csr_data[data_counter] = v
            data_counter += 1
        indptr[x+1] = data_counter

    #print "Size of sparse data vector is %f (mbs)" % (float(sys.getsizeof(csr_data)) / 1024 ** 2)

    #print("Constructing sparse matrix")
    features = csr_matrix((csr_data, col, indptr), shape=(nrows, ncols))
    # normalize(features, copy=True)
    return features, vocab


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


def read_chunks(file,f,k,chunksize):
    c = 0
    path ="data/feature_matrix." + f + str(k) + "/"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    for data in pd.read_csv(file, chunksize=chunksize, names=["label", "dna"], usecols=[0, 7], delimiter='\t', header=0):
        labels = data.label
        features, vocab = featurize_data(data, k)
        # print "There are %d unique kmers" % len(features[0])
        print "\nSize of sparse matrix chunk %d is %f (mbs)" % (c, float(sys.getsizeof(features)) / 1024 ** 2)
        save_sparse_csr(path + "chunk." + str(c) + ".csr", features, labels, vocab)
        c += 1


def featurize_nuc_counts(data):
    nuc_counts = [Counter(x.lower()) for x in data]
    M = [[c['a'], c['c'], c['g'], c['t']] for c in nuc_counts]
    return csr_matrix(np.array(M))


def featurize_aa_counts(data):
    nuc_counts = [Counter(x) for x in data]
    M = [[c['F'], c['S'], c['Y'], c['C'], c['L'], c['I'], c['M'], c['V'], c['P'], c['T'], c['A'], c['H'], c['Q'], c['N'], c['K'], c['D'], c['E'], c['W'], c['R'], c['S'], c['G']] for c in nuc_counts]
    return csr_matrix(np.array(M))


def read_whole(file,f,k):
    data = pd.read_csv(file, names=["label", "aa", "dna", "aa_len"], usecols=[0, 6, 7, 8], delimiter='\t', header=0)
    labels = data.label

    features, vocab = featurize_data(data.dna, k)
    #aa_features, aa_vocab = featurize_data(data.aa, k)

    #nonz = features.getnnz(0) > 0
    #features = features[:, nonz]

    nuc_features = featurize_nuc_counts(data.dna)
    aa_counts = featurize_aa_counts(data.aa)
    aa_lens = pd.to_numeric(data.aa_len).reshape((len(labels),1))
    seq_lens = csr_matrix(np.array([len(seq) for seq in data.dna]).reshape((len(labels),1)))
    features = hstack([features, nuc_features, aa_counts, seq_lens, aa_lens], format='csr')

    #seq_lens = seq_lens.reshape((seq_lens.shape[0],1))
    #print "There are %d unique kmers" % len(features[0])
    print "\nSize of sparse matrix is %f (mbs)" % (float(sys.getsizeof(features))/1024**2)
    save_sparse_csr("data/" + f + "/feature_matrix." + str(k) + ".csr", features, labels, vocab)


def main(lg_file=False, k=3, chunksize=100000):
    start = time()
    k = int(k)
    chunksize = int(chunksize)

    print "Generating labels and features"

    if lg_file == "True":
        file = "data/rep.1000ec.pgf.seqs.filter"
        f = "lg"
        if chunksize > 0:
            read_chunks(file, f, k, chunksize)
        else:
            read_whole(file, f, k)
    else:
        file = "data/ref.100ec.pgf.seqs.filter"
        f = "sm"
        read_whole(file, f, k)

    print "Time elapsed to build %d mers is %f" % (k, time() - start)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        #os.chdir("/home/ngetty/examples/protein-pred")
        args = sys.argv[1:]
        main(args[0], args[1], args[2])
    else:
        main()
