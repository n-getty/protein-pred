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


def reverse_complement(kmer, comp):
    """
    Generate a kmers complement
    Params:
        kmer....The kmer to generate the complement for
    Returns:
        A single kmer complement
    """
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
    comp = {'a': 't', 'c': 'g', 't': 'a', 'g': 'c'}
    for kmer in all_kmers:
        comps[kmer] = reverse_complement(kmer, comp)
    return comps


def gen_vocab(k=3, mode='dna'):
    """
    Generate index kmer pairs for all possible kmers, binning complements together
    Params:
        k....length of kmer
    Returns:
        Dictionary mapping kmers and their complements to an index (column) in feature matrix
    """
    if mode == 'dna':
        all_kmers = list(product('acgt', repeat=k))
        comps = gen_comp_dict(all_kmers)
    else:
        all_kmers = list(product('FSYCLIMVPTAHQNKDEWRG'.lower(), repeat=k))

    vocab = {}
    unique = 0

    for mer in all_kmers:
        if mode == 'dna':
            rc = comps[mer]
            if rc in vocab:
                vocab[mer] = vocab[rc]
            else:
                vocab[mer] = unique
                unique += 1
        else:
            vocab[mer] = unique
            unique += 1
    return vocab


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


def get_kmer_counts(data, k, p):
    """ 
    Pools workers and resequences results to match original order
    Params:
        data...Dna sequences
        k......kmer length
    Returns:
        kmer counts for all sequences in dataset
    """
    data = zip(range(len(data)), data, [k]*len(data))
    pool = Pool(processes=p)

    res = [None] * len(data)

    for i, r in enumerate(pool.imap_unordered(work, data)):
        res[i] = r
        sys.stderr.write('\rdone {0:%}'.format(float(i+1) / len(data)))

    res = np.array(res)
    indices = np.array(res[:,0],dtype=int)
    data = np.array(res[:,1])
    counts = data[indices]

    return counts


def featurize_data(data, k=3, mode='dna', p=8):
    """ 
    Featurize sequences and index labels
    Params:
        file....Delimited data file
    """

    # labels = convert_labels(data.label)
    # labels = data.label
    start = time()
    #kmers = [Counter(list(window(x.lower(), k))) for x in data.dna]
    kmers = get_kmer_counts(data, k, p)

    #print "\nCounted kmers for %d sequences in %d seconds" % (len(kmers), time()-start)
    nrows = data.shape[0]

    if mode == 'dna':
        vocab = gen_vocab(k)
        ncols = len(vocab) / 2 if k % 2 == 1 else (len(vocab) + 2 ** k) / 2
    else:
        vocab = gen_vocab(k, 'aa')
        ncols = len(vocab)

    start = time()
    #print "Generated vocab for complements in %d seconds" % (time() - start)
    # comb_kmers = combine_complements(kmers, comps)

    # features = normalize_tfidf(vocab, comb_kmers)
    nonzero_data = 0
    #print "Counting nonzero data"
    for kmer in kmers:
        nonzero_data += len(kmer)

    indptr = np.zeros(nrows+1, dtype="int32")
    col = np.empty(nonzero_data, dtype="int32")
    csr_data = np.empty(nonzero_data, dtype="int8")
    #print "Bulding feature matrix"
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

    ##print "Size of sparse data vector is %f (mbs)" % (float(sys.getsizeof(csr_data)) / 1024 ** 2)

    ##print("Constructing sparse matrix")
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
        # #print "There are %d unique kmers" % len(features[0])
        #print "\nSize of sparse matrix chunk %d is %f (mbs)" % (c, float(sys.getsizeof(features)) / 1024 ** 2)
        save_sparse_csr(path + "chunk." + str(c) + ".csr", features, labels, vocab)
        c += 1


def featurize_nuc_counts(data):
    nuc_counts = [Counter(x.lower()) for x in data]
    M = [[c['a'], c['c'], c['g'], c['t']] for c in nuc_counts]
    return csr_matrix(np.array(M))


def featurize_aa_counts(data):
    nuc_counts = [Counter(x) for x in data]
    M = [[c['F'], c['S'], c['Y'], c['C'], c['L'], c['I'], c['M'], c['V'], c['P'], c['T'], c['A'], c['H'], c['Q'], c['N'], c['K'], c['D'], c['E'], c['W'], c['R'], c['G']] for c in nuc_counts]
    return csr_matrix(np.array(M))


def read_cafa(file):
    #print "Reading cafa dataframe"
    f = "cafa"
    data = pd.read_csv(file,  header=0)
    #print "Removing unknown proteins"
    for x in range(len(data.aa)):
        if 'U' in data.aa[x]:
            data.aa[x] = data.aa[x].replace("U", "")
        if 'X' in data.aa[x]:
            data.aa[x] = data.aa[x].replace("X", "")
        if 'B' in data.aa[x]:
            data.aa[x] = data.aa[x].replace("B", "")
        if 'Z' in data.aa[x]:
            data.aa[x] = data.aa[x].replace("Z", "")
        if 'O' in data.aa[x]:
            data.aa[x] = data.aa[x].replace("O", "")

    #print "generating aa 2mer features"
    aa_features, aa_vocab = featurize_data(data.aa, 2, 'aa')
    #print "generating aa 3mer features"
    aa_features3, aa_vocab3 = featurize_data(data.aa, 3, 'aa')

    aa_features4, aa_vocab4 = featurize_data(data.aa, 4, 'aa')
    aa_counts = featurize_aa_counts(data.aa)
    aa_lens = csr_matrix(np.array([len(seq) for seq in data.aa]).reshape((len(data.aa), 1)))
    aa_counts = hstack([aa_counts, aa_lens], format='csr')
    if not os.path.exists("data/cafa"):
        os.makedirs("data/cafa")

    save_sparse_csr("data/" + f + "/feature_matrix.aa1.csr", aa_counts, [], aa_vocab)
    save_sparse_csr("data/" + f + "/feature_matrix.aa2.csr", aa_features, [], aa_vocab)
    save_sparse_csr("data/" + f + "/feature_matrix.aa3.csr", aa_features3, [], aa_vocab3)
    save_sparse_csr("data/" + f + "/feature_matrix.aa4.csr", aa_features4, [], aa_vocab4)


def read_whole(file,f,k,p):
    if f == 'core':
        data = pd.read_csv(file, names=["label", "dna", "aa"], usecols=[1, 5, 6], delimiter='\t', header=0)
        for x in range(len(data.aa)):
            if 'U' in data.aa[x]:
                data.aa[x] = data.aa[x].replace("U", "")
            if 'X' in data.aa[x]:
                data.aa[x] = data.aa[x].replace("X", "")

        #print "Removed all U and X aas"
    else:
        data = pd.read_csv(file, names=["label", "aa", "dna"], usecols=[0, 6, 7], delimiter='\t', header=0)
    labels = data.label

    features3, vocab = featurize_data(data.dna, 3, p)
    features5, vocab = featurize_data(data.dna, 5, p)
    features10, vocab = featurize_data(data.dna, 10, p)
    #print "generating aa 2mer features"
    aa_features, aa_vocab = featurize_data(data.aa, 2, 'aa', p)
    #print "generating aa 3mer features"
    aa_features3, aa_vocab3 = featurize_data(data.aa, 3, 'aa', p)

    aa_features4, aa_vocab3 = featurize_data(data.aa, 4, 'aa', p)

    #aa_features, aa_vocab = featurize_data(data.aa, k)

    #nonz = features.getnnz(0) > 0
    #features = features[:, nonz]

    nuc_features = featurize_nuc_counts(data.dna)
    aa_counts = featurize_aa_counts(data.aa)
    aa_lens = csr_matrix(np.array([len(seq) for seq in data.aa]).reshape((len(labels), 1)))
    seq_lens = csr_matrix(np.array([len(seq) for seq in data.dna]).reshape((len(labels),1)))

    #features = hstack([features, seq_lens], format='csr')

    aa_counts = hstack([aa_counts, aa_lens], format='csr')
    save_sparse_csr("data/" + f + "/feature_matrix.aa1.csr", aa_counts, labels, vocab)
    save_sparse_csr("data/" + f + "/feature_matrix.aa2.csr", aa_features, labels, vocab)
    save_sparse_csr("data/" + f + "/feature_matrix.aa3.csr", aa_features3, labels, vocab)
    save_sparse_csr("data/" + f + "/feature_matrix.aa4.csr", aa_features4, labels, vocab)

    ##print features.shape
    #seq_lens = seq_lens.reshape((seq_lens.shape[0],1))
    ##print "There are %d unique kmers" % len(features[0])
    nuc_features = hstack([nuc_features,seq_lens], format='csr')
    ##print "\nSize of sparse matrix is %f (mbs)" % (float(sys.getsizeof(features))/1024**2)
    save_sparse_csr("data/" + f + "/feature_matrix.1.csr", nuc_features, labels, vocab)
    save_sparse_csr("data/" + f + "/feature_matrix.3.csr", features3, labels, vocab)
    save_sparse_csr("data/" + f + "/feature_matrix.5.csr", features5, labels, vocab)
    save_sparse_csr("data/" + f + "/feature_matrix.10.csr", features10, labels, vocab)


def main(fn='sm', k=3, chunksize=0, p=8):
    start = time()
    k = int(k)
    chunksize = int(chunksize)

    #print "Generating labels and features"

    if fn == "lg":
        file = "data/rep.1000ec.pgf.seqs.filter"
        if chunksize > 0:
            read_chunks(file, fn, k, chunksize)
        else:
            read_whole(file, fn, k, p)
    elif fn == "core":
        file = "data/coreseed.train.tsv"
        if chunksize > 0:
            read_chunks(file, fn, k, chunksize)
        else:
            read_whole(file, fn, k, p)
    elif fn =="cafa":
        file = "data/cafa_df"
        read_cafa(file)
    else:
        file = "data/ref.100ec.pgf.seqs.filter"
        read_whole(file, fn, k, p)

    #print "Time elapsed to build %d mers is %f" % (k, time() - start)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        #os.chdir("/home/ngetty/examples/protein-pred")
        args = sys.argv[1:]
        main(args[0], args[1], args[2], args[3])
    else:
        main()
