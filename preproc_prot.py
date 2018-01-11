from scipy.sparse import csr_matrix
import numpy as np
import re
from collections import Counter, defaultdict
import networkx as nx
import pandas as pd
import math

def process_raw_seqs():
    print "loading data"
    file = "data/uniprot_sprot.dat"
    with open (file, 'r') as f:
        data = f.read().split("//")

    print "matching go terms"
    go_terms = [re.findall("GO:\d+", x) for x in data if "SQ" in x and "GO" in x]
    print len(go_terms)

    print "matching seqs"
    seqs = ["".join(x.split("SQ")[1].split(";")[-1].split()) for x in data if "SQ" in x and "GO" in x]
    print len(seqs)

    print "Saving trimmed data"
    trim_data = np.column_stack([seqs,go_terms])
    np.save("data/uniprot", trim_data)


def construct_dag():
    Gs = {'m': nx.DiGraph(), 'c': nx.DiGraph(), 'b': nx.DiGraph()}
    print "Loading GO data"
    file = "data/go-basic.obo"
    with open (file, 'r') as f:
        terms = f.read().split("[Term]")

    alt_dict = {}

    for term in terms:
        obs = re.search("is_obsolete: true", term)
        if not obs:
            id = re.match("id: GO:\d+", term.strip()).group(0)[4:]
            alts = re.findall("alt_id: GO:\d+", term)
            category = re.search("namespace: \w", term).group(0)[-1]
            if alts:
                for alt in alts:
                    alt_dict[alt[8:]] = id

            Gs[category].add_node(id)
            is_as = re.findall("is_a: GO:\d+", term)
            Gs[category].add_edges_from([(x[6:], id) for x in is_as if x])
            part_ofs = re.findall("part_of GO:\d+", term)
            Gs[category].add_edges_from([(x[8:], id) for x in part_ofs if x])

    for G in Gs.values():
        print G.number_of_edges()
        print G.number_of_nodes()
        print [n for n,d in G.in_degree().items() if d==0]

    return Gs, alt_dict


def add_parents(G, data, alt_dict):
    for row in data:
        terms = row[1]
        ancs = []
        for term in terms:
            if G.has_node(term):
                a = G.predecessors(term)
            elif term in alt_dict:
                a = G.predecessors(alt_dict[term])
            ancs.extend(a)
        terms.extend(ancs)

    return data


def proc_cafa():
    seqs_file = "data/uniprot_sprot_exp.fasta"
    term_file = "data/uniprot_sprot_exp.txt"
    seq_dict = {}
    print "Reading seqs"
    with open(seqs_file, 'r') as f:
        seqs = f.read().split(">")
        for seq in seqs[1:]:
            seq = seq.split("\n", 1)
            seq[1] = seq[1].replace("\n", "")
            seq_dict[seq[0]] = seq[1]

    X = []
    y = []
    seq_names = []
    term_dict = defaultdict(list)
    term_vocab = {}
    all_terms = []
    print "Reading terms"
    with open(term_file, 'r') as f:
        terms = f.readlines()
        for term in terms:
            term = term.split()
            seq_names.append(term[0])
            term_dict[term[0]].append(term[1])
            all_terms.append(term[1])
            if term[1] not in term_vocab:
                term_vocab[term[1]] = len(term_vocab)

    for k,v in seq_dict.items():
        X.append(v)
        label_vec = [0] * len(term_vocab)
        for term in term_dict[k]:
            label_vec[term_vocab[term]] = 1
        y.append(label_vec)

    y = csr_matrix(y)
    X = np.array(X)

    cafa_df = pd.DataFrame({"aa":X})
    return cafa_df, y


def save_sparse_csr(filename, array):
    """
    Save csr matrix in loadable format
    Params:
        filename...save path
        array......csr matrix
        labels.....ordered true labels
        vocab......maps kmer to feature vector index
    """
    np.savez(filename,data = array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def print_data_stats(data):
    go_terms = np.hstack(data[:, 1])
    counts = Counter(go_terms)
    print counts
    print len(counts)
    print len(data)


def longest_paths():
    print ''


def term_probs():
    term_file = "data/uniprot_sprot_exp.txt"
    term_df = pd.read_csv(term_file, header=0, names=['id', 'term', 'category'], sep='\t')

    m_df = term_df[term_df.category == 'F']
    c_df = term_df[term_df.category == 'C']
    b_df = term_df[term_df.category == 'P']

    term_counts_m = Counter(m_df.term)
    term_counts_c = Counter(c_df.term)
    term_counts_b = Counter(b_df.term)

    num_seqs = len(set(term_df.id))

    term_sens_m = {}
    sens_bins_m = defaultdict(list())
    for k,v in term_counts_m.items():
        sens = int(round(-1 * math.log(float(v)/num_seqs,2)))
        term_sens_m[k] = sens
        sens_bins_m[sens].append(k)

    sens_counts_m = Counter(term_sens_m.values())

    term_sens_c = {}
    sens_bins_c = defaultdict(list())
    for k, v in term_counts_c.items():
        sens = int(round(-1 * math.log(float(v) / num_seqs, 2)))
        term_sens_c[k] = sens
        sens_bins_c[sens].append(k)

    sens_counts_c = Counter(term_sens_c.values())

    term_sens_b = {}
    sens_bins_b = defaultdict(list())
    for k, v in term_counts_b.items():
        sens = int(round(-1 * math.log(float(v) / num_seqs, 2)))
        term_sens_b[k] = sens
        sens_bins_b[sens].append(k)

    sens_counts_b = Counter(term_sens_b.values())

    return term_sens_m, term_sens_c, term_sens_b

#print "Saving data and labels"
#cafa_df, y = proc_cafa()
#save_sparse_csr("data/cafa_labels", y)
#cafa_df.to_csv("data/cafa_df", index=0)

#process_raw_seqs()

#data = np.load("data/uniprot.npy")
#dags, alts = construct_dag()

term_sens = term_probs()





#data = add_parents(go_dag, data, alts)
#print_data_stats(data)

