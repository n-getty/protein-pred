
import numpy as np
import re
from collections import Counter, defaultdict
import networkx as nx
import pandas as pd


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
    G = nx.DiGraph()
    print "Loading GO data"
    file = "data/go-basic.obo"
    with open (file, 'r') as f:
        terms = f.read().split("[Term]")

    alt_dict = {}

    for term in terms:
        id = re.match("id: GO:\d+", term.strip()).group(0)[4:]
        alts = re.findall("alt_id: GO:\d+", term)
        if alts:
            for alt in alts:
                alt_dict[alt[8:]] = id

        G.add_node(id)
        is_as = re.findall("is_a: GO:\d+", term)
        G.add_edges_from([(x[6:], id) for x in is_as if x])
        part_ofs = re.findall("part_of GO:\d+", term)
        G.add_edges_from([(x[8:], id) for x in part_ofs if x])

    return G, alt_dict


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
    with open(term_file, 'r') as f:
        terms = f.readlines()
        for term in terms:
            term = term.split()
            seq_names.append(term[0])
            term_dict[term[0]].append(term[1])
            if term[1] not in term_vocab:
                term_vocab[term[1]] = len(term_vocab)

    for k,v in seq_dict.items():
        X.append(v)
        label_vec = [0] * len(term_vocab)
        for term in term_dict[k]:
            label_vec[term_vocab[term]] = 1
        y.append(label_vec)

    y = pd.Series(y)
    X = np.array(X)

    print y.shape
    print X.shape

    cafa_df = pd.DataFrame({"label":y, "aa":X})
    return cafa_df


def print_data_stats(data):
    go_terms = np.hstack(data[:, 1])
    counts = Counter(go_terms)
    print counts
    print len(counts)
    print len(data)


cafa_df = proc_cafa()
cafa_df.to_csv("data/cafa_df", index=0)

#process_raw_seqs()

#data = np.load("data/uniprot.npy")
#go_dag, alts = construct_dag()
#data = add_parents(go_dag, data, alts)
#print_data_stats(data)

