import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
from res50_nt import Res50NT
from sklearn.metrics import f1_score
import re
import math
import networkx as nx
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler

aa_chars = ' FSYCLIMVPTAHQNKDEWRGUXBZO'.lower()
aa_charlen = len(aa_chars)
CHARLEN = aa_charlen
SEED = 2017
MAXLEN = 256


class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None, snake2d=False):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        if snake2d:
            a = int(np.sqrt(maxlen))
            X2 = np.zeros((a, a, len(self.chars)))
            for i in range(a):
                for j in range(a):
                    k = i * a
                    k += a - j - 1 if i % 2 else j
                    X2[i, j] = X[k]
            X = X2
        return X

    def decode(self, X, snake2d=False):
        X = X.argmax(axis=-1)
        if snake2d:
            a = X.shape[0]
            X2 = np.zeros(a * a)
            for i in range(a):
                for j in range(a):
                    k = i * a
                    k += a - j - 1 if i % 2 else j
                    X2[k] = X[i, j]
            X = X2
        C = ''.join(self.indices_char[x] for x in X)
        return C


def load_data_cafa(maxlen=50, val_split=0.2, batch_size=128, snake2d=False, seed=SEED):
    ctable = CharacterTable(aa_chars.lower(), maxlen)

    cafa_df, labels, term_vocab = proc_cafa()

    df = cafa_df.aa

    n = len(df)

    if snake2d:
        a = int(np.sqrt(maxlen))
        x = np.zeros((n, a, a, aa_charlen), dtype=np.byte)
    else:
        x = np.zeros((n, maxlen, aa_charlen), dtype=np.byte)

    for i, seq in enumerate(df):
        #if len(seq) < maxlen:
            #seq += 'x' * (maxlen-len(seq)+1)
        x[i] = ctable.encode(seq[:maxlen].lower(), snake2d=snake2d)

    y = labels
    classes = labels.shape[1]

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                      random_state=seed)

    return (x_train, y_train), (x_val, y_val), classes, term_vocab


def simple_model(classes=100):
    model = Sequential(name='simple')
    model.add(Conv1D(200, 3, padding='valid', activation='relu', strides=1, input_shape=(MAXLEN, CHARLEN)))
    # model.add(Flatten())
    model.add(GlobalMaxPooling1D())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(classes, activation='sigmoid'))
    #model.add(Activation('sigmoid'))
    return model


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

    '''for G in Gs.values():
        print G.number_of_edges()
        print G.number_of_nodes()
        print [n for n,d in G.in_degree().items() if d==0]'''

    return Gs, alt_dict


def add_parents(Gs, terms, alt_dict):
    for G in Gs.values():
        ancs = []
        for term in terms:
            if G.has_node(term):
                a = G.predecessors(term)
                ancs.extend(a)
            elif term in alt_dict and G.has_node(alt_dict[term]):
                a = G.predecessors(alt_dict[term])
                ancs.extend(a)
        terms.extend(ancs)

    return terms


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

    dags, alts = construct_dag()

    l = len(term_vocab)
    for G in dags.values():
        for n in G.nodes():
            if n not in term_vocab:
                term_vocab[n] = len(term_vocab)

    print "added %d terms" % (len(term_vocab)-l)

    for k,v in seq_dict.items():
        X.append(v)
        label_vec = [0] * len(term_vocab)
        terms = term_dict[k]
        terms = add_parents(dags, terms, alts)
        for term in terms:
            label_vec[term_vocab[term]] = 1
        y.append(label_vec)

    #y = csr_matrix(y)
    y = np.array(y)
    X = np.array(X)

    cafa_df = pd.DataFrame({"aa":X})
    return cafa_df, y, term_vocab


def term_sens(counts, num_seqs):
    term_sens = {}
    sens_bins = defaultdict(list)
    for k, v in counts.items():
        sens = int(round(-1 * math.log(float(v) / num_seqs, 2)))
        term_sens[k] = sens
        sens_bins[sens].append(k)

    return term_sens, sens_bins


def term_probs():
    term_file = "data/uniprot_sprot_exp.txt"
    term_df = pd.read_csv(term_file, header=0, names=['id', 'term', 'category'], sep='\t')

    m_df = term_df[term_df.category == 'F']
    c_df = term_df[term_df.category == 'C']
    b_df = term_df[term_df.category == 'P']

    term_counts_m = Counter(m_df.term)
    term_counts_c = Counter(c_df.term)
    term_counts_b = Counter(b_df.term)

    term_sens_m, sens_bins_m = term_sens(term_counts_m, len(set(m_df.id)))

    term_sens_c, sens_bins_c = term_sens(term_counts_c, len(set(c_df.id)))

    term_sens_b, sens_bins_b = term_sens(term_counts_b, len(set(b_df.id)))

    return [term_sens_m, term_sens_c, term_sens_b], [sens_bins_m, sens_bins_c, sens_bins_b]


def sum_binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)


def main():
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger("results/multi_task.csv")

    (x_train, y_train), (x_test, y_test), classes, term_vocab = load_data_cafa(MAXLEN)

    '''nonzero_counts = y_train.getnnz(0)
    nonz = nonzero_counts > 50

    print "Removing %d go terms that do not have more than %s nonzero counts" % (
        y_train.shape[1] - np.sum(nonz), 50)
    y_train = y_train[:, nonz]
    y_test = y_test[:, nonz]'''


    term_sens, sens_bins = term_probs()
    #loss = 'binary_crossentropy'
    loss = sum_binary_crossentropy

    dense_layers = [768, 512]
    dropout = .8
    activation = 'relu'
    model_variation = 'v1'

    batch_size = 80
    epochs = 3

    for bins in sens_bins:
        for k,v in bins.items():
            classes = len(v)
            print "Training model with sensitivity of:", k
            print "Number of terms:", classes
            idxs = [term_vocab[term] for term in v]

            #model = simple_model(classes)

            model = Res50NT(input_shape=(MAXLEN, aa_charlen),
                            dense_layers=dense_layers,
                            dropout=dropout,
                            activation=activation,
                            variation=model_variation,
                            classes=classes, multi_label=True)

            model.compile(loss=loss,
                          optimizer='adam',
                          metrics=['accuracy'])

            y_train_sub = y_train[:,idxs]
            y_test_sub = y_test[:, idxs]
            '''for x in range(classes):
                examps = sum(y_train_sub[:,x])'''

            print("Training model")
            model.fit(x_train, y_train_sub,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test_sub), callbacks=[lr_reducer, early_stopper, csv_logger])

            #train_preds = model.predict(x_train)
            test_preds = model.predict(x_test)

            #x_train = np.hstack([x_train, train_preds])
            #x_test = np.hstack([x_test, test_preds])
            #print "Sum of train probs:", np.sum(train_preds)
            #print "Sum of test probs:", np.sum (test_preds)
            print "fmax:", (fmax(test_preds, y_test_sub))


def fmax(preds,true):
    print "Maximizing f score with prob threshhold"
    max = 0
    scaler = MinMaxScaler()
    preds = np.ravel(scaler.fit_transform(preds))
    for i in np.arange(0.1,1,0.1):
        preds[preds>i] = 1
        preds[preds<1] = 0
        f = f1_score(true, preds, average='weighted')
        if f>max:
            max=f
    return max


if __name__ == '__main__':
    main()