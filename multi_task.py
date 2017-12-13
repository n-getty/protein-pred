import warnings
import pandas as pd
import numpy as np
from time import time, gmtime, strftime
import logging
from scipy.sparse import csr_matrix, hstack, vstack
import argparse
from collections import Counter, defaultdict
import keras
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, merge
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from res50_nt import Res50NT


CHARS = ' atgc'
CHARLEN = len(CHARS)
SEED = 2017


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype="int32")


def seq_to_oh(data):
    min = 1000
    list_data = []
    print("Replacing seqs with char lists")
    for x in range(len(data)):
        list_data.append(list(data[x]))
        l = len(data[x])
        if l < min:
            min = l

    print min

    print("Slicing seqs")
    for x in range(len(data)):
        list_data[x] = list_data[x][:min]

    list_data = np.array(list_data)

    print("Transforming seqs to int")
    # transform to integer
    X_int = LabelEncoder().fit_transform(list_data.ravel()).reshape(*list_data.shape)
    print("Fitting seqs to onehot")
    print(X_int.shape)
    # transform to binary
    X_bin = OneHotEncoder().fit_transform(X_int).toarray()
    print np.array(X_bin).shape
    return np.array(X_bin)


def read_cafa():
    file = "data/cafa_df"
    data = pd.read_csv(file, header=0)
    labels = load_sparse_csr("data/cafa_labels.npz")

    return seq_to_oh(data), labels


def read_core():
    file = "data/coreseed.train.tsv"
    core_df = pd.read_csv(file, names=["label", "aa"], usecols=[1, 6], delimiter='\t', header=0)
    labels = core_df.label
    data = core_df.aa

    return seq_to_oh(data), to_categorical(labels)


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


def load_data_coreseed(maxlen=1000, val_split=0.2, batch_size=128, snake2d=False, seed=SEED):
    ctable = CharacterTable(CHARS, maxlen)

    df = pd.read_csv('coreseed.train.tsv', sep='\t', engine='c',
                     usecols=['function_index', 'dna'])

    n = df.shape[0]
    if snake2d:
        a = int(np.sqrt(maxlen))
        x = np.zeros((n, a, a, CHARLEN), dtype=np.byte)
    else:
        x = np.zeros((n, maxlen, CHARLEN), dtype=np.byte)

    for i, seq in enumerate(df['dna']):
        x[i] = ctable.encode(seq[:maxlen].lower(), snake2d=snake2d)

    y = pd.get_dummies(df.iloc[:, 0]).values
    classes = df.iloc[:, 0].nunique()

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                      random_state=seed,
                                                      stratify=df.iloc[:, 0])
    return (x_train, y_train), (x_val, y_val), classes


def build_attention_model(input_dim, nb_classes):
    inputs = Input(shape=(input_dim,))

    # ATTENTION PART STARTS HERE
    attention_probs = Dense(input_dim, activation='softmax', name='attention_vec')(inputs)
    attention_mul = merge([inputs, attention_probs], output_shape=nb_classes, name='attention_mul', mode='mul')
    # ATTENTION PART FINISHES HERE

    attention_mul = Dense(64)(attention_mul)
    output = Dense(units=nb_classes, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)

    return model


def main():
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger("results/multi_task.csv")

    print("Loading data")
    '''data, labels = read_core()
    print data.shape
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=0, stratify=labels)'''

    maxlen = 1000
    dense_layers = [0]
    dropout = .5
    activation = 'relu'
    model_variation = 'v1'

    (x_train, y_train), (x_test, y_test), classes = load_data_coreseed()

    print("Building model")
    #model = build_attention_model(data.shape[1], nb_classes)
    model = Res50NT(input_shape=(maxlen, CHARLEN),
                    dense_layers=dense_layers,
                    dropout=dropout,
                    activation=activation,
                    variation=model_variation,
                    classes=classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    batch_size = 80
    epochs = 20

    print("Training model")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test), callbacks=[lr_reducer, early_stopper, csv_logger])


if __name__ == '__main__':
    main()