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


def build_attention_model(input_dim, nb_classes):
    inputs = Input(shape=(input_dim[0],))

    # ATTENTION PART STARTS HERE
    attention_probs = Dense(input_dim[0], activation='softmax', name='attention_vec')(inputs)
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
    data, labels = read_core()

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=0, stratify=labels)

    nb_classes = 1000
    input_shape = (data.shape[1], 1)
    print("Building model")
    model = build_attention_model(input_shape, nb_classes)

    batch_size = 80
    epochs = 20

    print("Training model")
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test), callbacks=[lr_reducer, early_stopper, csv_logger])


if __name__ == '__main__':
    main()