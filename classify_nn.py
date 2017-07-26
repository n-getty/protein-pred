from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Input, merge
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from scipy.sparse import csr_matrix, hstack
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.utils import np_utils
import sys, os
import resnet
import threading
import argparse
import densenet


def nn_batch_generator(X_data, y_data, batch_size, csr_2d, m):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        if csr_2d:
            X_batch = X_data[index_batch,:].toarray()
        else:
            X_batch = X_data[index_batch,:]
        y_batch = y_data[index_batch]
        if m == "lstm":
            X_batch = X_batch.reshape(X_batch.shape[0],X_batch.shape[1], 1)
        else:
            X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], 1, 1)
        counter += 1
        yield np.array(X_batch),y_batch
        if (counter > number_of_batches):
            counter=0


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


def build_lstm_model(nb_classes, input_shape):
    # Convolution
    kernel_size = 7
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 128
    model = Sequential()
    #model.add(Dropout(0.25, input_shape=(input_shape)))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='same',
                     activation='relu',
                     strides=2, input_shape=(input_shape)
                                 ))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(units=nb_classes, kernel_initializer="he_normal"))
    model.add(Activation('softmax'))

    return model


def classify(features, labels, use_batches, file, m, batch_size,
                                  depth,
                                  nb_dense_block,
                                  growth_rate,
                                  nb_filter):
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger("results/" + file + '.lstm.log.csv')

    if len(labels) > 100000:
        nb_classes = 1000
    else:
        nb_classes = 100

    # Training
    epochs = 200

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=0, stratify=labels)

    # The data, shuffled and split between train and test sets:
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    rows = 1

    input_shape = (features.shape[1], rows)

    vsteps = X_test.shape[0] // batch_size if X_test.shape[0] > batch_size else 1

    if m == "lstm":
        print 'Building LSTM model...'
        model = build_lstm_model(nb_classes, input_shape)
    elif m == "attn":
        model = build_attention_model(input_shape, nb_classes)
    elif m == "deep":
        model = densenet.DenseNet(nb_classes,
                                  (1, features.shape[0], features.shape[1]),
                                  depth,
                                  nb_dense_block,
                                  growth_rate,
                                  nb_filter)
    else:
        print 'Building RES model...'
        model = resnet.ResnetBuilder.build_resnet_101((1, features.shape[1], 1), nb_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])
    #opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    if not use_batches:
        print('Not using batches.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_test, Y_test), callbacks=[lr_reducer, early_stopper, csv_logger])
    else:
        print('Using batches.')
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(nn_batch_generator(X_train, Y_train, batch_size, False, m),
                            steps_per_epoch=X_train.shape[0] // batch_size,
                            validation_steps= vsteps,
                            validation_data=nn_batch_generator(X_test, Y_test, batch_size, False, m),
                            #workers=2,
                            epochs=epochs, verbose=1, max_q_size=100,
                            callbacks=[lr_reducer, early_stopper, csv_logger])


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape']), loader['labels']


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


def get_parser():
    parser = argparse.ArgumentParser(description='Classify protein function with nn methods')
    parser.add_argument("--data", default='sm', type=str,
                        help="data to use")
    parser.add_argument("--model", default='lstm', type=str,
                        help="data to use")
    parser.add_argument("--dna1", default=False, action='store_true',
                        help="add 1mer features")
    parser.add_argument("--dna3", default=False, action='store_true',
                        help="add 3mer features")
    parser.add_argument("--dna5", default=False, action='store_true',
                        help="add 5mer features")
    parser.add_argument("--dna10", default=False, action='store_true',
                        help="add 10mer features")
    parser.add_argument("--aa1", default=False, action='store_true',
                        help="add 1mer aa features")
    parser.add_argument("--aa2", default=False, action='store_true',
                        help="add 2mer aa features")
    parser.add_argument("--aa3", default=False, action='store_true',
                        help="add 3mer aa features")
    parser.add_argument("--aa4", default=False, action='store_true',
                        help="add 4mer aa features")
    parser.add_argument("--gen", default=False, action='store_true',
                        help="use batch generator")
    parser.add_argument("--prune", default=0, type=int,
                        help="remove features with apperance below prune")
    parser.add_argument("--thresh", default=0, type=int,
                        help="zero counts below threshold")
    parser.add_argument("--trunc", default=0, type=int,
                        help="Use only top k ")
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--nb_epoch', default=30, type=int,
                        help='Number of epochs')
    parser.add_argument('--depth', type=int, default=7,
                        help='Network depth')
    parser.add_argument('--nb_dense_block', type=int, default=1,
                        help='Number of dense blocks')
    parser.add_argument('--nb_filter', type=int, default=16,
                        help='Initial number of conv filters')
    parser.add_argument('--growth_rate', type=int, default=12,
                        help='Number of new filters added by conv layers')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=1E-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1E-4,
                        help='L2 regularization on weights')
    parser.add_argument('--plot_architecture', type=bool, default=False,
                        help='Save a plot of the network architecture')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    features, labels = load_data(args.data, args.dna1, args.dna3, args.dna5, args.dna10, args.aa1, args.aa2,
                                      args.aa3, args.aa4)
    print features.shape

    if args.trunc > 0:
        fimp = np.genfromtxt("results/LightGBM.sorted_features")
        idxs = fimp[0][:args.trunc]
        features = features[:,idxs]

    labels = convert_labels(labels)
    print features.shape

    features = features.toarray()
    #normalize(features, copy=False)
    if args.model == 'lstm':
        features = features.reshape(features.shape[0],features.shape[1], 1)
    elif args.model == 'attn':
        pass
    elif args.model == 'deep':
        #features = features.reshape(features.shape[0], features.shape[1], 1)
        features = features.reshape(None,1,features.shape[0], features.shape[1])
        #features = features.reshape(1, features.shape[0], features.shape[1])
    else:
        features = features.reshape(features.shape[0], features.shape[1], 1, 1)

    classify(features, labels, args.gen, args.data, args.model, args.batch_size,
                                  args.depth,
                                  args.nb_dense_block,
                                  args.growth_rate,
                                  args.nb_filter)


if __name__ == '__main__':
    lock = threading.Lock()
    main()