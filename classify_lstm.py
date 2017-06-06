from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from scipy.sparse import csr_matrix, hstack
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from keras.utils import np_utils
import sys, os


def nn_batch_generator(X_data, y_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(y_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        y_batch = y_data[index_batch]
        counter += 1
        yield np.array(X_batch),y_batch
        if (counter > number_of_batches):
            counter=0


def classify(features, labels, use_batches, file):
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10)
    csv_logger = CSVLogger(file + '.lstm.log.csv')
    nb_classes = 100

    # Convolution
    kernel_size = 7
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 256

    # Training
    batch_size = 10000
    epochs = 200

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=0, stratify=labels)

    # The data, shuffled and split between train and test sets:
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print 'Build model...'

    model = Sequential()

    #model.add(Dropout(0.25, input_shape=(features.shape[1:])))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='same',
                     activation='relu',
                     strides=2, input_shape=(features.shape[1:])))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(units=nb_classes, kernel_initializer="he_normal"))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'top_k_categorical_accuracy'])

    print 'Train...'
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, Y_test), callbacks=[lr_reducer, early_stopper, csv_logger])
    #score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
    #print 'Test score:', score
    #print 'Test accuracy:', acc


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape']), loader['labels']


def main(file="feature_matrix.sm.3.csr.npz", file2="feature_matrix.sm.10.csr.npz"):
    use_batches = False

    features, labels = load_sparse_csr("data/" + file)

    if file2:
        features2, _ = load_sparse_csr("data/" + file2)
        features = hstack([features, features2])


    # input image dimensions
    #img_rows, img_cols = features.shape[1], features.shape[2]
    #img_channels = 1

    if not use_batches:
        features = features.toarray()
        normalize(features, copy=False)

    features = features.reshape(features.shape[0],features.shape[1], 1)

    #shape = (img_channels, img_rows, img_cols)
    classify(features, labels, use_batches, file)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        os.chdir("/home/ngetty/examples/protein-pred")
        args = sys.argv[1:]
        main(args[0], args[1])
    else:
        main()