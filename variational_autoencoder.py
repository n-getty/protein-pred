'''This script demonstrates how to build a variational autoencoder with Keras.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics
from scipy.sparse import csr_matrix, hstack
import os, sys


def encode(features):
    batch_size = 11890
    original_dim = 32
    latent_dim = 2
    intermediate_dim = 16
    epochs = 50
    epsilon_std = 1.0

    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim, activation='relu')(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x

    y = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, y)
    vae.compile(optimizer='rmsprop', loss=None)

    #X_train = features[:len(features)*.8]
    #X_test = features[len(features)*.8:]

    #X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
    #X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

    vae.fit(features,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2)

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    # display a 2D plot of the digit classes in the latent space
    features_encoded = encoder.predict(features)
    #x_train_encoded = encoder.predict(X_train, batch_size=batch_size)
    #x_test_encoded = encoder.predict(X_test, batch_size=batch_size)
    #print x_train_encoded.shape
    #print x_test_encoded.shape
    np.savez("data/encoded_features", features=features_encoded)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape']), loader['labels']


def main(file="feature_matrix.3.csr.npz", file2=False):
    use_batches = False

    features, _ = load_sparse_csr("data/" + file)
    features = features.toarray()

    if file2:
        features2, _ = load_sparse_csr("data/" + file)
        features = hstack(features, features2)

    #features = features.reshape(features.shape[0],features.shape[1], 1)

    encode(features)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        os.chdir("/home/ngetty/examples/protein-pred")
        args = sys.argv[1:]
        main(args[0])
    else:
        main()