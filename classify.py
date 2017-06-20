#/home/ngetty/dev/anaconda2/bin/python
import warnings
import os
import pandas as pd
import numpy as np
from time import time
import logging
import math
from itertools import islice, product
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
warnings.filterwarnings("ignore", category=DeprecationWarning)
from xgboost import XGBClassifier, DMatrix
from scipy.sparse import csr_matrix, hstack, issparse
import sys
from sklearn.decomposition import TruncatedSVD, MiniBatchSparsePCA
from memory_profiler import memory_usage
import operator
import xgboost as xgb
from lightgbm import LGBMClassifier


def cross_validation_accuracy(clf, X, labels, skf, m):
    """ 
    Compute the average testing accuracy over k folds of cross-validation. 
    Params:
        clf......A classifier.
        X........A matrix of features.
        labels...The true labels for each instance in X
        split........The fold indices
    Returns:
        The average testing accuracy of the classifier
        over each fold of cross-validation.
    """
    scores = []
    train_scores = []
    t5s = []
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if m == 'Random Forest':
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    early_stopping_rounds=2,
                    verbose=False,)
        t5, score = top_5_accuracy(clf.predict_proba(X_test), y_test)
        train_pred = clf.predict(X_train)
        train_score = accuracy_score(y_train, train_pred)
        scores.append(score)
        t5s.append(t5)
        train_scores.append(train_score)

    '''scores = [x for x in scores if str(x) != 'Nan']
    train_scores = [x for x in train_scores if str(x) != 'Nan']
    t5s = [x for x in t5s if str(x) != 'Nan']'''

    return np.mean(scores), np.mean(train_scores), np.mean(t5s)


def test_train_split(clf, split, m):
    """
    Compute the accuracy of a train/test split
    Params:
        clf......A classifier.
        split....indices
    Returns:
        The testing accuracy and the confusion
        matrix.
    """

    X_train, X_test, y_train, y_test = split
    if m == 'Random Forest':
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                early_stopping_rounds=2,
                verbose=False,)
    t5, score = top_5_accuracy(clf.predict_proba(X_test), y_test)
    train_pred = clf.predict(X_train)
    train_score = accuracy_score(y_train, train_pred)

    '''
    X_train = DMatrix(X_train, y_train)
    X_test = DMatrix(X_test, y_test)

    param = {'n_jobs': 8, 'n_estimators': 32, 'objective': "multi:softmax", "num_class": 100, "silent":1, "max_depth": 6, "eta": 0.3}
    clf = xgb.train(param, X_train, 10)
    preds = clf.predict(X_test)
    score = accuracy_score(y_test,preds)
    t5=0
    train_preds = clf.predict(X_train)
    train_score = accuracy_score(y_train, train_preds)'''

    #if m == "Random Forest":
        #train_score = clf.oob_score_

    print "Top 5 accuracy:", t5
    logging.info("Top 5 accuracy: %f", t5)

    return score, train_score, clf, t5


def classify_all(labels, features, clfs, folds, model_names):
    """ 
    Compute the average testing accuracy over k folds of cross-validation. 
    Params:
        labels.......The true labels for each instance in X
        features.....The feature vectors for each instance
        clfs.........The classifiers to fit and test
        folds........Number of folds for cross validation
        model_names..Readable names of each classifier
    """

    tts_split = train_test_split(
        features, labels, test_size=0.2, random_state=0, stratify=labels)

    skf = list(StratifiedKFold(n_splits=folds, shuffle=True).split(features, labels))

    results = pd.DataFrame(columns=["Model", "CV Train Acc", "CV Val Acc", "CV T5 Acc", "Split Train Acc", "Split Val Acc", "Top 5 Train Acc", "Max Mem", "Avg Mem", "Time"])

    for x in range(len(clfs)):
        start = time()
        mn = model_names[x]

        print "Classiying with", mn
        logging.info("Classifying with %s", mn)

        clf = clfs[x]

            #features = DMatrix(features)
        cv_score, cv_train_score, cv_t5 = cross_validation_accuracy(clf, features, labels, skf, mn)
        '''cv_score = -1
        cv_train_score = -1
        cv_t5 = -1'''
        print "%s %d fold cross validation mean train accuracy: %f" % (mn, folds, cv_train_score)
        logging.info("%s %d fold cross validation mean train accuracy: %f" % (mn, folds, cv_train_score))
        print "%s %d fold cross validation mean top 5 accuracy: %f" % (mn, folds, cv_t5)
        logging.info("%s %d fold cross validation mean top 5 accuracy: %f" % (mn, folds, cv_t5))
        print "%s %d fold cross validation mean validation accuracy: %f" % (mn, folds, cv_score)
        logging.info("%s %d fold cross validation mean validation accuracy: %f" % (mn, folds, cv_score))

        args = (clf, tts_split, mn)
        tts_score, tts_train_score, clf, t5 = test_train_split(*args)

        '''mem_usage, retval = memory_usage((test_train_split, args), interval=1.0, retval=True)
        tts_score, tts_train_score, clf, t5 = retval
        
        avg_mem = np.mean(mem_usage)
        max_mem = max(mem_usage)
        print('Average memory usage: %s' % avg_mem)
        print('Maximum memory usage: %s' % max_mem)'''
        avg_mem = -1
        max_mem = -1
        #np.savetxt("results/mem-usage/mem." + args[0], mem_usage, delimiter=',')

        #tts_score, tts_train_score, clf, t5 = test_train_split(clf, tts_split, mn)

        #if mn == "Random Forest":
            #print "test/train split accuracy:", top_5_accuracy(clf.predict_proba(),)
        if mn == "Random Forest" or mn == "LightGBM" or mn == "XGBoost":
            feat_score = clf.feature_importances_
            top_10_features = np.argsort(feat_score)[::-1][:10]
        elif mn == "XGBoost":
            print clf
            print clf.booster()
            fscore = clf.booster().get_fscore()
            fscore = sorted(fscore.items(), key=operator.itemgetter(1), reverse=True)
            top_features = [int(k[1:]) for k, _ in fscore]
            top_10_features = top_features[:10]
        else:
            feat_score = clf.coef_
            top_10_features = np.argsort(feat_score)[::-1][:10]

        print "Top ten feature idxs", top_10_features
        logging.info("Top ten feature idxs: %s", str(top_10_features))

        print "Training generalization accuracy:", tts_train_score
        print "Validation accuracy:", tts_score
        logging.info("Training generalization accuracy: %f", tts_train_score)
        logging.info("test/train split accuracy: %f", tts_score)
        end = time()
        elapsed = end-start
        print "Time elapsed for model %s is %f" % (mn, elapsed)
        logging.info("Time elapsed for model %s is %f" % (mn, elapsed))
        results.loc[results.shape[0]] = ([mn, cv_train_score, cv_score, cv_t5, tts_train_score, tts_score, t5, max_mem, avg_mem, elapsed])
        
    return results


def top_5_accuracy(probs, y_true):
    """ 
    Calculates top 5 and top 1 accuracy in 1 go
    Params:
        probs.....NxC matrix, class probabilities for each class
        y_true....True class labels
    Returns:
        top5 accuracy
        top1 accuracy
    """
    top5 = np.argsort(probs, axis=1)[:,-5:]
    c = 0
    top1c = 0
    for x in range(len(top5)):
        if np.in1d(y_true[x], top5[x], assume_unique=True)[0]:
            c += 1
        if y_true[x] == top5[x][4]:
            top1c += 1

    return float(c)/len(y_true), float(top1c)/len(y_true)


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


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype="float32"), loader['labels']


def load_data(size, file2, file3):

    path = "data/" + size + '/'

    if int(file2) * int(file3) == 1:
        print "Using 3, 5 and 10mer count features"
        features, labels = load_sparse_csr(path + "feature_matrix.3.5.10.csr.npz")
    else:
        features, labels = load_sparse_csr(path + "feature_matrix.3.csr.npz")
        if file2 != '0':
            print "Adding 5mer count features"
            features2, _ = load_sparse_csr(path + "feature_matrix.5.csr.npz")
            features2 = features2[:, :-5]
            features = hstack([features, features2], format='csr')

    labels = convert_labels(labels)

    return features, labels


def main(size='sm', file2='0', file3='0', red='0', tfidf='1', prune='0', est='32', thresh='0'):
    thresh = int(thresh)
    folds = 5

    # SVC(probability=True),
    # LogisticRegression(solver="newton-cg", multi_class="multinomial", n_jobs=-1),

    clfs = [RandomForestClassifier(n_jobs=-1,
                                   n_estimators=int(est),
                                   oob_score=False),
           XGBClassifier(n_jobs=-1,
                          n_estimators=int(est),
                          objective="multi:softprob",
                          max_depth=4,
                          learning_rate=0.3),

            LGBMClassifier(nthread=-1,
                           num_leaves=15,
                           learning_rate=0.3,
                           n_estimators=int(est))
            ]

    model_names = ["Random Forest",
                   "XGBoost",
                   "LightGBM"
             ]

    features, labels = load_data(size, file2, file3)

    # Zero-out counts below the given threshold
    if thresh > 0:
        v = np.sum(features.data <= thresh)
        print "Values less than threshhold,", v
        logging.info("Values less than threshhold,", v)
        features.data *= features.data > thresh

    # Remove feature columns that have sample below threshhold
    nonzero_counts = features.getnnz(0)
    nonz = nonzero_counts > int(prune)

    print "Removing %d features that do not have more than %s nonzero counts" % (
    features.shape[1] - np.sum(nonz), prune)
    logging.info(
        "Removing %d features that do not have more than %s nonzero counts" % (features.shape[1] - np.sum(nonz), prune))

    features = features[:, nonz]

    if tfidf != "0":
        print "Converting features to tfidf"
        logging.info("Converting features to tfidf")
        tfer = TfidfTransformer()
        tfer.fit(features[:,:32],labels)
        features_tf = tfer.transform(features[:,:32])
        #tfer.fit(features[:,31])
        #features[:,31] = tfer.transform(features[:,31])
        features = hstack([features_tf, features[:,32:]], format='csr')
        features = features.astype('float32')

    print "Final data shape:", features.shape
    logging.info("Final data shape: %s" % (features.shape,))

    # Reduce feature dimensionality
    if red != "0":
        print "Starting dimensionality reduction via TruncatedSVD"
        logging.info("Starting dimensionality reduction via TruncatedSVD")
        start = time()
        svd = TruncatedSVD(n_components=int(red), n_iter=5, random_state=42)
        svd.fit(features)
        features = svd.transform(features)
        end = time()
        elapsed = end - start
        print "Time elapsed for dimensionality reduction is %f" %  elapsed
        logging.info("Time elapsed for dimensionality reduction is %f" %  elapsed)

    #features = features.astype('float32')

    results = classify_all(labels, features, clfs, folds, model_names)
    #results.sort("Split Val Acc", inplace=True, ascending=False)
    results.to_csv("results/" + size + '.' + file2 + '.' + file3 + '.' + red + '.' + tfidf + '.' + prune + '.' + est, sep="\t")
    print results


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename="results/results.log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    if len(sys.argv) > 1:
        #os.chdir("/home/ngetty/examples/protein-pred")
        args = sys.argv[1:]
        main(*args)

    else:
        main()
