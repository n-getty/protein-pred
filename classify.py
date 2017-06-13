#/home/ngetty/dev/anaconda2/bin/python
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
from xgboost import XGBClassifier, DMatrix
from scipy.sparse import csr_matrix, hstack
import sys
from sklearn.decomposition import TruncatedSVD, MiniBatchSparsePCA
from memory_profiler import memory_usage


def cross_validation_accuracy(clf, X, labels, skf):
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
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf.fit(X_train, y_train)
        scores.append(accuracy_score(y_test, clf.predict(X_test)))
        train_scores.append(accuracy_score(y_train, clf.predict(X_train)))

    scores = [x for x in scores if str(x) != 'Nan']
    train_scores = [x for x in train_scores if str(x) != 'Nan']

    return np.mean(scores), np.mean(train_scores)


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
    clf.fit(X_train, y_train)
    if m != "XGBoost":
        t5, score = top_5_accuracy(clf.predict_proba(X_test), y_test)
    else:
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        t5 = "NA"
    if m != "Random Forest":
        train_pred = clf.predict(X_train)
        train_score = accuracy_score(y_train, train_pred)
    else:
        train_score = clf.oob_score_

    print "Top 5 accuracy:", t5
    #print "Top 1 accuracy:", score
    logging.info("Top 5 accuracy: %f", t5)
    #logging.info("Top 1 accuracy: %f", score)
    #cm = confusion_matrix(y_test, y_pred)

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

    #skf = list(StratifiedKFold(n_splits=folds, shuffle=True).split(features, labels))

    results = pd.DataFrame(columns=["Model", "CV Train Acc", "CV Val Acc", "Split Train Acc", "Split Val Acc", "Top 5 Train Acc", "Time"])

    for x in range(len(clfs)):
        start = time()
        mn = model_names[x]
        if mn == "XGBoost":
            features = DMatrix(features)

        print "Classiying with", mn
        logging.info("Classifying with %s", mn)

        clf = clfs[x]
        #cv_score, cv_train_score = cross_validation_accuracy(clf, features, labels, skf)
        cv_score = 0
        cv_train_score = 0
        #print "%s %d fold cross validation mean accuracy: %f" % (mn, folds, cv_score)
        #logging.info("%s %d fold cross validation mean accuracy: %f" % (mn, folds, cv_score))

        tts_score, tts_train_score, clf, t5 = test_train_split(clf, tts_split, mn)

        #if mn == "Random Forest":
            #print "test/train split accuracy:", top_5_accuracy(clf.predict_proba(),)
        if mn != "XGBoost":
            if mn == "Random Forest":
                feat_score = clf.feature_importances_
            else:
                feat_score = clf.coef_
            top_10_features = np.argsort(feat_score)[::-1][:10]
            print "Top ten feature idxs", top_10_features
            logging.info("Top ten feature idxs: %s", str(top_10_features))

        print "Training generalization accuracy:", tts_train_score
        print "Validation accuracy:", tts_score
        logging.info("Training generalization accuracy: %f", tts_train_score)
        logging.info("test/train split accuracy: %f", tts_score)
        #np.savetxt("results/" + mn + "_cm.txt", cm, fmt='%i', delimiter="\t")
        end = time()
        elapsed = end-start
        print "Time elapsed for model %s is %f" % (mn, elapsed)
        logging.info("Time elapsed for model %s is %f" % (mn, elapsed))
        results.loc[results.shape[0]] = ([mn, cv_train_score, cv_score, tts_train_score, tts_score, t5, elapsed])
        
    return results


def top_5_accuracy(probs, y_true):
    #print "Calculating top 5 accuracy"
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
                         shape=loader['shape']), loader['labels']


def main(file="feature_matrix.sm.3.csr.npz", file2="False", file3="False", red="False", tfidf="False"):
    folds = 5
    # SVC(probability=True),
    # LogisticRegression(solver="newton-cg", multi_class="multinomial", n_jobs=-1),
    clfs = [RandomForestClassifier(n_jobs=-1, n_estimators=200, oob_score=True), XGBClassifier(nthread=320, n_estimators=200)]
    model_names = ["Random Forest", "XGBoost"]
    features, labels = load_sparse_csr("data/" + file)
    labels = convert_labels(labels)

    #features = features[:, :-5]
    log_info = "Dimensionality reduction with 3,5 and 10mers"
    #log_info = "Testing tfidf transformation"
    print log_info
    logging.info(log_info)

    #tfer.fit(features)
    #features = tfer.transform(features)
    if file2 != "False":
        print "Combining kmer feature matrices"
        features2, _ = load_sparse_csr("data/" + file2)
        features2 = features2[:,:-5]
        #tfer.fit(features2)
        #features2 = tfer.transform(features2)
        #normalize(features2, copy=False)
        features = hstack([features, features2],format='csr')
    if file3 != "False":
        print "Combining kmer feature matrices with 3rd file"
        features3, _ = load_sparse_csr("data/" + file3)
        features3 = features3[:,:-5]
        #tfer.fit(features3)
        #features3 = tfer.transform(features3)
        #normalize(features3, copy=False)
        features = hstack([features, features3],format='csr', dtype="Float32")

    #tfer.fit(features[:, :-5])
    #tfer.transform(features[:, :-5], copy=False)
    if tfidf:
        tfer = TfidfTransformer()
        tfer.fit(features)
        features = tfer.transform(features)

    #normalize(features[:, :-5], copy=False)
    #normalize(features[:, -5:-1], copy=False)
    #normalize(features[-1], copy=False,axis=0)
    print features.shape

    if red != "False":
        print "Starting dimensionality reduction via TruncatedSVD"
        start = time()
        svd = TruncatedSVD(n_components=10000, n_iter=7, random_state=42)
        svd.fit(features)
        features = svd.transform(features)
        end = time()
        elapsed = end - start
        print "Time elapsed for dimensionality reduction is %f" %  elapsed
        logging.info("Time elapsed for dimensionality reduction is %f" %  elapsed)
    results = classify_all(labels, features, clfs, folds, model_names)
    #results.sort("Split Val Acc", inplace=True, ascending=False)
    results.to_csv("results/results.svd." + file, sep="\t")
    print results


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename="results/results.log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    if len(sys.argv) > 1:
        #os.chdir("/home/ngetty/examples/protein-pred")
        args = sys.argv[1:]
        mem_usage = memory_usage((main,args), interval=1.0)
        print('Average memory usage: %s' % np.mean(mem_usage))
        print('Maximum memory usage: %s' % max(mem_usage))
        np.savez("results/mem-usage" + file, mem_usage)

    else:
        main()
