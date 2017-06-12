#/home/ngetty/dev/anaconda2/bin/python
import os
import pandas as pd
import numpy as np
import time
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


def test_train_split(clf, split):
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
    y_pred = clf.predict(X_test)
    train_pred = clf.predict(X_train)
    score = accuracy_score(y_test, y_pred)
    train_score = accuracy_score(y_train, train_pred)
    cm = confusion_matrix(y_test, y_pred)

    return score, train_score, cm, clf


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
        features, labels, test_size=0.2, random_state=0)

    skf = list(StratifiedKFold(n_splits=folds, shuffle=True).split(features, labels))

    results = pd.DataFrame(columns=["Model", "CV Train Acc", "CV Val Acc", "Split Train Acc", "Split Val Acc", "Time", "top_10"])

    #top_10 = []

    for x in range(len(clfs)):
        start = time.time()
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

        tts_score, tts_train_score, cm, clf = test_train_split(clf, tts_split)

        if mn == "XGBoost":
            feat_score = clf.get_fscore
        else:
            feat_score = clf.feature_importances_

        top_10_features = np.argsort(feat_score)[::-1][:10]

        print "test/train split accuracy:", tts_score
        logging.info("test/train split accuracy: %f", tts_score)
        np.savetxt("results/" + mn + "_cm.txt", cm, fmt='%i', delimiter="\t")
        end = time.time()
        elapsed = end-start
        print "Time elapsed for model %s is %f" % (mn, elapsed)
        logging.info("Time elapsed for model %s is %f" % (mn, elapsed))
        results.loc[results.shape[0]] = ([mn, cv_train_score, cv_score, tts_train_score, tts_score, elapsed, top_10_features])
        
    return results


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape']), loader['labels']


def main(file="feature_matrix.sm.3.csr_2d.npy", file2="False", file3="False"):
    folds = 5
    #clfs = [XGBClassifier(), SVC(), GaussianNB(), MultinomialNB(), LogisticRegression(), RandomForestClassifier(n_jobs=-1), AdaBoostClassifier(n_estimators=10)]
    #model_names = ["XGBoost", "SVC", "Gaussian bayes", "Multinomial bayes", "Logistic Regression", "Random Forest", "AdaBoost"]
    clfs = [RandomForestClassifier(n_jobs=-1, n_estimators=50), XGBClassifier(nthread=320, n_estimators=50)]
    model_names = ["Random Forest", "XGBoost"]
    features, labels = load_sparse_csr("data/" + file)
    #features = features[:, :-5]
    #normalize(features[:, :-5], copy=False)
    #normalize(features[:, -5:-1], copy=False)
    #log_info = "Dimensionality reduction with 3,5 and 10mers"
    log_info = "Testing tfidf transformation"
    print log_info
    logging.info(log_info)
    features = TfidfTransformer.fit_transform(features)

    if file2 != "False":
        print "Combining kmer feature matrices"
        features2, _ = load_sparse_csr("data/" + file2)
        features2 = features2[:,:-5]
        features2 = TfidfTransformer.fit_transform(features2)
        normalize(features2, copy=False)
        features = hstack([features, features2])
    if file3 != "False":
        print "Combining kmer feature matrices with 3rd file"
        features3, _ = load_sparse_csr("data/" + file3)
        features3 = features3[:,:-5]
        features3 = TfidfTransformer.fit_transform(features3)
        #normalize(features3, copy=False)
        features = hstack([features, features3])

    print features.shape
    #normalize(features, copy=False,axis=0)


    #svd = TruncatedSVD(n_components=10000, n_iter=7, random_state=42)
    #svd.fit(features)
    #features = svd.transform(features)

    results = classify_all(labels, features, clfs, folds, model_names)
    results.sort("Split Val Acc", inplace=True, ascending=False)
    results.to_csv("results/results.svd." + file, sep="\t")
    print results


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename="results/results.log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    if len(sys.argv) > 1:
        #os.chdir("/home/ngetty/examples/protein-pred")
        args = sys.argv[1:]
        main(args[0], args[1], args[2])
    else:
        main()
