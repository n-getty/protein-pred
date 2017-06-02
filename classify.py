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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix
import sys


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
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf.fit(X_train, y_train)
        scores.append(accuracy_score(y_test, clf.predict(X_test)))

    scores = [x for x in scores if str(x) != 'Nan']
    return np.mean(scores)


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
    score = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return score, cm


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

    results = pd.DataFrame(columns=["Model", "CV Score", "Test Score", "Time"])

    for x in range(len(clfs)):
        start = time.time()
        print "Classiying with", model_names[x]
        logging.info("Classifying with %s", model_names[x])

        clf = clfs[x]
        cv_score = cross_validation_accuracy(clf, features, labels, skf)

        print "%s %d fold cross validation mean accuracy: %f" % (model_names[x], folds, cv_score)
        logging.info("%s %d fold cross validation mean accuracy: %f" % (model_names[x], folds, cv_score))

        tts_score, cm = test_train_split(clf, tts_split)

        print "test/train split accuracy:", tts_score
        logging.info("test/train split accuracy: %f", tts_score)
        np.savetxt("results/" + model_names[x] + "_cm.txt", cm, fmt='%i', delimiter="\t")
        end = time.time()
        elapsed = end-start
        print "Time elapsed for model %s is %f" % (model_names[x], elapsed)
        logging.info("Time elapsed for model %s is %f" % (model_names[x], elapsed))
        results.loc[results.shape[0]] = ([model_names[x], cv_score, tts_score, elapsed])
    return results


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape']), loader['labels']


def main(folds=5, k=3):
    #clfs = [XGBClassifier(), SVC(), GaussianNB(), MultinomialNB(), LogisticRegression(), RandomForestClassifier(n_jobs=-1), AdaBoostClassifier(n_estimators=10)]
    #model_names = ["XGBoost", "SVC", "Gaussian bayes", "Multinomial bayes", "Logistic Regression", "Random Forest", "AdaBoost"]
    clfs = [RandomForestClassifier(n_jobs=-1, n_estimators=200)]
    model_names = ["Random Forest"]
    features, labels = load_sparse_csr("data/feature_matrix." + str(k) + ".csr.npz")
    features = features.toarray()
    results = classify_all(labels, features, clfs, folds, model_names)
    results.sort("CV Score", inplace=True, ascending=False)
    results.to_csv("results/results.csv", sep="\t")
    print results


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename="results/results.log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    if len(sys.argv) > 1:
        os.chdir("/home/ngetty/examples/protein-pred")
        args = sys.argv[1:]
        main(args[0], args[1])
    else:
        main()
