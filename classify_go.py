import warnings
import pandas as pd
import numpy as np
from time import time, gmtime, strftime
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=DeprecationWarning)
from xgboost import XGBClassifier
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.decomposition import TruncatedSVD
from memory_profiler import memory_usage
from lightgbm import LGBMClassifier
import plot_cm as pcm
import argparse
from collections import Counter, defaultdict
from sklearn.multiclass import OneVsRestClassifier


def cross_validation_accuracy(clf, X, labels, skf, m):
    """
    Compute the average testing accuracy over k folds of cross-validation.
    Params:
        clf......A classifier.
        X........A matrix of features.
        labels...The true labels for each instance in X
        split........The fold indices
        m............The model name
    Returns:
        The average testing accuracy of the classifier
        over each fold of cross-validation.
    """
    scores = []
    train_scores = []
    i = 0
    for train_index, test_index in skf:
        print "Classifying on fold:", i
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if m == 'RandomForest' or m == 'Regression' or m == 'LightGBM':
            clf.fit(X_train, y_train)
        else:
            clf.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    early_stopping_rounds=2,
                    verbose=False,)
        probs = clf.predict_proba(X_test)
        train_probs = clf.predict_proba(X_train)
        train_score = fmax(train_probs,y_train)
        scores.append(fmax(probs,y_test))
        train_scores.append(train_score)

    return np.mean(scores), np.mean(train_scores)


def test_train_split(clf, split, m, class_names):
    """
    Compute the accuracy of a train/test split
    Params:
        clf......A classifier.
        split....indices
    Returns:
        The testing accuracy and the confusion
        matrix.
    """

    X_train, X_test, y_train, y_test, train_idx, test_idx = split

    if m == 'RandomForest' or m =='Regression' or m=='LightGBM':
        clf.fit(X_train, y_train)
    else:
        clf.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                early_stopping_rounds=2,
                verbose=False,)

    probs = clf.predict_proba(X_test)
    train_probs = clf.predict_proba(X_train)
    allp = np.vstack([probs, train_probs])
    idxs = test_idx + train_idx
    all_probs = [None] * len(idxs)
    for x in range(len(idxs)):
        all_probs[x] = allp[idxs[x]]
    all_probs = np.array(all_probs)
    np.savetxt('results/stored_probs.csv', all_probs, delimiter=',')

    score = fmax(probs, X_test)
    train_pred = clf.predict_proba(X_train)
    train_score = accuracy_score(y_train, train_pred)

    return score, train_score, clf


def classify_all(labels, features, clfs, folds, model_names, cv, mem):
    """
    Compute the average testing accuracy over k folds of cross-validation.
    Params:
        labels.......The true labels for each instance in X
        features.....The feature vectors for each instance
        clfs.........The classifiers to fit and test
        folds........Number of folds for cross validation
        model_names..Readable names of each classifier
        cv...........Whether to use cross validation
        mem..........Whether to store memory usage
    """

    tts_split = train_test_split(
        features, labels, range(labels.shape[0]), test_size=0.2, random_state=0)
    if cv:
        skf = list(KFold(n_splits=folds, shuffle=True).split(features, labels))

    results = pd.DataFrame(columns=["Model", "CV Train Acc", "CV Val Acc", "Split Train Acc", "Split Val Acc", "Max Mem", "Avg Mem", "Time"])

    for x in range(len(model_names)):
        start = time()
        mn = model_names[x]

        print "Classiying with", mn
        logging.info("Classifying with %s", mn)

        clf = OneVsRestClassifier(clfs[x], n_jobs=100)

        if cv:
            cv_score, cv_train_score = cross_validation_accuracy(clf, features, labels, skf, mn)
            print "%s %d fold cross validation mean train fscore: %f" % (mn, folds, cv_train_score)
            logging.info("%s %d fold cross validation mean train fscore: %f" % (mn, folds, cv_train_score))
            print "%s %d fold cross validation mean validation fscore: %f" % (mn, folds, cv_score)
            logging.info("%s %d fold cross validation mean validation fscore: %f" % (mn, folds, cv_score))
        else:
            cv_score = -1
            cv_train_score = -1

        args = (clf, tts_split, mn, labels)
        if mem:
            mem_usage, retval = memory_usage((test_train_split, args), interval=0.5, retval=True)
            tts_score, tts_train_score, clf, t5 = retval

            avg_mem = np.mean(mem_usage)
            max_mem = max(mem_usage)
            print('Average memory usage: %s' % avg_mem)
            print('Maximum memory usage: %s' % max_mem)
            np.savetxt("results/mem-usage/mem." + args[2], mem_usage, delimiter=',')
        else:
            tts_score, tts_train_score, clf, t5 = test_train_split(*args)
            avg_mem = -1
            max_mem = -1

        print "Training fscore:", tts_train_score
        print "Validation fscore", tts_score
        logging.info("Training fscofre: %f", tts_train_score)
        logging.info("test/train split fscore: %f", tts_score)
        end = time()
        elapsed = end-start
        print "Time elapsed for model %s is %f" % (mn, elapsed)
        logging.info("Time elapsed for model %s is %f" % (mn, elapsed))
        results.loc[results.shape[0]] = ([mn, cv_train_score, cv_score, tts_train_score, tts_score, max_mem, avg_mem, elapsed])

    return results


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype="float32")


def load_data(size, aa1, aa2, aa3, aa4):
    path = "data/" + size + '/'
    files = []
    if aa1:
        features = load_sparse_csr(path + "feature_matrix.aa1.csr.npz")
        files.append(features)
    if aa2:
        features = load_sparse_csr(path + "feature_matrix.aa2.csr.npz")
        files.append(features)
    if aa3:
        features = load_sparse_csr(path + "feature_matrix.aa3.csr.npz")
        files.append(features)
    if aa4:
        features = load_sparse_csr(path + "feature_matrix.aa4.csr.npz")
        files.append(features)

    labels = load_sparse_csr("data/cafa_labels.npz")

    if not files:
        print "No dataset provided"
        exit(0)

    features = hstack(files, format='csr')
    return features, labels


def get_parser():
    parser = argparse.ArgumentParser(description='Classify protein function with ensemble methods')
    parser.add_argument("--aa1", default=False, action='store_true', help="add 1mer aa features")
    parser.add_argument("--aa2", default=False, action='store_true', help="add 2mer aa features")
    parser.add_argument("--aa3", default=False, action='store_true', help="add 3mer aa features")
    parser.add_argument("--aa4", default=False, action='store_true', help="add 4mer aa features")
    parser.add_argument("--est", default=16, type=int, help="number of estimators (trees)")
    parser.add_argument("--cv", default=False, action='store_true', help="calculate cross validation results")
    parser.add_argument("--mem", default=False, action='store_true', help="store memory usage statistics")
    parser.add_argument("--rf", default=False, action='store_true', help="build random forest model")
    parser.add_argument("--xgb", default=False, action='store_true', help="build xgboost model")
    parser.add_argument("--lgbm", default=False, action='store_true', help="build lightgbm model")
    parser.add_argument("--gpu", default=False, action='store_true', help="use gpu for lgbm")
    parser.add_argument("--regr", default=False, action='store_true', help="build regression model")
    parser.add_argument("--thread",  default=-1, type=int, help="specify number of threads to to run with")
    parser.add_argument("--prune", default=0, type=int, help="remove features with apperance below prune")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    est = args.est
    prune = args.prune

    folds = 5

    if args.gpu:
        print "Using gpu enabled light gbm"
        device = 'gpu'
    else:
        device = 'cpu'

    all_clfs = [RandomForestClassifier(n_jobs=args.thread
                                   ,n_estimators=est
                                   #,oob_score=True
                                   #,max_depth=12
                                   ),

                XGBClassifier(n_jobs=args.thread,
                          n_estimators=est
                          ,objective="multi:softprob"
                          ,max_depth=2
                          ,learning_rate=0.5
                          ,colsample_bytree=0.8
                          #,subsample=0.5
                          #,max_bins=15
                          #,min_child_weight=6
                         ),

                LGBMClassifier(nthread=args.thread
                           ,max_depth=6
                           ,num_leaves=31
                           ,learning_rate=0.1
                           ,n_estimators=est
                           ,max_bin=15
                           ,colsample_bytree=0.8
                           ,device=device
                           #,subsample=0.8
                           #,min_child_weight=6
                           ),
                LogisticRegression(multi_class='multinomial', n_jobs=-1, solver='lbfgs', max_iter=est)
            ]
    model_names = []
    clfs = []
    if args.rf:
        model_names.append("RandomForest")
        clfs.append(all_clfs[0])
    if args.xgb:
        model_names.append("XGBoost")
        clfs.append(all_clfs[1])
    if args.lgbm:
        model_names.append("LightGBM")
        clfs.append(all_clfs[2])
    if args.regr:
        model_names.append("Regression")
        clfs.append(all_clfs[3])

    features, labels = load_data("cafa", args.aa1, args.aa2, args.aa3, args.aa4)

    print "Original data shape:", features.shape
    print labels.shape

    nonzero_counts = labels.getnnz(0)
    nonz = nonzero_counts > 1000

    print "Removing %d go terms that do not have more than %s nonzero counts" % (
        labels.shape[1] - np.sum(nonz), 1000)
    labels = labels[:, nonz]

    #labels = labels.todense()
    print labels.shape

    # Remove feature columns that have sample below threshhold
    nonzero_counts = features.getnnz(0)
    nonz = nonzero_counts > int(prune)

    print "Removing %d features that do not have more than %s nonzero counts" % (
    features.shape[1] - np.sum(nonz), prune)
    features = features[:, nonz]

    results = classify_all(labels, features, clfs, folds, model_names, args.cv, args.mem)
    for t in results.Time:
        print t,
    print
    print results.to_string()


def fmax(preds,true):
    max = 0
    for i in np.arange(0.01,1,0.01):
        pr, rc = precision_recall(preds, true, i)
        f = (2 * pr * rc)/(pr + rc)
        if f > max:
            max = f

    return max


def precision_recall(preds, true, thresh):
    print true.shape
    m = 0
    tps = []
    rcs = []
    for x in range(true.shape[0]):
        t = true[x].todense()
        pred = preds[x] > thresh
        if len(pred):
            tp = sum(np.logical_and(pred, t == 1))
            tps.append(tp/sum(pred))
            rcs.append(tp/sum(t))
            m += 1

    pr = 1/m * sum(tps)
    rc = 1/len(preds) * sum(rcs)

    return pr, rc


if __name__ == '__main__':
    main()