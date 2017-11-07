import warnings
import pandas as pd
import numpy as np
from time import time, gmtime, strftime
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
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
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain


def classify_all(class_names, features, clfs, folds, model_names, cv, mem, save_feat):
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
    labels = convert_labels(class_names)
    class_names = unique_class_names(class_names)

    tts_split = train_test_split(
        features, labels, range(len(labels)), test_size=0.2, random_state=0, stratify=labels)
    if cv:
        skf = list(StratifiedKFold(n_splits=folds, shuffle=True).split(features, labels))

    results = pd.DataFrame(columns=["Model", "CV Train Acc", "CV Val Acc", "CV T5 Acc", "Split Train Acc", "Split Val Acc", "Top 5 Val Acc", "Max Mem", "Avg Mem", "Time"])

    for x in range(len(model_names)):
        start = time()
        mn = model_names[x]

        print "Classiying with", mn
        logging.info("Classifying with %s", mn)

        clf = clfs[x]

        if cv:
            cv_score, cv_train_score, cv_t5 = cross_validation_accuracy(clf, features, labels, skf, mn)
            print "%s %d fold cross validation mean train accuracy: %f" % (mn, folds, cv_train_score)
            logging.info("%s %d fold cross validation mean train accuracy: %f" % (mn, folds, cv_train_score))
            print "%s %d fold cross validation mean top 5 accuracy: %f" % (mn, folds, cv_t5)
            logging.info("%s %d fold cross validation mean top 5 accuracy: %f" % (mn, folds, cv_t5))
            print "%s %d fold cross validation mean validation accuracy: %f" % (mn, folds, cv_score)
            logging.info("%s %d fold cross validation mean validation accuracy: %f" % (mn, folds, cv_score))
        else:
            cv_score = -1
            cv_train_score = -1
            cv_t5 = -1

        args = (clf, tts_split, mn, class_names)
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

        if save_feat:
            feat_score = clf.feature_importances_
            sorted_feats = np.argsort(feat_score)[::-1]
            np.savetxt('results/' + mn + strftime("%m-%d_%H_%M", gmtime()) + '.sorted_features', np.vstack((sorted_feats,feat_score[sorted_feats])))
            np.savetxt('results/' + mn + strftime("%m-%d_%H_%M", gmtime()) + '.feat_scores', feat_score)
            top_10_features = sorted_feats[:10]
            print "Top ten feature idxs", top_10_features
            logging.info("Top ten feature idxs: %s", str(top_10_features))

        print "Training accuracy:", tts_train_score
        print "Validation accuracy:", tts_score
        logging.info("Training accuracy: %f", tts_train_score)
        logging.info("test/train split accuracy: %f", tts_score)
        end = time()
        elapsed = end-start
        print "Time elapsed for model %s is %f" % (mn, elapsed)
        logging.info("Time elapsed for model %s is %f" % (mn, elapsed))
        results.loc[results.shape[0]] = ([mn, cv_train_score, cv_score, cv_t5, tts_train_score, tts_score, t5, max_mem, avg_mem, elapsed])

    return results


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype="float32"), loader['labels']

def load_data(size, aa1, aa2, aa3, aa4):
    path = "data/" + size + '/'
    files = []
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
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    est = args.est
    thresh = args.thresh
    prune = args.prune

    folds = 5

    if args.gpu:
        print "Using gpu enabled light gbm"
        device = 'gpu'
    else:
        device = 'cpu'

    all_clfs = [OneVsRestClassifier(RandomForestClassifier(n_jobs=args.thread
                                   ,n_estimators=est
                                   #,oob_score=True
                                   #,max_depth=12
                                   )),

                OneVsRestClassifier(XGBClassifier(n_jobs=args.thread,
                          n_estimators=est
                          ,objective="multi:softprob"
                          ,max_depth=2
                          ,learning_rate=0.5
                          ,colsample_bytree=0.8
                          #,subsample=0.5
                          #,max_bins=15
                          #,min_child_weight=6
                         )),

                OneVsRestClassifier(LGBMClassifier(nthread=args.thread
                           ,max_depth=6
                           ,num_leaves=31
                           ,learning_rate=0.1
                           ,n_estimators=est
                           ,max_bin=15
                           ,colsample_bytree=0.8
                           ,device=device
                           #,subsample=0.8
                           #,min_child_weight=6
                           )),
                OneVsRestClassifier(LogisticRegression(multi_class='multinomial', n_jobs=-1, solver='lbfgs', max_iter=est))
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

    features, class_names = load_data(args.data, args.dna1, args.dna3, args.dna5, args.dna10, args.aa1, args.aa2, args.aa3, args.aa4)
    print "Original data shape:", features.shape

    if args.data == "cafa":
        print len(class_names)
        print "Removing insignificant go terms"
        term_count = Counter(class_names)
        print "Unique go terms:", len(term_count)
        idxs = [i > 100 for i in term_count.values()]
        sig_terms = set(np.array(term_count.keys())[idxs])
        print "Go terms with more than 100 seqs:", len(sig_terms)
        sig_rows = np.array([c in sig_terms for c in class_names])
        print "Seqs labeled with sig term:", sum(sig_rows)
        features = features[sig_rows, :]
        print features.shape
        class_names = class_names[sig_rows]
        print len(class_names)

    # Remove feature columns that have sample below threshhold
    nonzero_counts = features.getnnz(0)
    nonz = nonzero_counts > int(prune)

    print "Removing %d features that do not have more than %s nonzero counts" % (
    features.shape[1] - np.sum(nonz), prune)
    features = features[:, nonz]

    results = classify_all(class_names, features, clfs, folds, model_names, args.cv, args.mem, args.save_feat)
    for t in results.Time:
        print t,
    print
    print results.to_string()

if __name__ == '__main__':
    main()