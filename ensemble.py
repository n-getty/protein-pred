from classify import load_data
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def get_idxs(clusters, y):
    idxnum = np.array(range(len(y)))
    all_idx = []
    for x in clusters:
        if x[1] == '1':
            x[1] = '6'
    print Counter(clusters[:,1])
    for x in range(10):
        idxs =[]
        for l in clusters[:,0][clusters[:,1] == str(x)]:
            idxs.extend(idxnum[y==l])
        all_idx.append(idxs)

    return all_idx


def main():
    X, y = load_data('sm', False, False, False, False, True, True, True, False)

    classes = list(set(y))
    cx_avg = []

    for c in classes:
        cx = X[y == c]
        cx_avg.append(np.array(csr_matrix.mean(cx, axis=0)).flatten())

    cx_avg = np.array(cx_avg)

    kmeans = KMeans(n_clusters=10, random_state=0).fit(cx_avg)

    ccount = Counter(kmeans.labels_)

    model = LGBMClassifier(n_jobs=8
                           #,objective='ova'
                   #, max_depth=6
                   # ,num_leaves=31
                   , learning_rate=0.1
                   , n_estimators=100
                   #, max_bin=15
                   , colsample_bytree=0.8
                   , device="gpu"
                   #, verbose=-1
                    # ,gpu_platform_id=2
                    # ,gpu_device_id=2
                    # ,subsample=0.8
                    # ,min_child_weight=6
                    )

    tts_split = train_test_split(
        X, y, range(len(y)), test_size=0.2, random_state=0, stratify=y)

    X_train, X_test, y_train, y_test, train_idx, test_idx = tts_split

    clabels = kmeans.labels_

    idxs = get_idxs(np.asarray(zip(classes, clabels)), y_train)
    
    models = []
    for x in range(10):
        if x != 1:
            print("Number of classes:", ccount[x])
            model.fit(X_train[idxs[x]], y_train[idxs[x]],
                      #eval_set=[(X_train[idxs[x]], y_train[idxs[x]])],
                      #early_stopping_rounds=2,
                      verbose=False)

            train_score = model.score(X_train[idxs[x]], y_train[idxs[x]])
            #test_score = model.score(X_test, y_test)

            print "Train accuracy:", train_score
            #print "Test accuracy:", test_score
            models.append(model)
    
    train_probs = []
    test_probs = []

    '''models.append(model.fit(X_train, y_train,
                      #eval_set=[(X_train[idxs[x]], y_train[idxs[x]])],
                      #early_stopping_rounds=2,
                      verbose=False))'''

    for m in models:
        train_p = m.predict_proba(X_train)
        train_probs.append(train_p)

        test_p = m.predict_proba(X_test)
        test_probs.append(test_p)

    train_probs = np.hstack(train_probs)
    test_probs = np.hstack(test_probs)

    normalize(train_probs, copy=False, axis=1)

    model.fit(train_probs, y_train,
              #eval_set=[(X_train[idxs[x]], y_train[idxs[x]])],
              #early_stopping_rounds=2,
              verbose=False)

    print "Train accuracy:", model.score(train_probs, y_train)
    print "Test accuracy:", model.score(test_probs, y_test)

    normalize(test_probs, copy=False, axis=1)

    model.fit(train_probs, y_train,
              # eval_set=[(X_train[idxs[x]], y_train[idxs[x]])],
              # early_stopping_rounds=2,
              verbose=False)

    print "Train accuracy:", model.score(train_probs, y_train)
    print "Test accuracy:", model.score(test_probs, y_test)


if __name__ == '__main__':
    main()