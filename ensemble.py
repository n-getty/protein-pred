from classify import load_data
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter


def main():
    X, y = load_data('sm', False, False, False, False, True, True, True, False)
    classes = list(set(y))
    cx_avg = []
    for c in classes:
        cx = X[y == c]
        cx_avg.append(np.average(cx))

    kmeans = KMeans(n_clusters=10, random_state=0).fit(cx_avg)
    
    print zip(classes, kmeans.labels_)
    print Counter(kmeans.labels_)


if __name__ == '__main__':
    main()