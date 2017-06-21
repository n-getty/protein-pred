import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #print(cm)
    cm = cm.astype('float')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def pcm(y_true, y_probs, mn):
    n_classes = int(np.max(y_true) + 1)
    # Compute confusion matrix
    cnf_matrix = np.zeros((n_classes, n_classes))
    class_counts = Counter(y_true)
    class_counts = [class_counts[x] for x in range(n_classes)]

    for x in range(len(y_true)):
        cnf_matrix[int(y_true[x])] += y_probs[x]

    cnf_matrix /= class_counts

    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure(figsize=(50,50))
    plot_confusion_matrix(cnf_matrix, classes=range(n_classes),
                          title=mn + ' Probability confusion matrix')

    plt.autoscale()
    plt.savefig("testcmfig")
    exit(0)
    # plt.show()