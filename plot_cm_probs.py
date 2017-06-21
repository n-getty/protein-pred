import itertools
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
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

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def pcm(y_test, y_probs, mn):
    # Compute confusion matrix
    #cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix = np.zeros((y_probs.shape[1],y_probs.shape[1]), dtype='float16')
    class_counts = Counter(y_test)
    class_counts = class_counts[range(np.max(y_test) + 1)]

    for x in range(len(y_test)):
        cnf_matrix[y_test[x]] += y_probs[x]
    cnf_matrix /= class_counts

    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=y_test, normalize=True,
                          title=mn + ' Probability confusion matrix')

    plt.savefig("results/plts/" + mn + '.prob')
    # plt.show()