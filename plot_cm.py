import itertools
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd


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

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    #thresh = cm.max() / 2.
    thresh = 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def class_statistics(y_true, y_pred, class_names):

    cnf_matrix = confusion_matrix(y_true, y_pred)
    sens = []
    spec = []
    most_conf_for = []
    most_conf_by = []

    for x in range(int(max(y_true))+1):
        row = cnf_matrix[x]
        col = cnf_matrix[:, x]
        tp = row[x]
        fp = sum(col)-tp
        fn = sum(row)-tp
        tn = len(y_true)-tp-fp
        sens.append(tp/tp+fn)
        spec.append(tn/tn+fp)
        sorted_row_idxs = np.argsort(row)
        sorted_col_idxs = np.argsort(col)
        mcf = sorted_row_idxs[-2] if sorted_row_idxs[-1] == x else sorted_row_idxs[-1]
        mcb = sorted_col_idxs[-2] if sorted_col_idxs[-1] == x else sorted_col_idxs[-1]
        most_conf_for.append(class_names[mcf])
        most_conf_by.append(class_names[mcb])

    stats_df = pd.DataFrame({'PGF': class_names, 'Sensitivity': sens, 'Specicifity': spec,
                             'Most FN': most_conf_for, 'Most FP': most_conf_by})
    stats_df.sort_values(by='Sensitivity', ascending=1, inplace=1,)
    return stats_df


def pcm(y_true, y_pred, mn):
    '''cm = ConfusionMatrix(y_true, y_pred)
    cm.plot(normalized=True)
    plt.tight_layout()
    plt.savefig("testcmfig")
    #cm.print_stats()
    exit(0)'''
    class_names=range(100)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)

    # Plot normalized confusion matrix
    plt.figure(figsize=(25,25))

    #plt.figure(figsize=(50, 50))
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                          title=mn + ' confusion matrix')
    plt.autoscale()
    #plt.savefig("results/plts" + mn)
    plt.savefig("testcmfig")
