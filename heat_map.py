import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
#import seaborn as sns; sns.set(color_codes=True)


def unique_class_names(names):
    """ 
    Generate ordered unique class names
    Params:
        names....Label for every data point
    Returns:
        Name for each class in the set
    """
    cns = set()
    unique = []
    for c in names:
        if c not in cns:
            unique.append(c)
            cns.add(c)

    return unique

core=True
if core:
    print "Loading coreseed functions"
    file = "data/coreseed.train.tsv"
    labels = pd.read_csv(file, names=["peg", "function"], usecols=[0, 2], delimiter='\t', header=0)
else:
    print "Loading sm functions"
    file = "data/ref.100ec.pgf.seqs.filter"
    labels = pd.read_csv(file, names=["peg", "function"], usecols=[2, 5], delimiter='\t')

uni_funcs = np.array(unique_class_names(labels.function))
a = np.genfromtxt("results/stats/cnfmLightGBM2017-07-28 11:01.csv", delimiter=',', dtype="float32")

accs = []
for x in range(len(a)):
    accs.append( a[x][x]/np.sum(a[x]))

idxs = np.argsort(accs)[:100]
a = a[idxs][:,idxs]
normalize(a, copy=False)
#g = sns.clustermap(a, metric="correlation")
uni_funcs = uni_funcs[idxs]
plt.xticks(range(len(uni_funcs)), uni_funcs, rotation=90, fontsize=6)
plt.yticks(range(len(uni_funcs)), uni_funcs, fontsize=6)
plt.imshow(a, cmap='coolwarm', interpolation='nearest', aspect=1)
plt.show()

