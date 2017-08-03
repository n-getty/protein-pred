import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
import seaborn as sns; sns.set(color_codes=True)


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
#np.fill_diagonal(a, 0)
normalize(a, copy=False)
uni_funcs = uni_funcs[idxs]
#network_pal = sns.cubehelix_palette(len(idxs),
                                    #light=.9, dark=.1, reverse=True,
                                    #start=1, rot=-2)
#network_lut = dict(zip(idxs, network_pal))

#network_colors = pd.Series(labels).map(network_lut)

g = sns.clustermap(a, method="median", metric="euclidean", center=0,

                  # Turn off the clustering
                  row_cluster=True, col_cluster=False,

                  # Add colored class labels
                  #row_colors=network_colors,

                  # Make the plot look better when many rows/cols
                  linewidths=0, xticklabels=uni_funcs, yticklabels=uni_funcs)

plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0, fontsize=6)
plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize=6)

#g.set_yticklabels(uni_funcs, rotation = 0, fontsize = 8)
#g.set_xticklabels(uni_funcs, rotation = 90, fontsize = 8)

#for x in idxs:
    #g.ax_col_dendrogram.bar(0, 0, color=network_lut[x],
                            #label=uni_funcs[x], linewidth=0)

#g.ax_col_dendrogram.legend(loc="center", ncol=2)

#g.cax.set_position([-.10, .2, .03, .45])
g.savefig("results/clustermaps/conf_worst2.png")


'''#g = sns.clustermap(a, metric="correlation")

plt.xticks(range(len(uni_funcs)), uni_funcs, rotation=90, fontsize=6)
plt.yticks(range(len(uni_funcs)), uni_funcs, fontsize=6)
plt.imshow(a, cmap='coolwarm', interpolation='nearest', aspect=1)
plt.show()'''

