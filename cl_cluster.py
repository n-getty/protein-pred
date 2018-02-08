import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def load_cell_data(fulldf=True):
    print "Reading cell line SD"
    sd_file = "GDSC-Bootstrap-N=10-Almanac/Cell-Line-Synergy.csv"
    sd_data = pd.read_csv(sd_file, delimiter=',', usecols=[0, 6], header=0)

    print "Reading feature file"
    if fulldf:
        feat_file = "GDSC-Bootstrap-N=10-Almanac/combined_rnaseq_data"
        feat_data = pd.read_csv(feat_file, delimiter='\t', header=0)
        print "Getting subset"
        feat_data = feat_data[(feat_data.Sample.str.startswith("GDSC")) | (feat_data.Sample.str.startswith("NCI60"))]

        print 'Saving data subset'
        feat_data.to_csv('GDSC-Bootstrap-N=10-Almanac/gdsc_nci60_data', delimiter='\t', index=0)
    else:
        feat_file = "GDSC-Bootstrap-N=10-Almanac/gdsc_nci60_data"
        feat_data = pd.read_csv(feat_file, delimiter=',', header=0)

    return sd_data, feat_data


def load_all_sets():
    feat_file = "GDSC-Bootstrap-N=10-Almanac/combined_rnaseq_data"
    #feat_file = "GDSC-Bootstrap-N=10-Almanac/cl_lincs1000"
    #feat_file = "GDSC-Bootstrap-N=10-Almanac/ccle_rnaseq_rpkm"

    #names = ['Sample']
    #names.extend([str(x) for x in range(942)])

    feat_data = pd.read_csv(feat_file, delimiter='\t'
                            #, names=names
                            #, header=None
                            , header=0
                            ,skiprows=range(1,11082)
                            #,usecols=range(100)
                            )

    #feat_data.rename(columns={"Name": "Sample"}, inplace=True)

    '''for x in range(len(feat_data.Sample)):
            feat_data.Sample[x] = "CCLE." + feat_data.Sample[x]'''

    #print "Getting subset"
    #print np.sum(feat_data.Sample.str.startswith("GDC"))
    #feat_data = feat_data[feat_data.Sample.str.startswith("CCLE")]

    unique_sets = set()

    feat_data.Sample.loc[30] = 'Scaled' + feat_data.Sample.loc[30]

    for s in feat_data['Sample']:
        se = s.split('.')[0]
        unique_sets.add(se)

    '''for s in unique_sets:
        sdf = feat_data[feat_data.Sample.str.startswith(s)]
        print("Mean for %s is %f" % (s, np.mean(sdf.mean(0))))
    exit(0)'''

    print unique_sets

    return feat_data, unique_sets


def transform_all(feat_data):
    samples = feat_data['Sample']
    feat_data = feat_data.drop('Sample', axis=1)

    feat_data.loc[:30] += 2.5

    model = TSNE(n_components=2, verbose=2, n_iter=1000)
    tsne_coords = model.fit_transform(feat_data)

    #pca = PCA(n_components=2, iterated_power=1000)
    #tsne_coords = pca.fit_transform(feat_data)

    df = pd.DataFrame({'x': tsne_coords[:, 0], 'y': tsne_coords[:, 1], 'Sample': samples})

    return df


def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def plot_all(df, unique_sets):
    fig, ax = plt.subplots()
    colors = ['r', 'b', 'g', 'y', 'k']
    c = 0
    scs = []
    #unique_sets.remove('CCLE')
    for s in unique_sets:
        sdf = df[df.Sample.str.startswith(s)]
        scs.append(ax.scatter(rand_jitter(sdf.x), rand_jitter(sdf.y), c=colors[c], marker='.', alpha=0.7))
        #scs.append(ax.scatter(sdf.x, sdf.y, c=colors[c], marker='.', alpha=0.5))
        c+=1

    ax.set_aspect("equal")
    plt.legend(scs, unique_sets)
    plt.show()


def plot(df, names, stdict):
    fig, ax = plt.subplots()

    sc = ax.scatter(df.x, df.y, c=df.z, cmap="cool")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.ax.set_ylabel('Normalized StDev', rotation=270)

    nci = df[names.str.startswith('N')]

    nc = ax.scatter(nci.x, nci.y, c='r', marker='.', alpha=0.7)
    ax.set_aspect("equal")

    plt.legend((sc, nc),  ('GDSC', 'NCI'))
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    pset = set()

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(" ".join([names[n] for n in ind["ind"]]))

        for n in ind["ind"]:
            if names[n] in stdict:
                s = stdict[names[n]]
            else:
                s = None
            pset.add((names[n], s))

        annot.set_text(text)

        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    print "Showing plot"
    plt.savefig('test.png')
    plt.show()
    for x in pset:
        print x


def transform_data(sd_data, feat_data):
    scaler = MinMaxScaler()
    sd_scaled = np.ravel(scaler.fit_transform(sd_data['Mean_SD'].reshape(-1, 1)))

    sd_dict = dict(zip(sd_data['CellLine'], sd_scaled))

    samples = feat_data['Sample']
    feat_data = feat_data.drop('Sample', axis=1)

    sd_col = []
    mcount = 0
    ncount = 0
    for s in samples:
        if s in sd_dict.keys():
            sd_col.append(sd_dict[s])
            mcount +=1
        else:
            '''if s[0] == 'N':
                ncount += 1
            else:
                print s'''
            sd_col.append(0)

    #print "Matches:", mcount
    #print "NCI samples:", ncount

    model = TSNE(n_components=2, verbose=2, n_iter=1000)
    tsne_coords = model.fit_transform(feat_data)

    df = pd.DataFrame({'x': tsne_coords[:,0], 'y': tsne_coords[:,1], 'z': sd_col})

    return df, samples, sd_dict


def tsne_all():
    print "Loading data"
    feat_data, sets = load_all_sets()

    print "Computing tsne"
    df = transform_all(feat_data)

    print "Plotting"
    plot_all(df, sets)
    #plot_subs(df)


def gdsc_sd():
    sd_data, feat_data = load_cell_data(False)

    print "Transforming with tsne"
    df, samples, sd_dict = transform_data(sd_data, feat_data)

    print "Plotting data"
    plot(df, samples, sd_dict)


def main():
    tsne_all()
    #gdsc_sd()


if __name__ == '__main__':
    main()