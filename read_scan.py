import numpy as np


file = "data/coreseed.test/scans/1000.scan"

preds = []
y_true = []
top = []
next = False
with open(file, mode='rb') as scan:
    for line in scan:
        if len(line) > 10:
            if line[0] != '#':
                vals = line.split('\t')
                top.append(vals[0])
                if next:
                    next = False
                    y_true.append(vals[2])
            if line[2] == 't':
                preds.append(top)
                top = []
                next = True

np.savetxt("data/hmm_preds.csv", np.column_stack((preds,y_true)))