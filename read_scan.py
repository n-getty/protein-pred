import numpy as np
from sklearn.metrics import accuracy_score

file = "data/coreseed.test/scans/all.scan"

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

first_pred = [x[0] for x in preds]
acc = accuracy_score(y_true, first_pred)
c=0
for x in range(len(y_true)):
    if y_true[x] in preds[x]:
        c+=1
print "Top 5 accuarcy:", float(c)/len(y_true)
print "Accuracy:", acc
np.savetxt("data/hmm_preds.csv", np.column_stack((preds,y_true)))