import sys
from collections import Counter

def main(data="sm"):
    file = "data/" + data + ".test/scans/all.scan"

    preds = []
    y_true = []
    top = []
    next = False

    with open(file, mode='rb') as scan:
        for line in scan:
            if len(line) > 10:
                if line[0] != '#':
                    vals = line.split()
                    top.append(vals[0])
                    if next:
                        next = False
                        y_true.append(vals[2])
                if line[2] == 't':
                    if top != []:
                        preds.append(top)
                    top = []
                    next = True
    preds.append(top)

    first_pred = [x[0] for x in preds]

    t5=0
    t1=0
    missed = []
    mistaken = []
    for x in range(len(y_true)):
        if y_true[x] == preds[x][0]:
            t1+=1
        else:
            mistaken.append(preds[x][0])
            missed.append(y_true[x])
        if y_true[x] in preds[x]:
            t5+=1

    missed = Counter(missed)
    mistaken = Counter(mistaken)

    print "Top 5 accuarcy:", float(t5)/len(y_true)
    print "Accuracy:", float(t1)/len(y_true)
    print "Missed counts", missed
    print "Classes with mistakes:", len(missed)
    print "Mistaken counts", mistaken


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0])
