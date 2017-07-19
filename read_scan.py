import sys
from collections import Counter


def main(data="coreseed", split="test"):
    file = "data/" + data + "." + split + "/scans/all.scan"

    preds = []
    y_true = []
    top = []
    next = False
    c = 0
    with open(file, mode='rb') as scan:
        for line in scan:
            #c+=1
            if len(line) > 10:
                if line[0] != '#':
                    vals = line.split()
                    top.append(vals[0])
                    if next:
                        next = False
                        y_true.append(vals[2])
                if line[2] == 't':
                    #c+=1
                    if top != []:
                        preds.append(top)
                    top = []
                    next = True
                elif next and line[2] != '-':
                    c+=1
    preds.append(top)

    first_pred = [x[0] for x in preds]
    print c
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

    print "Number of predictions", len(preds)
    print "Top 5 accuarcy:", float(t5)/(len(y_true)+c)
    print "Accuracy:", float(t1)/(len(y_true)+c)
    print "Missed counts", missed
    print "Classes with mistakes:", len(missed)
    print "Mistaken counts", mistaken


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 0:
        main(args[0], args[1])
    else:
        main()
