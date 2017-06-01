
import pandas as pd


df = pd.DataFrame
file = "data/ref.100ec.pgf.seqs.filter"

data = pd.read_csv(file, names=["protein", "sequence"], usecols=[0, 7], delimiter = '\t')
data.protein = '>' + data.protein + '\n' + data.sequence
for x in range(len(data.protein)):
    with open("data/ref.100.seqs/seq" + str(x) + ".fasta", mode='wb') as file:
            print>>file, data.protein[x]


'''
memory usage metric
larger dataset
fasta input
start with res networks on this data
add metafeatures
'''