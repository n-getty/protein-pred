
import pandas as pd


df = pd.DataFrame
file = "data/ref.100ec.pgf.seqs.filter"

data = pd.read_csv(file, names=["protein", "sequence"], usecols=[0, 7], delimiter = '\t')
data.protein = '>' + data.protein + '\n' + data.sequence

with open("data/ref.100ec.pgf.seqs.fasta", mode='wb') as file:
        for x in data.protein:
            print>>file, x
