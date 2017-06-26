
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame
#file = "data/ref.100ec.pgf.seqs.filter"
file = "data/coreseed.train.tsv"

#data = pd.read_csv(file, names=["protein", "sequence"], usecols=[0, 7], delimiter = '\t')
data = pd.read_csv(file, names=["protein", "sequence"], usecols=[1, 5], delimiter='\t', header=0)
data.protein = data.protein.astype(str)
X_train, X_test, y_train, y_test = train_test_split(data.sequence, data.protein, test_size=0.2, random_state=0, stratify=data.protein)

train_df = pd.DataFrame({'X_train': X_train, 'y_train': y_train})

test_df = pd.DataFrame({'X_test': X_test, 'y_test': y_test})

train_seqs = '>' + train_df.y_train + '\n' + train_df.X_train
test_seqs = '>' + test_df.y_test + '\n' + test_df.X_test

train_df.train_fasta = train_seqs
test_df.test_seqs = test_seqs

train_df.to_csv("data/coreseed.train_df.csv", index=0)
test_df.to_csv("data/coreseed.test_df.csv", index=0)

with open("data/coreseed.train.fasta", mode='wb') as file:
        for x in train_seqs:
            print>>file, x

with open("data/coreseed.test.fasta", mode='wb') as file:
    for x in test_seqs:
        print>> file, x