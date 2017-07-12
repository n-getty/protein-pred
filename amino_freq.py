#adapted from https://github.com/Prooffreader/word_list_tools

from collections import Counter
import pandas as pd
import sys
from multiprocessing import Pool
import os

data_path = 'data'
nb_path = 'letter_distributions'
save_filename = 'aa_dist_10'
file = "data/coreseed.train.tsv"
letters_pickle = "data/letters_10.p"


def work(wd):
    freq = freqs[wd[0]]
    wd = wd[1]
    p_len = len(wd)
    b_step = p_len
    bp_mult = b_len * p_len  # use multiple instead of range of 0 to 1 (or 0 to 100) to avoid floats not adding together exactly
    b_curnum = 0  # current bin
    p_curnum = 0  # current letter
    curmult = 0  # current position of algorithm from 0 to bp_mult
    temp = 0

    z = [0] * b_len
    letters = pd.DataFrame({'F': z, 'S': z, 'Y': z, 'C': z, 'L': z, 'I': z, 'M': z,
                            'V': z, 'P': z, 'T': z, 'A': z, 'H': z, 'Q': z, 'N': z,
                            'K': z, 'D': z, 'E': z, 'W': z, 'R': z, 'G': z})

    if p_len > 1:
        while curmult < bp_mult:
            temp += 1
            overlap = min((b_curnum + 1) * b_step, (p_curnum + 1) * p_step) - curmult
            # try:
            letters[wd[p_curnum]][b_curnum] += freq * overlap / bp_mult
            curmult += overlap
            if (b_curnum + 1) * b_step == curmult:
                b_curnum += 1
            if (p_curnum + 1) * p_step == curmult:
                p_curnum += 1

    return letters

b_len = 10  # number of bins, decided by user

if not os.path.isfile(letters_pickle):
    words = pd.read_csv(file, names=["label", "aa"], usecols=[1, 6], delimiter='\t', header=0)

    for x in range(len(words.aa)):
        if 'U' in words.aa[x]:
            words.aa[x] = words.aa[x].replace("U", "")
        if 'X' in words.aa[x]:
            words.aa[x] = words.aa[x].replace("X", "")

    freqs = Counter(words.label)

    print 'Calculating letters dataframe.'
    p_step = b_len  # to facilitate readability; cross product

    # dataframe for results; z is just a temporary list to facilitate dataframe initialization

    z = [0] * b_len
    letters = pd.DataFrame({'F': z, 'S': z, 'Y': z, 'C': z, 'L': z, 'I': z, 'M': z,
                            'V': z, 'P': z, 'T': z, 'A': z, 'H': z, 'Q': z, 'N': z,
                            'K': z, 'D': z, 'E': z, 'W': z, 'R': z, 'G': z})

    pool = Pool(processes=160)
    words = zip(words.label, words.aa)

    for i, l in enumerate(pool.imap_unordered(work, words)):
        sys.stderr.write('\rdone {0:%}'.format(float(i+1) / len(words)))
        letters+=l

    letters.to_pickle(letters_pickle)
else:
    letters = pd.read_pickle(letters_pickle)

colors = [[0, '#ffffcc'],
          [0.1, '#ffeda0'],
          [0.5, '#fed976'],
          [1, '#feb24c'],
          [2, '#fd8d3c'],
          [3, '#fc4e2a'],
          [5, '#e31a1c'],
          [9, '#b10026']]

alphabet = 'FSYCLIMVPTAHQNKDEWRG'

letters_norm = letters.copy()  # note that values are kept as integers for now; the graphs are narrow enough that it should not matter
letters_equal_area = letters.copy()
letters_overall = {}

letters_stats = pd.DataFrame({'max_freq': [0] * 20}, index=list(alphabet))
letters_stats['max_bin'] = 0
letters_stats['total_freq'] = 0
letters_stats['pct_freq'] = 0.0
letters_stats['norm_area'] = 0
letters_stats['color'] = ''

for ltr in alphabet:
    letters_stats.max_freq.ix[ltr] = letters[ltr].max()
    letters_stats.max_bin.ix[ltr] = letters[letters[ltr] == letters_stats.max_freq.ix[ltr]].index[0]
    letters_stats.total_freq.ix[ltr] = letters[ltr].sum()

letters_overall['max_freq'] = letters_stats.max_freq.max()
letters_overall['total_freq'] = letters_stats.total_freq.sum()
letters_overall['max_letter'] = letters_stats[letters_stats.max_freq == letters_overall['max_freq']].iloc[0].name

for ltr in alphabet:
    letters_stats.pct_freq.ix[ltr] = (letters_stats.total_freq.ix[ltr] * 100.0
                                      / letters_overall['total_freq'])
    for rw in range(len(letters_norm)):
        letters_norm[ltr].iloc[rw] *= 100
        letters_norm[ltr].iloc[rw] /= letters_stats['max_freq'].ix[ltr]

letters_overall['max_pct'] = letters_stats.pct_freq.max()

for ltr in alphabet:
    # assign colors based on pct_max and color list
    color = ''
    for i in range(len(colors)):
        if letters_stats.pct_freq.ix[ltr] >= colors[i][0]:
            color = colors[i][1]
    letters_stats.color.ix[ltr] = color
    # calculate area under norm lines
    area = 0
    for rw in range(len(letters_norm) - 1):
        height0 = letters_norm[ltr].iloc[rw]
        height1 = letters_norm[ltr].iloc[rw + 1]
        area += min(height0, height1)
        area += 0.5 * abs(height1 - height0)
    letters_stats.norm_area[ltr] = area

letters_overall['max_area'] = letters_stats.norm_area.max()
letters_overall['max_equal_area'] = 0

for ltr in alphabet:
    for rw in range(len(letters_equal_area)):
        letters_equal_area[ltr].iloc[rw] = (letters_norm[ltr].iloc[rw] *
                                            letters_overall['max_area'] / letters_stats.norm_area[ltr])
    letters_overall['max_equal_area'] = max(letters_overall['max_equal_area'], letters_equal_area[ltr].max())

# rescale to 100
for ltr in alphabet:
    for rw in range(len(letters_equal_area)):
        letters_equal_area[ltr].iloc[rw] *= 100
        letters_equal_area[ltr].iloc[rw] /= letters_overall['max_equal_area']

import math

letters_overall['max_pct_for_legend'] = int(math.ceil(letters_overall['max_pct']))

letters_overall['max_compromise'] = 0
letters_compromise = letters_norm.copy()
for ltr in alphabet:
    for rw in range(len(letters_equal_area)):
        letters_compromise[ltr].iloc[rw] = (letters_norm[ltr].iloc[rw] + letters_equal_area[ltr].iloc[rw]) / 2
    letters_overall['max_compromise'] = max(letters_overall['max_compromise'], letters_compromise[ltr].max())

# rescale to 100
for ltr in alphabet:
    for rw in range(len(letters_equal_area)):
        letters_compromise[ltr].iloc[rw] *= 100
        letters_compromise[ltr].iloc[rw] /= letters_overall['max_compromise']


save_plot = True

column_list = list(alphabet)
x_length = b_len


import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(20, 1, figsize=(12, 90))
#plt.title("Title", size=18, color='k')

for pos in range(len(column_list)):
    ltr = column_list[pos]
    axes[pos].plot(range(x_length), letters_compromise[ltr], color='k', linewidth = 3, label = ltr)
    axes[pos].set_ylim(0,100)
    fill_color = letters_stats['color'].ix[ltr]
    axes[pos].fill_between(range(x_length), letters_compromise[ltr], color=fill_color, interpolate=True)
    axes[pos].set_xticks([])
    axes[pos].set_yticks([])
    axes[pos].set_xticklabels([], size=0)
    axes[pos].set_yticklabels([])
    axes[pos].get_xaxis().set_visible(False)
    axes[pos].set_ylabel(ltr+'       ', size=24, rotation='horizontal')
    plt.subplots_adjust(hspace=0.1)

if save_filename != '':
    plot_name = nb_path + '/' + save_filename + '.png'

if save_plot == True:
    plt.savefig(plot_name)
