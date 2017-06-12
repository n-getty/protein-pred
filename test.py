import numpy as np
import sys
from scipy.sparse import csr_matrix
from collections import Counter
from sklearn.preprocessing import normalize


x = [1,2,3,4]

y = 3

print np.in1d(y,x)[0]