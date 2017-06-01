from multiprocessing import Pool, TimeoutError
import time
import os
import numpy as np
import sys


x = np.array([4,3,2,1,0])
y = np.array([4,3,2,1,0])

print x[y]


x = np.array([[1,2],[3,4]])

print x[:,1]


x = [1,2,3,4]
y = [5,6,7,8]
z = [1,2,3,4]
print zip(x,y,z)

print sys.argv