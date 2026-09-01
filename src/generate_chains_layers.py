import numpy as np
np.set_printoptions(legacy='1.25')
from time import time

from functions import *


init_point =  [80, None, [1/80 for _ in range(80-1)]]
model_name = 'big'
prior_name = 'big'
proposer_name = 'big_maximum_dimensionality_individual'
n_iter = 10
folder_name = 'dummy'

if __name__ == '__main__':
    start = time()
    worker(model_name, prior_name, proposer_name, init_point, n_iter, folder_name)
    end = time()
    print((end-start))