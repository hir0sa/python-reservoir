import numpy as np

def check_dim(X, dim):
    dimX = np.ndim(X)
    if(dimX != dim):
        raise ValueError("{0}d array is expected, but {1}d is given".format(dim, dimX))
