from os.path import dirname, join
import numpy as np

def load_narma2():

    datadir = join(dirname(__file__), 'data/NARMA')
    inp = np.loadtxt(join(datadir, 'Input_narma.dat'))
    target = np.loadtxt(join(datadir, 'NARMA2.dat'))
    x_train, x_test = np.split(np.reshape(inp, [1, -1, 1]), 2, axis=1)
    y_train, y_test = np.split(np.reshape(target, [-1, 1]), 2, axis=0)

    return(x_train, x_test, y_train, y_test)
