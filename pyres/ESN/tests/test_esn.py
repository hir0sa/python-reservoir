from os.path import dirname, join
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_almost_equal
from pyres.ESN import ESNbase, ESNR
from pyres.datasets import load_narma2

def test_esn_parameters():
  
    param = dict(
        N_nodes = 10,
        N_in = 2,
        g_in = 1.0,
        alpha = 0.1,
        W_res = np.random.rand(10, 10),
        W_in = np.random.rand(10, 2), 
        g_res = 0.5,
        washout_t = 10
    )

    esn = ESNbase(**param)

    assert param['N_nodes'] == esn.N_nodes
    assert param['N_in'] == esn.N_in
    assert param['g_in'] == esn.g_in
    assert_array_equal(param['W_res'], esn.W_res)
    assert_array_equal(param['W_in'], esn.W_in)
    assert param['g_res'] == esn.g_res
    assert param['washout_t'] == esn.washout_t

def test_feed_time():
    N_in = 3
    T = 100
    N_nodes = 50
    washout_t = 10
    inp = np.random.rand(T, N_in)
    esn = ESNbase(N_nodes=N_nodes, N_in=N_in, washout_t=washout_t)

    assert esn.feed_time(inp).shape == (T-washout_t, N_nodes+1)
    assert esn.feed_time_with_input(inp).shape == (T-washout_t, N_nodes+N_in+1)


def test_transform():
    N_in = 3
    T = 100
    N_nodes = 50
    N_sample=2
    washout_t = 10

    X = np.random.rand(N_sample, T, N_in)
    esn = ESNbase(N_nodes=N_nodes, N_in=N_in, washout_t=washout_t, with_input=True, with_time=True)

    assert esn.transform(X).shape == (N_sample*(T-washout_t), N_nodes+N_in+1)
    esn.with_input = True
    esn.with_time = False
    assert esn.transform(X).shape == (N_sample, N_nodes+N_in+1)
    esn.with_input = False
    esn.with_time = True
    assert esn.transform(X).shape == (N_sample*(T-washout_t), N_nodes+1)
    esn.with_time = False
    esn.with_time = False
    assert esn.transform(X).shape == (N_sample, N_nodes+1)

def test_performace_esnr(): 
    
    N_nodes=100
    N_in = 1
    washout_t  = 100
    g_in = 10
    x_train, x_test, y_train, y_test = load_narma2()
    y_train = y_train[washout_t:, 0]

    ## using default classifier 
    esnr = ESNR(N_nodes, N_in, g_in=g_in, washout_t=washout_t)
    esnr.fit(x_train, y_train)
    esnr.washout_t=0
    out = esnr.predict(x_test)
    nrmse = np.sqrt(mean_squared_error(out, y_test))/np.std(y_test)

    assert nrmse < 0.01

    ## using user defined classifier 
    clf = Ridge
    cl_param = {'alpha': 0.0001}
    
    esnr = ESNR(N_nodes, N_in, g_in=g_in, washout_t=washout_t, clf=clf, cl_param=cl_param)
    esnr.fit(x_train, y_train)
    esnr.washout_t=0
    out = esnr.predict(x_test)
    nrmse = np.sqrt(mean_squared_error(out, y_test))/np.std(y_test)

    assert nrmse < 0.01
