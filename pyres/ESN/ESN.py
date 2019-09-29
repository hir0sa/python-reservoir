import sys
import numpy as np
from scipy import sparse, linalg, stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm, linear_model
from ..utils import check_dim
import multiprocessing as mp
import sys

class ESNbase(BaseEstimator, TransformerMixin):
    """Base class for echo state network as scikit-learn compatible model.

    Parameters
    ----------
    N_nodes : int
        Number of internal nodes.

    N_in : int
        Dimension of a input.

    g_in : float
        Strength of input weights
    
    alpha : float
        leaking rate of dynamics

    initial_state : array, optional
        Initial state of echo state network.
        If not provided, zero vector is given.

    W_res : array or sparse matrix, shape (N_nodes, N_nodes), optional
        Weights of internal network.
        If not provided, random sparse matrix (10% edge density) is given.

    W_in : array, (N_nodes, N_in),  optional
        Weights of input to internal network.
        If not provided, random matrix is given.

    g_in : float
        Strength of weights of internal network.

    input_bias : bool
        If true, extra bias dimension is added to input.

    washout_t : int
        Duration to discard initial time steps

    with_time : bool
        If true, the model output all timeseries

    with_input : bool
        If true, the model output with input  
        

    """


    def __init__(self, N_nodes=100, N_in=1, g_in = 0.1, alpha = 0.3, initial_state = None, W_res = None, W_in= None, g_res = 1.0, input_bias=True, washout_t = 0, with_time = True, with_input = True):
        self.N_nodes = N_nodes
        self.N_in = N_in
        self.input_bias = input_bias
        self.washout_t = washout_t
        self.with_time = with_time
        self.with_input = with_input

        #set reservoir weight
        if W_res is None:
            self.W_res = self._generate_random_sparse_matrix(N_nodes, N_nodes)
                
        else:            
            self.W_res = W_res

        #g_res
        self.g_res = g_res

        #set input weight
        self.g_in = g_in
        if W_in is None :
            np.random.seed(seed=32)
            if(input_bias):
                self.W_in = self._generate_random_dense_matrix(N_nodes, 1+N_in)
            else:
                self.W_in = self._generate_random_dense_matrix(N_nodes, N_in)
        else:
            self.W_in = W_in

        #set initial state
        if initial_state is None:
            self.x = np.zeros(self.N_nodes)
        else:
            self.x = initial_state

        #set leaking rate
        self.alpha = alpha

    def _generate_random_sparse_matrix(self, x_dim, y_dim):
        n_try = 10
        for i in range(n_try):
            W = sparse.random(x_dim, y_dim, density=0.1, data_rvs=stats.uniform(loc=-0.5, scale=1.0).rvs, random_state=42)
            rhoW = max(abs(linalg.eig(W.todense())[0]))
            if rhoW != 0:
                break
        else :
            raise ValueError("tried to make random matrix {} times, but all eigenvalues of them are zero".format(n_try))
    
        return  W.tocsr()/ rhoW

    def _generate_random_dense_matrix(self, x_dim, y_dim):
        np.random.seed(seed=32)
        return np.random.rand(x_dim, y_dim) - 0.5


    @property
    def W_res(self):
        return self._W_res

    @W_res.setter
    def W_res(self, W_res):
        check_dim(W_res, 2)
        self._W_res = W_res

    @property
    def W_in(self):
        return self._W_in

    @W_in.setter
    def W_in(self, W_in):
        check_dim(W_in, 2)
        self._W_in = W_in


    def iterate_csr(self, u):
        if(self.input_bias):
            self.x = (1-self.alpha)*self.x + self.alpha*np.tanh( self.g_in*np.dot( self._W_in, np.concatenate([[1],u])) + self.g_res * (self._W_res * self.x ))
        else:
            self.x = (1-self.alpha)*self.x + self.alpha*np.tanh( self.g_in*np.dot( self._W_in, u ) + self.g_res * (self._W_res * self.x ))

    def iterate(self, u):
        if(self.input_bias):
            self.x = (1-self.alpha)*self.x + self.alpha*np.tanh( self.g_in*np.dot( self._W_in, np.concatenate([[1],u])) + self.g_res * np.dot( self._W_res, self.x ))
        else :
            self.x = (1-self.alpha)*self.x + self.alpha*np.tanh( self.g_in*np.dot( self._W_in,u) + self.g_res * np.dot( self._W_res, self.x ))

    def feed(self, In):
        """Run ESN dynamics with input time series to ESN 

        Parameters
        ----------

        In : array, shape = [n_timesteps, n_features]
            Input timeseries.

        """
        if(isinstance(self._W_res, sparse.csr_matrix)):
            for u in In:
                self.iterate_csr(u)
        else:
            for u in In:
                self.iterate(u)

    def feed_time(self, In):
        """Run ESN dynamics with input time series to ESN and output internal states timeseries

        Parameters
        ----------

        In : array, shape = [n_timesteps, n_features]
            Input timeseries.

        Returns
        ----------
        States : array, shape = [n_timesteps - washout_t, N_nodes + 1]
            return timeseries of internal ESN states.

        """

        check_dim(In, 2)
        T, n_features = np.shape(In)
        States = np.zeros([T - self.washout_t, self.N_nodes + 1])
        if(isinstance(self._W_res, sparse.csr_matrix)):
            for t in range(T):
                self.iterate_csr(In[t])
                if t > self.washout_t:
                    States[t - self.washout_t] = np.hstack([self.x, 1])
        else:
            for t in range(T):
                self.iterate(In[t])
                if t > self.washout_t:
                    States[t - self.washout_t] = np.hstack([self.x, 1])

        return States

    def feed_time_with_input(self, In):
        """Run ESN dynamics with input time series and output and output internal states and input timeseries.

        Parameters
        ----------

        In : array, shape = [n_timesteps, n_features]
            Input timeseries.

        Returns
        ----------
        States : array, shape = [n_timesteps - washout_t, N_nodes + n_features + 1]
            return timeseries of internal ESN states and input.

        """
        check_dim(In, 2)
        T, n_features = np.shape(In)
        X = np.zeros([T - self.washout_t, self.N_nodes + n_features + 1 ])
        if(isinstance(self._W_res, sparse.csr_matrix)):
            for t in range(T):
                self.iterate_csr(In[t])
                if t > self.washout_t:
                    X[t - self.washout_t] = np.hstack([self.x, In[t], 1])
        else:
            for t in range(T):
                self.iterate(In[t])
                if t > self.washout_t:
                    X[t - self.washout_t] = np.hstack([self.x, In[t], 1])

        return X

    def fit(self, X, y=None):
        pass
    
    def transform(self, X, initiate=False):
        """Feed input time series data to ESN

        Parameters
        ----------

        X : array, shape = [n_samples, n_timesteps, n_features]
            Input timeseries data.

        initiate : bool
            If true, initiate internal states everytime

        Returns
        ----------
        self : returns an instance of self.
        """

        check_dim(X, 3)
        n_samples, T, n_features = np.shape(X)

        if self.with_input is True:
            if self.with_time is True:
                Out = np.zeros([n_samples, T - self.washout_t, self.N_nodes + n_features + 1])
                for i in range(n_samples):
                    if(initiate):
                        self.initiate()
                    Out[i] = self.feed_time_with_input(X[i])
                Out = Out.reshape([-1, Out.shape[2]])
      
            else:
                Out = np.zeros([n_samples, self.N_nodes + n_features + 1])
                for i in range(n_samples):
                    if(initiate):
                        self.initiate()
                    self.feed(X[i])
                    Out[i] = np.hstack([self.x, X[i][-1], 1])

        else:
            if self.with_time is True:
                Out = np.zeros([n_samples, T - self.washout_t, self.N_nodes + 1])
                for i in range(n_samples):
                    if(initiate):
                        self.initiate()
                    Out[i] = self.feed_time(X[i])
                Out = Out.reshape([-1, Out.shape[2]])

            
            else:
                Out = np.zeros([n_samples, self.N_nodes + 1])
                for i in range(n_samples):
                    if(initiate):
                        self.initiate()
                    self.feed(X[i])
                    Out[i] = np.hstack([self.x,1])
        return Out

    def fit_transform(self, X, y=None, with_time=True):
        Out = self.transform(X)
        return Out
    
                
    def initiate(self):
        self.x = np.zeros(self.N_nodes)

class ESNR(ESNbase):
    """Echo state network with regressor 

    Parameters
    ----------

    reg : object
        regressor as scikit-learn compatible model

    reg_param : dict
        Parameters for reg

    """

    def __init__(self, N_nodes, N_in, g_in = 0.1, alpha = 0.3, initial_state = None, W_res = None, W_in= None, g_res = 1.0, input_bias = True,  washout_t = None, with_input = True, reg=None):
        super().__init__(N_nodes, N_in, g_in, alpha, initial_state, W_res, W_in, g_res, input_bias, washout_t, with_time=True, with_input=with_input)
        
        if reg is None:
            self.reg = linear_model.Ridge(alpha = 0.00001)
        else:

            self.reg = reg

    def fit(self, X, y):
        """Fit regression model

        Parameters
        ----------

        X : array, shape = [n_samples * n_timesteps, n_features]
            Input timeseries data.

        y : array, shape = [n_samples * n_timesteps, ]
            Target values

        Returns
        ----------
        self : returns an instance of self.
        """

        Out = self.transform(X)
        self.reg.fit(Out, y)
        return self

    def predict(self, X):
        """Predict using the fitted model
        
        X : array, shape = [n_samples * n_timesteps, n_features]
            Input timeseries data.
        
        Returns:
        ----------
        Out : array [n_sample * n_timesteps, ]
            predicted values.            

        """

        Out = self.transform(X)
        return self.reg.predict(Out)


class ESNC(ESNbase):
    """Echo state network with classifier 

    Parameters
    ----------

    clf : object
        regressor as scikit-learn compatible model

    cl_param : dict
        Parameters for clf

    """

    def __init__(self, N_nodes, N_in, g_in = 0.1, alpha = 0.3, initial_state = None, W_res = None, W_in= None, g_res = 1.0, input_bias = True,  washout_t = None,  with_time = False, with_input = True, clf=None):
        super().__init__(N_nodes, N_in, g_in, alpha, initial_state, W_res, W_in, g_res, input_bias, washout_t, with_time, with_input)
        if clf is None:
            self.clf = linear_model.svm.LinearSVC()
        else:
            self.clf = clf

    def fit(self, X, y):
        """Fit classifier model

        Parameters
        ----------

        X : array, shape = [n_samples, n_timesteps, n_features]
            Input timeseries data.

        y : array, shape = [n_samples, ]
            Target values

        Returns
        ----------
        self : returns an instance of self.
        """

        Out = self.transform(X)
        self.clf.fit(Out, y)  
        return self      

    def predict(self, X):
        """Predict using the fitted model
        
        X : array, shape = [n_samples, n_timesteps, n_features]
            Input timeseries data.
        
        Returns:
        ----------
        Out : array [n_sample, ]
            predicted values.            
                        
        """
        Out = self.transform(X)
        return self.clf.predict(Out)
