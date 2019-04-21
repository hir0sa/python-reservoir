import sys
import numpy as np
from scipy import sparse, linalg, stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import svm, linear_model
from ..utils import check_dim
import multiprocessing as mp
import sys

class ESN(BaseEstimator, TransformerMixin):
    def __init__(self, N_nodes=100, N_in=1, g_in = 0.1, alpha = 0.3, initial_state = None, W_res = None, W_in= None, g_res = 1.0, input_bias=True, washout_t = 0, with_time = True, with_input = True):
        self.N_nodes = N_nodes
        self.N_in = N_in
        self.input_bias = input_bias
        self.washout_t = washout_t
        self.with_time = with_time
        self.with_input = with_input

        #set reservoir weight
        if W_res is None:
            
            n_try = 10
            for i in range(n_try):
                W = sparse.random(self.N_nodes, self.N_nodes, density=0.1, data_rvs=stats.uniform(loc=-0.5, scale=1.0).rvs, random_state=42)
                rhoW = max(abs(linalg.eig(W.todense())[0]))
                if rhoW != 0:
                    break
            if rhoW == 0:
                raise ValueError("tried to make random matrix {} times, but all eigenvalues of them are zero".format(n_try))

            self.W_res =  W.tocsr()/ rhoW
    
        else:            
            self.W_res = W_res

        #g_res
        self.g_res = g_res

        #set input weight
        self.g_in = g_in
        if W_in is None :
            np.random.seed(seed=32)
            if(input_bias):
                self.W_in = (np.random.rand(self.N_nodes, 1 + self.N_in)-0.5)
            else:
                self.W_in = (np.random.rand(self.N_nodes, self.N_in)-0.5)
        else:
            self.W_in = W_in

        #set initial state
        if initial_state is None:
            self.x = np.zeros(self.N_nodes)
        else:
            self.x = initial_state

        #set leaking rate
        self.alpha = alpha

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
        if(isinstance(self._W_res, sparse.csr_matrix)):
            for u in In:
                self.iterate_csr(u)
        else:
            for u in In:
                self.iterate(u)

    def feed_time(self, In):
        check_dim(In, 2)
        T, n_features = np.shape(In)
        X = np.zeros([T - self.washout_t, self.N_nodes + 1])
        if(isinstance(self._W_res, sparse.csr_matrix)):
            for t in range(T):
                self.iterate_csr(In[t])
                if t > self.washout_t:
                    X[t - self.washout_t] = np.hstack([self.x, 1])
        else:
            for t in range(T):
                self.iterate(In[t])
                if t > self.washout_t:
                    X[t - self.washout_t] = np.hstack([self.x, 1])

        return X

    def feed_time_with_input(self, In):
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

class ESNR(ESN):
    def __init__(self, N_nodes, N_in, g_in = 0.1, alpha = 0.3, initial_state = None, W_res = None, W_in= None, g_res = 1.0, input_bias = True,  washout_t = None, with_input = True, clf=None, **cl_pram):
        super().__init__(N_nodes, N_in, g_in, alpha, initial_state, W_res, W_in, g_res, input_bias, washout_t, with_time=True, with_input=with_input)
        if clf is None:
            self.clf = linear_model.Ridge(alpha=0.00001)
        else:
            self.clf = clf(**cl_pram)

    def fit(self, X, y):        
        Out = self.transform(X)
        self.clf.fit(Out, y)        
        return self

    def predict(self, X):
        Out = self.transform(X)
        return self.clf.predict(Out)


class ESNC(ESN):
    def __init__(self, N_nodes, N_in, g_in = 0.1, alpha = 0.3, initial_state = None, W_res = None, W_in= None, g_res = 1.0, input_bias = True,  washout_t = None,  with_time = True, with_input = True, **cl_pram):
        super().__init__(N_nodes, N_in, g_in, alpha, initial_state, W_res, W_in, g_res, input_bias, washout_t, with_time, with_input)
        if clf is None:
            self.clf = linear_model.svm.LinearSVC()
        else:
            self.clf = clf(**cl_pram)

    def fit(self, X, y):        
        Out = self.transform(X)
        self.clf.fit(Out, y)  
        return self      

    def predict(self, X):
        Out = self.transform(X)
        return self.clf.predict(Out)
