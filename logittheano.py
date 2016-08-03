import numpy as np
import theano
import theano.tensor as T
from theano import pp
import pandas as pd
import theano.gradient
import pyipopt
import scipy.stats
import numpy.random
import os.path
import sys

ghalton_flag = True

try:
    import ghalton
except ImportError:
    ghalton_flag = False
    

class DataContent:
    def __init__(self, df, lbls, n, seed = 1234):
        pricelbls, covarlbls, pcovarlbls, zcovarlbls = lbls
        
        # extract variable using above labels        
        self.price  = df.loc[:, pricelbls].fillna(10000).as_matrix().transpose()
        self.covar  = df.loc[:, covarlbls].as_matrix().transpose()
        self.pcovar = df.loc[:, pcovarlbls].as_matrix().transpose()
        self.zcovar = df.loc[:, zcovarlbls].as_matrix().transpose()
        self.choice = df.loc[:, 'choice'].as_matrix().transpose()-1

        # draws from Halton sequence (using ghalton module)
        filename = 'draw_' + '_'.join(str(x) for x in [seed, n.draw, n.choice-1, n.obs]) + '.npy'
        if os.path.isfile(filename):
            print '*** read random draws from ' + filename
            self.z = np.load(filename)
            assert self.z.shape == (n.draw, n.choice-1,n.obs)
        elif ghalton_flag:
            print '*** no draw.npy found, generate random draws from halton sequence'
            sequencer = ghalton.GeneralizedHalton(ghalton.EA_PERMS[:n.choice-1])
            draws = scipy.stats.norm().ppf(sequencer.get(n.draw*n.obs)).reshape((n.obs, n.draw, n.choice-1))
            self.z = np.transpose(draws, (1,2,0))
            np.save(filename, self.z)
        else:
            raise Exception('!!! cannot load module ghalton nor find ' + filename)
            
        # np.random.seed(seed)
        # data.z = np.random.randn(n.draw, n.choice-1, n.obs)

        # construct the look-up table to quickly find the actual choice for each individual
        # i.e. given index (i,r), find index (i,j,r) with j = choice of i
        rr, ii = np.mgrid[0:n.draw, 0:n.obs]
        jj = np.tile(self.choice, (n.draw, 1))
        self.choice_flat_idx_3D = np.ravel_multi_index((rr, jj, ii), (n.draw, n.choice, n.obs))

class DataStructure:
    def __init__(self, df, lbls, draw=100):
        pricelbls, covarlbls, pcovarlbls, zcovarlbls = lbls
        
        self.pcovar = len(pcovarlbls)
        self.covar  = len(covarlbls)
        self.zcovar = len(zcovarlbls)
        self.choice = len(pricelbls) + 1
        
        self.alpha  = self.pcovar
        self.beta   = self.covar*(self.choice-1)
        self.sigma  = self.zcovar*(self.choice-1)
        self.mu     = self.choice-2
        
        self.obs    = df.shape[0]
        self.draw   = draw

# ## Model Specification
# * Indirect utility $$u_{rji}=\alpha D_i \left(p_{ji} + \sigma_{ji} z_{rji}\right) + \beta_j X_i + \mu_j \epsilon_{rji}=\underbrace{\alpha D_i p_{ji} + \beta_j X_i}_{V_{ji}^{fixed}} + \underbrace{\alpha D_i \sigma_{ji} z_{rji}}_{V_{rji}^{noise}} + \mu_j \epsilon_{rji}$$
# 
# * Choice probability $$P_{rji} = \frac{\exp\left(\frac{V_{rji}}{\mu_j}\right)}{1+\sum_k \exp\left(\frac{V_{rki}}{\mu_k}\right)}$$
# 
# * Log likelihood ($j_i^*$ denotes the actual choice of $i$) $$L = \sum_i \ln\left(\frac{1}{R} \sum_r P_{rj_i^*i}\right)$$

# In[6]:

# theano.optimizer='fast_compile'
# theano.exception_verbosity='high'

def getparams(theta, n):
    alpha  = theta[                      :n.alpha               ]
    beta   = theta[n.alpha               :n.alpha+n.beta        ].reshape((n.choice-1, n.covar))
    sigma  = theta[n.alpha+n.beta        :n.alpha+n.beta+n.sigma].reshape((n.choice-1, n.zcovar))
    mu     = theta[n.alpha+n.beta+n.sigma:                      ]
    
    return alpha, beta, sigma, mu
 
def buildtheano(data, n):
    theta  = T.fvector('theta')
    alpha, beta, sigma, mu = getparams(theta, n)
    
    mu     = T.concatenate([T.ones(1, dtype='float32'), mu])
    
    price  = theano.shared(data.price.astype('float32'),  name='price')
    covar  = theano.shared(data.covar.astype('float32'),  name='X1')
    pcovar = theano.shared(data.pcovar.astype('float32'), name='X2')
    draw   = theano.shared(data.z.astype('float32'),      name='z')
    zcovar = theano.shared(data.zcovar.astype('float32'), name='X3')

    alphai     = T.dot(alpha,pcovar)
    valuefixed = alphai*price + T.dot(beta,covar)
    valuenoise = alphai*T.exp(T.dot(sigma, zcovar)).dimshuffle('x',0,1)*draw
    value      = (valuefixed.dimshuffle('x',0,1) + valuenoise)/mu.dimshuffle('x',0,'x')

    value2     = T.concatenate([T.zeros((n.draw,1,n.obs), dtype='float32'), value], axis = 1)
    value3     = value2 - value2.max(axis=1, keepdims=True)

    expvalue   = T.exp(value3)
    prob       = expvalue/expvalue.sum(axis=1,keepdims=True)

    nloglf = -T.log(prob.flatten()[data.choice_flat_idx_3D].mean(axis=0)).sum()
    
    return theta, nloglf

def buildfunc(theta, nloglf):
    return (theano.function([theta], outputs = nloglf),
        theano.function([theta], outputs = T.grad(nloglf, [theta])),
        theano.function([theta], outputs = theano.gradient.hessian(nloglf, [theta])))

def buildeval(theta, nloglf):
    f, grad, hess = buildfunc(theta, nloglf)
    
    def eval_f(thetavalue):
        return f(thetavalue)

    def eval_grad(thetavalue):
        return np.squeeze(grad(thetavalue))

    def eval_hess(thetavalue):
        return np.squeeze(hess(thetavalue))

    return eval_f, eval_grad, eval_hess

def buildpartialeval(theta, nloglf, idxdelete, thetafix):
    f, grad, hess = buildfunc(theta, nloglf)    
    idxinsert = [idxdelete[i]-i for i in range(len(idxdelete))]
    
    def filled(thetavalue):
        return np.insert(thetavalue, idxinsert, thetafix)
    
    def eval_f(thetavalue):
        return f(filled(thetavalue))

    def eval_grad(thetavalue):
        g= np.delete(np.squeeze(grad(filled(thetavalue))), idxdelete)
        return g

    def eval_hess(thetavalue):
        H = np.squeeze(hess(filled(thetavalue)))
        return np.delete(np.delete(H, idxdelete, axis=0), idxdelete, axis=1)
    
    return eval_f, eval_grad, eval_hess

def calcovbhhh(thetahat, loglikelihoodfunc):
    g = np.zeros((len(thetahat), n.obs))
    
    for i in range(len(thetahat)):
        theta1 = np.array(thetahat)
        theta2 = np.array(thetahat)

        theta1[i] = thetahat[i]*(1-1e-4)
        theta2[i] = thetahat[i]*(1+1e-4)

        f1 = np.squeeze(loglikelihoodfunc(theta1))
        f2 = np.squeeze(loglikelihoodfunc(theta2))

        g[i,:] = (f2-f1)/(theta2[i]-theta1[i])
        
    return np.linalg.inv(np.asmatrix(g.dot(np.transpose(g))))
    
    
def calcovhess(thetahat, hess):
    return np.linalg.inv(hess(thetahat))

def calse(cov):
    return np.diag(cov)**0.5

def __init__():
    pass
