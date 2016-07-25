import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
from scipy.optimize import minimize
from scipy.misc import logsumexp
from numpy import newaxis

inputfile =  '../data/data_new.csv'
# inputfile = "D:\\Dropbox\PhD Study\\research\\mnp_demand_gas_ethanol_dhl\\noisy price signal real data\\data\\data_new.csv"

df = pd.read_csv(inputfile)
df['const'] = 1

df = df.sample(frac=0.1, replace=True)

# variable labels
pcovarlbls = ['const', 'dv_carpriceadj_p75p100', 'dv_usageveh_p75p100']
covarlbls = ['dv_female', 
	'dv_age_25to40y', 'dv_age_40to65y', 'dv_age_morethan65y',
	'dv_somesecondary','dv_somecollege',
	'dv_ctb', 'dv_bh', 'dv_rec', 'const']

pricelbls = ['rel_p_km_adj2', 'rel_p_km_adj3']

np.random.seed(1234)


# place holder object for the structure of 'n' and 'data' used belew
class DataStructure:
	pass

class DataContent:
	pass

def makedata(df, covarlbls, pricelbls, pcovarlbls):
	# extract variable using above labels
	pcovar = df.loc[:, pcovarlbls]
	covar = df.loc[:, covarlbls]
	price = df.loc[:, pricelbls]

	# missing price is imputed with a very high price (10000)
	price.fillna(10000, inplace=True)

	# n holds the structure of the data
	# most information is extracted from the data frame retrived above

	n = DataStructure()
	n.obs, n.pcovar = pcovar.shape
	n.obs, n.covar  = covar.shape

	n.choice = df['choice'].max()
	n.treat  = df['treattype'].max()+1

	n.alpha  = n.pcovar
	n.beta   = n.covar*(n.choice-1)
	n.sigma  = (n.treat-1)*(n.choice-1)
	n.mu     = n.choice-2

	n.thetaidx = np.cumsum([n.alpha, n.beta, n.sigma])

	n.draw   = 100

	# data hold the actual content of the data frame above in numpy format
	# the data is transposed so that choice will be the first index 
	data = DataContent()
	data.price  = price.as_matrix().transpose()
	data.pcovar = pcovar.as_matrix().transpose()
	data.covar  = covar.as_matrix().transpose()
	data.treat  = df['treattype'].as_matrix()
	data.choice = df['choice'].as_matrix() - 1

	data.choice_dummies = pd.get_dummies(data.choice).as_matrix().transpose()
	data.z = np.random.randn(n.draw, n.choice-1, n.obs)

	# data.price[data.price > 1000] = 10000

	# order of indexing
	# 0. draw
	# 1. choice
	# 2. individual
	#
	# numpy is row-major, so 2D indexing order should be
	# 0. choice
	# 1. individual

	# construct the look-up table to quickly find the actual choice for each individual
	# i.e. given index (i,r), find index (i,j,r) with j = choice of i
	rr, ii = np.mgrid[0:n.draw, 0:n.obs]
	jj = np.tile(data.choice, (n.draw, 1))
	data.choice_flat_idx_3D = np.ravel_multi_index((rr, jj, ii), (n.draw, n.choice, n.obs))


	dd, ii = np.mgrid[0:n.pcovar, 0:n.obs]
	jj = np.tile(data.choice, (n.pcovar, 1))
	data.choice_flat_idx_3D_pcovar = np.ravel_multi_index((dd, jj, ii), (n.pcovar, n.choice, n.obs))

	#
	# # test: a.flat[3Didx][k,i] == a[k,choice[i],i]
	#
	# a = np.random.rand(n.draw, n.choice, n.obs)
	# a1 = a.flat[data.choice_flat_idx_3D]
	# a2 = np.zeros((n.draw, n.obs))
	# for i in range(n.obs):
	# 	a2[:,i] = a[:, data.choice[i], i]

	# assert (a1==a2).all(), 'choice_flat_idx_3D not match' 

	# a = np.random.rand(n.pcovar, n.choice, n.obs)
	# a1 = a.flat[data.choice_flat_idx_3D_pcovar]
	# a2 = np.zeros((n.pcovar, n.obs))
	# for i in range(n.obs):
	# 	a2[:,i] = a[:, data.choice[i], i]

	# assert (a1==a2).all(), 'choice_flat_idx_3D_pcovar not match' 

	# mark missing data
	# the first choice (gasoline), which is ommited, is never missing
	data.missing = np.pad(data.price > 10, ((1,0), (0,0)), 'constant', constant_values=True)

	return n, data

# @profile
def choiceprob(V):
	# value of the first choice is normalzied tp 0
	V = np.pad(V, ((0,0), (1,0), (0,0)), 'constant') 

	# normalize all values by subtracting logsumexp across choices
	V -= logsumexp(V, axis=1)[:,newaxis,:]

	return np.exp(V)

def getparams(theta, n):
	alpha, beta, sigma, mu = np.split(theta, n.thetaidx)

	beta  = beta.reshape(n.choice-1, n.covar)
	sigma = sigma.reshape(n.choice-1, n.treat-1)
	sigma = np.hstack((np.array([[0.4],[0.15]]), sigma))
	mu    = np.hstack((1,mu)).reshape(n.mu+1,1)

	return alpha, beta, sigma, mu

# @profile
def value(theta, data, n):
	alpha, beta, sigma, mu = getparams(theta, n)
	alphai = alpha.dot(data.pcovar)
	print 'value shape', alphai.shape, data.price.shape
	return (alphai*data.price + beta.dot(data.covar) + alphai*sigma[:,data.treat]*data.z)/mu

def loglikelihood(V):
	return np.log(choiceprob(V).flat[data.choice_flat_idx_3D].mean(axis=0))

# @profile
def nloglf(theta, data, n):
	return -loglikelihood(value(theta, data, n)).sum()

def nloglf_grad(theta, data, n):
	v     = value(theta, data, n)
	p_rji = choiceprob(v)  
	p_ri  = p_rji.flat[data.choice_flat_idx_3D]
	p_i   = p_ri.mean(axis=0)

	alpha, beta, sigma, mu = getparams(theta, n)

	# intermediate calculation
	p_ri_normalized = p_ri/(p_i*n.draw)
	factor_rji = data.choice_dummies[1:,] - p_rji[:,1:,]
	factor_ji  = data.choice_dummies[1:,] - np.sum(p_ri_normalized[:,newaxis,:]*p_rji[:,1:,:], axis=0)

	dL_alpha = data.pcovar[:,newaxis,newaxis,:]*(data.price + sigma[:,data.treat]*data.z)*factor_rji/mu
	dL_alpha[:,:,data.missing[1:,:]] = 0.0
	dL_alpha = (dL_alpha.sum(axis=2)*p_ri_normalized).sum(axis=(1,2))

	dL_beta = (data.covar[:,newaxis,:]*factor_ji/mu).sum(axis=2)

	dLi_sigma = (alpha.dot(data.pcovar)*data.z*factor_rji*p_ri_normalized[:,newaxis,:]/mu).sum(axis=0)
	dL_sigma = np.array([dLi_sigma[:,data.treat==i].sum(axis=1) for i in range(1,n.treat)])

	dL_mu = -(v[:,1:,:]*p_rji[:,2:,:]*((data.choice_dummies[2:,:]-p_ri[:,newaxis,:])/(n.draw*p_i*mu[1:,:]))).sum(axis=(0,2))

	return -np.concatenate((dL_alpha, dL_beta.transpose().flatten(), dL_sigma.transpose().flatten(), dL_mu))


def findiff(f, x):
	grad = None

	for i in range(len(x)):
		x1 = np.array(x)
		x2 = np.array(x)

		if abs(x[i]) > 1e-5:
			x1[i] *= (1-1e-5)
			x2[i] *= (1+1e-5)
		else:
			x1[i] -= 1e-5
			x2[i] += 1e-5

		f1, f2 = f(x1), f(x2)

		if grad is None:
			k = 1 if not hasattr(f1, '__len__') else len(f1)
			grad = np.zeros((k,len(x)))
			
		grad[:,i]=(f(x2)-f(x1))/(x2[i]-x1[i])

	return np.squeeze(grad)


def mlecov(thetahat, loglikelihood):
	g = findiff(loglikelihood, thetahat)
	return g.dot(g)

theta0 = np.array([
	-10,
	0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, -6, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, -8,
	0.6, 0.2,
	0.6, 0.2,
	3.0 ])

def nloglf_alpha(alpha, theta, data, n):
	theta[0] = alpha
	f = nloglf(theta, data, n)
	return f


lb = np.array([-10]*len(theta0))
ub = np.array([10]*len(theta0))

lb[0] = -100
ub[0] = 5

n, data = makedata(df, covarlbls, pricelbls, pcovarlbls)

# print nloglf(theta0,data,n)
# print findiff(lambda x: nloglf(x,data,n), theta0)
# print nloglf_grad(theta0, data, n)
print mlecov(theta0, lambda x: loglikelihood(value(x,data,n)))
exit()

alpha = np.linspace(-50,-10)
fval = [nloglf_alpha(alphai, theta0, data,n) for alphai in alpha]


plt.scatter(alpha,fval)
plt.show()
# print nloglf_grad(theta0, data, n)

# v = value(theta0, data, n)
# p = choiceprob(v)

print 'start optimization routine'

alpha = minimize(nloglf, theta0, args=(data, n), jac=nloglf_grad, options={'disp':2})
print alpha
# import time
# start = time.clock()
# for i in range(10):
# 	nloglf(theta0, data, n)

# end = time.clock()

# print end - start
