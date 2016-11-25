import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import theano.gradient
import theano.tensor.slinalg
import pyipopt

inputfile = '../data/data_new_lp.csv'
df = pd.read_csv(inputfile)

# generate const and treatment dummies
df['const'] = 1
df = pd.concat([df, pd.get_dummies(df['treattype'], prefix='dv_treat')], axis=1)

#df = df[(df['choice'] == 1) | (df['choice']==2)] 
#df['choice'][df['choice']==3]=2
#%%

pricelbls = ["rel_lp_km_adj2", "rel_lp_km_adj3" ]
#pricelbls = ["rel_p_km_adj2" ]
Xlbls =     [
      "dv_female", 
      "dv_age_25to40y", 
      "dv_age_40to65y", 
      "dv_age_morethan65y", 
      "dv_somesecondary", 
      "dv_somecollege", 
      "dv_carpriceadj_p75p100", 
      "dv_usageveh_p75p100", 
      "stationvisit_avgcarprice_adj",
      "dv_ctb", 
      "dv_bh", 
      "dv_rec", 
      "const"
    ]

Xplbls = ['const']

choice  = df['choice'].as_matrix()
price   = df.loc[:, pricelbls].as_matrix().transpose()
X       = df.loc[:, Xlbls].as_matrix().transpose()
Xp      = df.loc[:, Xplbls].as_matrix().transpose()
groupid = df.loc[:, 'treattype'].as_matrix().transpose()

nX, nXp, nchoice  = len(Xlbls), len(Xplbls), len(pricelbls) + 1

nalpha  = nXp
nbeta   = nX*(nchoice-1)
nsigma  = (nchoice-1)*nchoice/2 - 1

nobs    = df.shape[0]
ndraw   = 10
ngroup  = np.unique(groupid).size

nallsigma = (nsigma+1)*ngroup - 1

#%%
tril_index_matrix = np.zeros((ngroup, nchoice-1, nchoice-1), dtype=int) + nallsigma + 1

tril_index = np.vstack([np.repeat(np.arange(ngroup), nsigma+1), 
                       np.tile(np.tril_indices(nchoice-1), ngroup)]).tolist()

tril_index_matrix[tril_index] = np.arange(nallsigma+1)


def getparams(theta):
    offset = 0
    alpha  = theta[offset: offset + nalpha]
    
    offset += nalpha
    beta   = theta[offset: offset + nbeta].reshape((nchoice-1, nX))
    
    offset += nbeta
    s = T.concatenate([[1], theta[offset: offset + nallsigma], [0]])
    S = s[tril_index_matrix]
    
    return alpha, beta, S

floatX = 'float64'

theta  = T.dvector('theta')
alpha, beta, S = getparams(theta)
    
#Tprice  = theano.shared(price.astype(floatX),  name='price')
#TX      = theano.shared(X.astype(floatX),  name='X')
#TXp     = theano.shared(Xp.astype(floatX), name='Xp')
#TXhet   = theano.shared(Xhet.astype(floatX), name='Xhet')

V       = (T.dot(alpha,Xp)*price + T.dot(beta,X))
Vfull   = T.concatenate([T.zeros((1, nobs)), V], axis = 0)
Vchoice = Vfull[(choice-1,np.arange(nobs))]
Vnorm   = (Vfull - Vchoice)

nonchoicedummy = np.ones((nobs, nchoice), dtype = bool)
nonchoicedummy[(range(nobs), choice-1)] = False
iii = np.arange(nobs*nchoice).reshape(nonchoicedummy.shape)
nonchoiceidx = iii[nonchoicedummy].reshape((nobs, nchoice-1))

Vnonchoice = Vnorm.transpose().flatten()[nonchoiceidx].transpose()               

#%%
M = np.stack([[[1,0],[0,1]], [[-1,0],[-1,1]], [[0,-1],[1,-1]]])
MS = T.dot(M, S).dimshuffle((0,2,1,3)).reshape((ngroup*nchoice, nchoice-1, nchoice-1))
Sigma = T.batched_dot(MS, MS.dimshuffle((0,2,1))).reshape((nchoice, ngroup, nchoice-1, nchoice-1))

#%%

iii = (choice-1, groupid)

c00 = T.sqrt(Sigma[:,:,0,0])
c10 = Sigma[:,:,1,0]/c00
c11 = T.sqrt(Sigma[:,:,1,1] - c10**2)

def normcdf(x):
    return 0.5 + 0.5*T.erf(x/np.sqrt(2))
                       
def norminv(p):
    return np.sqrt(2)*T.erfinv(2*p-1)
    
ndraws = 50
draws = np.random.random((ndraws,nobs))

prob0 = normcdf(-Vnonchoice[0,:]/c00[iii])
draws1 = norminv(draws*prob0)
prob1 = normcdf(-(Vnonchoice[1,:] + c10[iii]*draws1)/c11[iii])

nlogl = -T.log(prob0).sum() - T.log(prob1.sum(axis=0)/ndraws).sum()
#nlogl = -T.log(prob0).sum()
    
nlogllogit = T.log(T.sum(T.exp(Vfull), axis=0)).sum() - (Vchoice).sum()
theta0 = np.zeros((nalpha+nbeta+nsigma,))
theta0[0] = -20
theta0[-1] = 0.1
theta0[-2] = 0.2

obj = nlogl
eval_f = theano.function([theta], outputs = obj)
grad = theano.function([theta], outputs = T.grad(obj, [theta]))
hess = theano.function([theta], outputs = theano.gradient.hessian(obj, [theta]))

eval_grad = lambda t: np.squeeze(grad(t))
eval_hess = lambda t: np.squeeze(hess(t))


#%%
alpha0 = [-5.63116686]
beta0 = [-0.14276449833885524, 0.07799550939333146, 0.0690886479616759, -0.031114683614026983, -0.09391731389704802, -0.1269116325321836, -0.09564480677074452, -0.035482238485123836, -0.050698241761471995, -0.03731223127056641, -0.7783360705348067, -0.5328394135746228, 1.6107622200281881, 
         -0.1383741971290979, 0.16894742379408748, 0.33464904615230423, 0.5473575675980583, 0.0022791624344226727, 0.12501040929703963, 0.1474707888708112, 0.10599018593441098, -0.051455999185487045, -0.33470501668838093, -0.5669505382552235, -0.7647587714144124, 0.17373775908415154]
gamma0 = [-0.10, -0.05]
S0 = [0.4389639,1.07280456,1,0.4389639,1.07280456,1,0.4389639,1.07280456]

theta0 = np.concatenate([alpha0, beta0, S0])

#%%

pyipopt.set_loglevel(1)
thetahat , _, _, _, _, fval = pyipopt.fmin_unconstrained(
    eval_f,
    theta0,
    fprime=eval_grad,
    fhess=eval_hess,
    )



