import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import theano.gradient
import theano.tensor.slinalg
import pyipopt
import sys
import json

from scipy.stats import norm

#specname = sys.argv[1]
specname = 'spec1het'

with open(specname + '.json', 'r') as f:
    spec = json.load(f)
    inputfile = spec['inputfile']
    lbls = spec['label']
    Xlbls = lbls['X']
    Xplbls = lbls['Xp']
    pricelbls = lbls['price']
    grouplbls = lbls['group']

df = pd.read_csv(inputfile)

# make sure there is const in the data
df['const'] = 1

#%%

choice  = df['choice'].as_matrix()
price   = df.loc[:, pricelbls].as_matrix().transpose()
X       = df.loc[:, Xlbls].as_matrix().transpose()
Xp      = df.loc[:, Xplbls].as_matrix().transpose()
groupid = df.loc[:, grouplbls].as_matrix().transpose()

nX, nXp, nchoice  = len(Xlbls), len(Xplbls), len(pricelbls) + 1

nalpha  = nXp
nbeta   = nX*(nchoice-1)
nsigma  = (nchoice-1)*nchoice/2 - 1

nobs    = df.shape[0]
ndraw   = 10
ngroup  = np.unique(groupid).size

nallsigma = (nsigma+1)*ngroup - 1

stationidold = df['stationid'].astype(int).as_matrix()
uniquestationid = np.unique(stationidold)
nstation = uniquestationid.shape[0]

stationid = np.zeros(stationidold.shape, dtype=int)
for i in range(nstation):
    stationid[stationidold==uniquestationid[i]] = i
    
dv_choice = np.arange(nchoice).reshape((nchoice,1)) == (choice-1)
nobsstation = np.array([(stationid==sid).sum() for sid in range(nstation)])


#%% nuisance xi
sumchoice_station = np.array([dv_choice[1:, stationid == i].sum(axis = 1) for i in range(nstation)]).transpose()
nuisancexi = sumchoice_station == 0
nxi = (1-nuisancexi).sum()

xi_idx = np.zeros(nuisancexi.shape, dtype = int) + nxi
xi_idx[~nuisancexi] = np.arange(nxi)

xi_idx_flat = np.where(~nuisancexi.flatten())[0]

#%%
tril_index_matrix = np.zeros((ngroup, nchoice-1, nchoice-1), dtype=int) + nallsigma + 1

tril_index = np.vstack([np.repeat(np.arange(ngroup), nsigma+1), 
                       np.tile(np.tril_indices(nchoice-1), ngroup)]).tolist()

tril_index_matrix[tril_index] = np.arange(nallsigma+1)


def getparams(theta, mm=T):
    offset = 0
    alpha  = theta[offset: offset + nalpha]
    
    offset += nalpha
    beta   = theta[offset: offset + nbeta].reshape((nchoice-1, nX))
    
    offset += nbeta
    sraw = mm.concatenate([[1], theta[offset: offset + nallsigma], [0]])
    S = sraw[tril_index_matrix]
    
    offset += nallsigma
    xiraw = mm.concatenate([theta[offset:offset+nxi], [-10000.]])
    xi = xiraw[xi_idx]
    
    return alpha, beta, S, xi

floatX = 'float64'

theta  = T.dvector('theta')
alpha, beta, S, xi = getparams(theta)
    
V       = T.dot(alpha,Xp)*price + T.dot(beta,X) + xi[:,stationid]
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

c00 = T.sqrt(Sigma[:,:,0,0])
c10 = Sigma[:,:,1,0]/c00
c11 = T.sqrt(Sigma[:,:,1,1] - c10**2)

iii = (choice-1, groupid)

normcdf = lambda x: 0.5 + 0.5*T.erf(x/np.sqrt(2))
norminv = lambda p: np.sqrt(2)*T.erfinv(2*p-1)
    
ndraws = 10
#draws = np.random.random((ndraws,nobs))
draws = (np.tile(np.arange(ndraws), (nobs,1)).transpose() + 0.5)/ndraws

prob0 = normcdf(-Vnonchoice[0,:]/c00[iii])
draws1 = norminv(draws*prob0)
prob1 = normcdf(-(Vnonchoice[1,:] + c10[iii]*draws1)/c11[iii])

nlogl = -T.log(prob0).sum() - T.log(prob1.sum(axis=0)/ndraws).sum()
#nlogl = -T.log(prob0).sum()
    
nlogllogit = T.log(T.sum(T.exp(Vfull), axis=0)).sum() - (Vchoice).sum()
obj = nlogl
eval_f = theano.function([theta], outputs = obj)
grad = theano.function([theta], outputs = T.grad(obj, [theta])[0])
hess = theano.function([theta], outputs = theano.gradient.hessian(obj, [theta])[0])

#%%
alpha0 = [-5.63116686]
beta0 = [-0.14276449833885524, 0.07799550939333146, 0.0690886479616759, -0.031114683614026983, -0.09391731389704802, -0.1269116325321836, -0.09564480677074452, -0.035482238485123836, -0.050698241761471995, -0.03731223127056641, -0.7783360705348067, -0.5328394135746228, 1.6107622200281881, 
         -0.1383741971290979, 0.16894742379408748, 0.33464904615230423, 0.5473575675980583, 0.0022791624344226727, 0.12501040929703963, 0.1474707888708112, 0.10599018593441098, -0.051455999185487045, -0.33470501668838093, -0.5669505382552235, -0.7647587714144124, 0.17373775908415154]
gamma0 = [-0.10, -0.05]
S0 = [0.4389639,1.07280456,1,0.4389639,1.07280456,1,0.4389639,1.07280456]
xi0 = np.zeros((nxi,))
theta0 = np.concatenate([alpha0, beta0, S0, xi0])

#%%

Vallbase        = T.dot(M[1:,:,:], V)
p0allbase       = normcdf(-Vallbase[:,0,:]/c00[1:,groupid])
#drawsallbase    = np.random.random((ndraws,nchoice,nobs))
drawsallbase    =  (np.tile(np.arange(ndraws), (nobs,nchoice-1,1)).transpose() + 0.5)/ndraws
draws1allbase   = norminv(drawsallbase*prob0)
p1allbase       = normcdf(-(Vallbase[:,1,:] + c10[1:,groupid]*draws1allbase)/c11[1:,groupid]).mean(axis=0)

pallbase = p0allbase*p0allbase

pstation = T.stack([pallbase[:,np.where(stationid==i)[0]].mean(axis=1) for i in range(nstation)]).transpose().flatten()[xi_idx_flat]
pstationtrue = np.stack([dv_choice[:,stationid==i].mean(axis=1) for i in range(nstation)]).transpose().flatten()[xi_idx_flat]
               
obj_multiplier = T.dscalar('obj_multiplier')
lagrange_multiplier = T.dvector('lagrange_multiplier')
lagrange = obj_multiplier*obj + T.dot(lagrange_multiplier,pstation)

constr = theano.function([theta], pstation)
jac = theano.function([theta], T.jacobian(pstation, [theta])[0])
hess_constr = theano.function([theta, lagrange_multiplier, obj_multiplier], 
                              theano.gradient.hessian(lagrange, [theta])[0])

ntheta1 = nalpha + nbeta + nallsigma
nxifull = (nchoice-1)*nstation
mask00 = np.ones((ntheta1, ntheta1+nxi), dtype = bool)
mask10 = np.ones((nxi, ntheta1), dtype = bool)
mask11 = np.tile(np.eye(nstation, dtype = bool), (nchoice-1, nchoice-1))[xi_idx_flat,:][:,xi_idx_flat]

maskj = np.hstack((mask10, mask11))
maskh = np.vstack((mask00, maskj))

nvar = theta0.size
ncon = pstationtrue.size

x_L = np.array([pyipopt.NLP_LOWER_BOUND_INF]*nvar, dtype=float)
x_U = np.array([pyipopt.NLP_UPPER_BOUND_INF]*nvar, dtype=float)        
g_L = g_U = pstationtrue
nnzj = maskj.sum()
nnzh = maskh.sum()    
idxrj, idxcj = np.mgrid[:ncon,:nvar]
idxrh, idxch = np.mgrid[:nvar,:nvar]    

eval_j = lambda t, f: (idxrj[maskj], idxcj[maskj]) if f else jac(t)[maskj]
eval_h = lambda t, l, o, f: (idxrh[maskh], idxch[maskh]) if f else hess_constr(t,l,o)[maskh]

def solve_constr(theta0, use_hess = False):
    pyipopt.set_loglevel(1)    
    if use_hess:
        nlp = pyipopt.create(theta0.size, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, grad, constr, eval_j, eval_h)
    else:
        nlp = pyipopt.create(theta0.size, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, grad, constr, eval_j)
    results = nlp.solve(theta0)
    nlp.close()
    return results
    
#%%
def findiff(f, x):
    dx = 1e-5*abs(x)
    dx[dx <1e-7] = 1e-7
    
    df = []
    for i in range(x.size):
        x1 = np.array(x)
        x2 = np.array(x)
        
        x1[i] -= dx[i]
        x2[i] += dx[i]
        
        df.append((f(x2)-f(x1)).flatten()/(2*dx[i]))
        
    return np.stack(df)
    
def solve_unconstr(theta0):
    pyipopt.set_loglevel(1)
    thetahat , _, _, _, _, fval = pyipopt.fmin_unconstrained(
        eval_f,
        theta0,
        fprime=grad,
        fhess=hess,
    )
    return thetahat

#%%

def contraction(theta):
    x = np.array(theta)
    xihat = np.array(x[-nxi:])
    
    toler = 1e-13
    convergent = False
    share = lambda t: np.squeeze(constr(t))
    
    i = 1
    
    error = 1
        
    while not convergent:
        xihatold = np.array(xihat)
        x[-nxi:] = xihat
        
        ss = share(x)
        ss[ss<1e-40]=1e-40
#        xihat -= 0.2*(np.log(ss) - np.log(pstationtrue.flatten()))
        xihat -= 0.9*(norm.ppf(ss/(1+ss)) - norm.ppf(pstationtrue/(1+pstationtrue)))
        error = np.abs(xihat-xihatold).max()
        print xihat-xihatold
        print "iter", i, "error =", error
        convergent = error < toler
                
        i += 1
        if i > 5000:
            break
        
    return x


#%%

#thetahat = solve_unconstr(theta0)
#rrr = solve_constr(thetahat)

#thetahat2 = solve_constr(thetahat)
#pyipopt.set_loglevel(1)
#thetahat , _, _, _, _, fval = pyipopt.fmin_unconstrained(
#    eval_f,
#    theta0,
#    fprime=eval_grad,
#    fhess=eval_hess,
#    )
#


