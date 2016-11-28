import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import theano.gradient
import theano.tensor.slinalg
import pyipopt

from scipy.stats import norm

inputfile = '../data/data_new_lp.csv'
df = pd.read_csv(inputfile)

#df = df[df['stationid'] < 200]
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
    
    return alpha, beta, S

floatX = 'float64'

theta  = T.dvector('theta')
alpha, beta, S = getparams(theta)

xi = T.dvector('xi')
xifull = T.concatenate([xi, [-10000.]])[xi_idx]
    
#Tprice  = theano.shared(price.astype(floatX),  name='price')
#TX      = theano.shared(X.astype(floatX),  name='X')
#TXp     = theano.shared(Xp.astype(floatX), name='Xp')
#TXhet   = theano.shared(Xhet.astype(floatX), name='Xhet')

V       = T.dot(alpha,Xp)*price + T.dot(beta,X) + xifull[:,stationid]
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

def normcdf(x):
    return 0.5 + 0.5*T.erf(x/np.sqrt(2))
                       
def norminv(p):
    return np.sqrt(2)*T.erfinv(2*p-1)
    
ndraws = 10
draws = np.random.random((ndraws,nobs))

prob0 = normcdf(-Vnonchoice[0,:]/c00[iii])
draws1 = norminv(draws*prob0)
prob1 = normcdf(-(Vnonchoice[1,:] + c10[iii]*draws1)/c11[iii])

nlogl = -T.log(prob0).sum() - T.log(prob1.sum(axis=0)/ndraws).sum()
#nlogl = -T.log(prob0).sum()
    
nlogllogit = T.log(T.sum(T.exp(Vfull), axis=0)).sum() - (Vchoice).sum()

obj = nlogl


#%%
alpha0 = [-5.63116686]
beta0 = [-0.14276449833885524, 0.07799550939333146, 0.0690886479616759, -0.031114683614026983, -0.09391731389704802, -0.1269116325321836, -0.09564480677074452, -0.035482238485123836, -0.050698241761471995, -0.03731223127056641, -0.7783360705348067, -0.5328394135746228, 1.6107622200281881, 
         -0.1383741971290979, 0.16894742379408748, 0.33464904615230423, 0.5473575675980583, 0.0022791624344226727, 0.12501040929703963, 0.1474707888708112, 0.10599018593441098, -0.051455999185487045, -0.33470501668838093, -0.5669505382552235, -0.7647587714144124, 0.17373775908415154]
gamma0 = [-0.10, -0.05]
S0 = [0.4389639,1.07280456,1,0.4389639,1.07280456,1,0.4389639,1.07280456]
xi0 = np.zeros((nxi,))
theta0 = np.concatenate([alpha0, beta0, S0])

#%%

Vallbase        = T.dot(M, V)
p0allbase       = normcdf(-Vallbase[:,0,:]/c00[:,groupid])
drawsallbase    = np.random.random((ndraws,nchoice,nobs))
draws1allbase   = norminv(drawsallbase*prob0)
p1allbase    = normcdf(-(Vallbase[:,1,:] + c10[:,groupid]*draws1allbase)/c11[:,groupid]).mean(axis=0)

pallbase = p0allbase*p0allbase

share = T.stack([pallbase[1:,np.where(stationid==i)[0]].mean(axis=1) for i in range(nstation)]).transpose().flatten()[(~nuisancexi).flatten().nonzero()[0]]
sharetrue = np.stack([dv_choice[1:,stationid==i].mean(axis=1) for i in range(nstation)]).transpose().flatten()[~nuisancexi.flatten()]

dnll = T.grad(nlogl, [theta, xi])
dshare = T.jacobian(share, [theta,xi])
dnllopt = dnll[0] - T.dot(T.dot(dshare[0].transpose(), T.nlinalg.matrix_inverse(dshare[1])), dnll[1])


#%%

eval_share = theano.function([theta, xi], share)

def contraction(theta, xi0):
    xi1 = np.array(xi0)
    
    toler = 1e-13
    convergent = False    
    i = 1
            
    while not convergent:
        sharesim = eval_share(theta, xi1)
        sharesim[sharesim<1e-40]=1e-40
#        xihat -= 0.2*(np.log(ss) - np.log(pstationtrue.flatten()))
        decr = (norm.ppf(sharesim/(1+sharesim)) - norm.ppf(sharetrue/(1+sharetrue)))
        error = np.abs(decr).max()
#        print xihat-xihatold
#        print "iter", i, "error =", error
        convergent = error < toler
        
        xi1 -= decr
                
        i += 1
        if i > 5000:
            break
        
    print 'iter =', i, 'error =', error
        
    return xi1
    
#%%

station_xi = np.repeat(np.arange(nstation), nchoice-1)[~nuisancexi.flatten()]
samestationmask = station_xi == station_xi.reshape((nxi, 1))
samestationidx = np.zeros(samestationmask.shape, dtype = int) + samestationmask.sum()
samestationidx[samestationmask] = np.arange(samestationmask.sum())

sscontr = T.sum((T.log(share) - np.log(sharetrue))**2)
dsscontr = T.grad(sscontr, [xi])[0]
hsscontr = theano.gradient.hessian(sscontr, [xi])[0]

sscontrf = theano.function([theta, xi], sscontr)
dsscontrf = theano.function([theta, xi], dsscontr)
hsscontrf = theano.function([theta, xi], hsscontr)

def solve_xi(theta0, xi0):
    pyipopt.set_loglevel(0)
    return pyipopt.fmin_unconstrained(lambda x: sscontrf(theta0, x), 
                                      xi0, 
                                      fprime=lambda x: dsscontrf(theta0, x), 
                                      fhess=lambda x: hsscontrf(theta0, x))[0]
 
     
#%%

constr = T.log(share)
jac = T.jacobian(constr, [xi])[0]

maskj = np.tile(np.eye(nstation, dtype = bool), (nchoice-1, nchoice-1))[~nuisancexi.flatten(),:][:,~nuisancexi.flatten()]

nnzj = nnzh = maskj.sum()
idxrj, idxcj = np.mgrid[:nxi, :nxi]
rj, cj = idxrj[maskj], idxcj[maskj]

constrf = theano.function([theta, xi], constr)
jacf = theano.function([theta, xi], jac)

def solve_xi_constr(theta0, xi0):
    pyipopt.set_loglevel(0)
    eval_c = lambda x: constrf(theta0, x)
    eval_j = lambda x, f: (rj, cj) if f else jacf(theta0, x)[maskj]
    x_L = np.array([pyipopt.NLP_LOWER_BOUND_INF]*nxi, dtype=float)        
    x_U = np.array([pyipopt.NLP_UPPER_BOUND_INF]*nxi, dtype=float)        
    g_L = g_U = np.log(sharetrue)
    nlp = pyipopt.create(nxi, x_L, x_U, nxi, g_L, g_U, nnzj, nnzh, lambda x: 0, lambda x: np.zeros((nxi,)), eval_c, eval_j)
    results = nlp.solve(xi0)
    nlp.close()
    return results
    


#%%

eval_nlogl = theano.function([theta, xi], nlogl)
eval_dnllopt = theano.function([theta, xi], dnllopt)

eval_f = lambda t: eval_nlogl(t, solve_xi(t, xi0))
eval_g = lambda t: eval_dnllopt(t, solve_xi(t, xi0))

def solve_unconstr(theta0):
    pyipopt.set_loglevel(1)
    thetahat , _, _, _, _, fval = pyipopt.fmin_unconstrained(
        eval_f,
        theta0,
        fprime=eval_g,
        fhess=None,
    )
    return thetahat

#%%
solve_unconstr(theta0)


#%%

#thetahat = solve_unconstr(theta0)

#thetahat2 = solve_constr(thetahat)
#pyipopt.set_loglevel(1)
#thetahat , _, _, _, _, fval = pyipopt.fmin_unconstrained(
#    eval_f,
#    theta0,
#    fprime=eval_grad,
#    fhess=eval_hess,
#    )
#


