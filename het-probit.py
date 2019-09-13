import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import theano.gradient
import theano.tensor.slinalg
# import pyipopt
import ipyopt as pyipopt
import scipy
import json
import sys

specname = sys.argv[1]
#specname = 'spec2het'
purpose = 'solve'

subsample = None

with open(specname + '.json', 'r') as f:
    spec = json.load(f)
    inputfile = spec['inputfile']
    lbls = spec['label']
    Xlbls = lbls['X']
    Xplbls = lbls['Xp']
    pricelbls = lbls['price']
    grouplbls = lbls['group']
    
    use_fe = False
    use_share_moments = False
    if 'settings' in spec:
        settings = spec['settings']
        if 'use_fe' in settings:
            use_fe = settings['use_fe']
        if 'use_share_moments' in settings:
            use_share_moments = settings['use_share_moments']

    if "subsample" in spec:
        subsample = spec["subsample"]
    
print(use_fe)
with open(inputfile, 'rb') as fi:
    df = pd.read_stata(fi)

# convert indexing variables to int
df.treattype = df.treattype.astype(int)
df.choice = df.choice.astype(int)
df.consumerid = df.consumerid.astype(int)
df.stationid = df.stationid.astype(int)

# old coding: 2 = midgrade gasoline, 3 = ethanol
# new coding: 2 = ethanol, 3 = midgrad gasoline
choice = df.choice.as_matrix()
df.loc[choice==3, 'choice']=2
df.loc[choice==2, 'choice']=3

# drop RJ, drop midgrade ethanol and treatment 3 and 4
df = df[df.dv_rj==0]
df = df[df.choice < 4]
df = df.loc[df.treattype < 3]

# df['treat1'] = df.treattype == 1
# df['ntreat1'] = df.groupby(['date', 'stationid']).treat1.transform(sum)
# df = df[df.ntreat1 > 0]


df['const'] = 1

# impute missing prices
df['pgmidgrade_km_adj'].fillna(value=1e9, inplace=True)
df['pemidgrade_km_adj'].fillna(value=1e9, inplace=True)

# relative prices
df['rel_lpgmidgrade_km_adj'] = np.log(df.pgmidgrade_km_adj) - np.log(df.pg_km_adj)
df['rel_lpe_km_adj'] = np.log(df.pe_km_adj) - np.log(df.pg_km_adj)

df['ltank'] = np.log(df.car_tank)

# car brand dummies
df['gm']   = df.car_make == "GM"
df['vw']   = df.car_make == "VW"
df['fiat'] = df.car_make == "FIAT"
df['ford'] = df.car_make == "FORD"

# car class
df['dv_carclass_compact'] = df.car_class == "Compact"
df['dv_carclass_fullsize'] = df.car_class == "Fullsize"
df['dv_carclass_midsize'] = df.car_class == "Midsize"
df['dv_carclass_minivan'] = df.car_class == "Minivan"
df['dv_carclass_suv'] = df.car_class == "SUV"
df['dv_carclass_smalltruck'] = df.car_class == "Smalltruck"
df['dv_carclass_subcompact'] = df.car_class == "Subcompact"

df['car_age'] = 2012 - df.car_model_year
df['car_lprice'] = np.log(df.car_price_adj)

df['dv_carpriceadj_p0p75'] = 1 - df['dv_carpriceadj_p75p100']
df['dv_usageveh_p0p75'] = 1 - df['dv_usageveh_p75p100']
df['dv_nocollege'] = 1 - df['dv_somecollege']

df['p_ratio'] = df['pe_lt']/df['pg_lt']
df['e_favor'] = df['p_ratio'] > 0.705

if subsample is not None:
    df = df.loc[df[subsample]==1,:]
print('Number of observations {}'.format(len(df)))

# generate day of week dummies
dow_dummies = pd.get_dummies(df['date'].dt.dayofweek, prefix='dv_dow')
df[dow_dummies.columns[1:]] = dow_dummies[dow_dummies.columns[1:]]

#%%

choice  = df['choice'].as_matrix()
price   = df.loc[:, pricelbls].as_matrix().transpose()
X       = df.loc[:, Xlbls].as_matrix().transpose()
Xp      = df.loc[:, Xplbls].as_matrix().transpose()
groupid = df.loc[:, grouplbls].as_matrix().transpose()

nX, nXp, nchoice  = len(Xlbls), len(Xplbls), len(pricelbls) + 1

nalpha  = nXp
nbeta   = nX*(nchoice-1)
nsigma  = (nchoice-1)*nchoice//2 - 1

nobs    = df.shape[0]
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

np.random.seed(1234)

floatX = 'float64'
theta  = T.dvector('theta')

# unpack parameters
offset = 0
alpha  = theta[offset: offset + nalpha]

offset += nalpha
beta   = theta[offset: offset + nbeta].reshape((nchoice-1, nX))

offset += nbeta
print(nallsigma)
print(theta[offset:offset+nallsigma])
sraw = T.concatenate([[1], theta[offset: offset + nallsigma], [0]])
S = sraw[tril_index_matrix]

Tprice  = theano.shared(price.astype(floatX),  name='price')
TX      = theano.shared(X.astype(floatX),  name='X')
TXp     = theano.shared(Xp.astype(floatX), name='Xp')

if use_fe:
    offset += nallsigma
    xiraw = T.concatenate([theta[offset:offset+nxi], [-10000.]])
    xi = xiraw[xi_idx]
    V = T.dot(alpha,TXp)*Tprice + T.dot(beta,TX) + xi[:,stationid]
else:
    V = T.dot(alpha,TXp)*Tprice + T.dot(beta,TX)

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

nlogl_i = -T.log(prob0) - T.log(prob1.sum(axis=0)/ndraws)
nlogl = nlogl_i.sum()

obj = nlogl
eval_f = theano.function([theta], outputs = obj)
grad = theano.function([theta], outputs = T.grad(obj, [theta]))
hess = theano.function([theta], outputs = theano.gradient.hessian(obj, [theta]))

eval_grad = lambda t: np.squeeze(grad(t))
eval_hess = lambda t: np.squeeze(hess(t))


#%%
#alpha0 = [-5.63116686]
#beta0 = [-0.14276449833885524, 0.07799550939333146, 0.0690886479616759, -0.031114683614026983, -0.09391731389704802, -0.1269116325321836, -0.09564480677074452, -0.035482238485123836, -0.050698241761471995,# -0.03731223127056641, -0.7783360705348067, -0.5328394135746228, 1.6107622200281881, 
#         -0.1383741971290979, 0.16894742379408748, 0.33464904615230423, 0.5473575675980583, 0.0022791624344226727, 0.12501040929703963, 0.1474707888708112, 0.10599018593441098, -0.051455999185487045]#, -0.33470501668838093, -0.5669505382552235, -0.7647587714144124, 0.17373775908415154]
#S0 = [0.4389639,1.07280456,1,0.4389639,1.07280456,1,0.4389639,1.07280456]
#      #1,0.4389639,1.07280456,1,0.4389639,1.07280456,1,0.4389639,1.07280456,
#      #1,0.4389639,1.07280456,1,0.4389639,1.07280456,1,0.4389639,1.07280456]
#theta0 = np.concatenate([alpha0, beta0, S0])
#
#if use_fe:
#    xi0 = np.zeros((nxi,))
#    theta0 = np.hstack([theta0, xi0])

S0 = np.tril(np.ones(nchoice-1))*0.5 + np.tril(np.eye(nchoice-1))*0.7
S0 = np.tile(S0, (ngroup,1,1))[tril_index][1:]

theta0 = np.zeros((nalpha + nbeta,))
theta0 = np.hstack([theta0, S0])
if use_fe:
    theta0 = np.hstack([theta0, np.zeros((nxi,))])
#%%

Vallbase        = T.dot(M, V)
p0allbase       = T.maximum(normcdf(-Vallbase[:,0,:]/c00[:,groupid]), 1e-8)
#drawsallbase    = np.random.random((ndraws,nchoice,nobs))
drawsallbase    =  (np.tile(np.arange(ndraws), (nobs,nchoice,1)).transpose() + 0.5)/ndraws
draws1allbase   = norminv(drawsallbase*p0allbase)
p1allbase    = normcdf(-(Vallbase[:,1,:] + c10[:,groupid]*draws1allbase)/c11[:,groupid]).mean(axis=0)

pallbase = p0allbase*p1allbase

if use_fe and use_share_moments:    
    pstation = T.stack([pallbase[1:,np.where(stationid==i)[0]].mean(axis=1) for i in range(nstation)]).transpose().flatten()[(~nuisancexi).flatten().nonzero()[0]]
    pstationtrue = np.stack([dv_choice[1:,stationid==i].mean(axis=1) for i in range(nstation)]).transpose().flatten()[~nuisancexi.flatten()]
                   
    obj_multiplier = T.dscalar('obj_multiplier')
    lagrange_multiplier = T.dvector('lagrange_multiplier')
    lagrange = obj_multiplier*obj + (lagrange_multiplier*pstation).sum()
    
    constr = theano.function([theta], pstation)
    jab = theano.function([theta], T.jacobian(pstation, [theta]))
    hess_constr = theano.function([theta, lagrange_multiplier, obj_multiplier], 
                                  outputs=theano.gradient.hessian(lagrange, [theta]))
    
    ntheta1 = nalpha + nbeta + nallsigma
    nxifull = (nchoice-1)*nstation
    mask00 = np.ones((ntheta1, ntheta1), dtype = bool)
    mask01 = np.ones((ntheta1, nxi), dtype = bool)
    mask10 = np.ones((nxi, ntheta1), dtype = bool)
    mask11 = np.tile(np.eye(nstation, dtype = bool), (nchoice-1, nchoice-1))[~nuisancexi.flatten(),:][:,~nuisancexi.flatten()]
    
    maskj = np.hstack((mask10, mask11))
    maskh = np.hstack((np.vstack((mask00, mask10)), np.vstack((mask01, mask11))))
    
    def solve_constr(theta0, use_hess = False):
        pyipopt.set_loglevel(1)    
        n = theta0.size    
        x_L = np.array([pyipopt.NLP_LOWER_BOUND_INF]*n, dtype=float)
        x_U = np.array([pyipopt.NLP_UPPER_BOUND_INF]*n, dtype=float)        
        ncon = pstationtrue.size
        g_L = g_U = pstationtrue    
        nnzj = maskj.sum()
        nnzh = maskh.sum()    
        idxrj, idxcj = np.mgrid[:ncon, :n]
        idxrh, idxch = np.mgrid[:n, :n]    
        eval_c = lambda t: constr(t)
        eval_j = lambda t, f: (idxrj[maskj], idxcj[maskj]) if f else np.squeeze(jab(t))[maskj]
        if use_hess:
            eval_h = lambda t, l, o, f: (idxrh[maskh], idxch[maskh]) if f else np.squeeze(hess_constr(t,l,o))[maskh]
            nlp = pyipopt.create(theta0.size, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad, eval_c, eval_j, eval_h)
        else:
            nlp = pyipopt.create(theta0.size, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad, eval_c, eval_j)
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
    

def _eval_g(_X, _out):
    return

def _eval_jac_g(_X, _out):
    return

_eval_jac_g.sparsity_indices = (numpy.array([]), numpy.array([]))

_eval_h.sparsity_indices = tuple(map(np.array, zip(*[[i,j] for i in range(theta0.size) for j in range(i+1)])))

def solve_unconstr(theta0):
    pyipopt.set_loglevel(1)
    x_L = np.array([pyipopt.NLP_LOWER_BOUND_INF]*n, dtype=float)
    x_U = np.array([pyipopt.NLP_UPPER_BOUND_INF]*n, dtype=float)        
    ncon = 0
    g_L = numpy.array([], dtype=float)
    g_U = numpy.array([], dtype=float)

    nlp = pyipopt.Problem(theta0.size, x_L, x_U, ncon, g_L, g_U, _eval_jac_g.sparsity_indices, _eval_h.sparsity_indices, eval_f, eval_grad, _eval_g, _eval_jac_g, eval_hess)
    # thetahat , _, _, _, _, fval = pyipopt.solve(
    #     eval_f,
    #     theta0,
    #     fprime=eval_grad,
    #     fhess=eval_hess,
    # )
    thetahat, _, _ = nlp.solve()
    
    return thetahat

#%%
eval_mean_pallbase = theano.function([theta], pallbase.mean(axis=1))
eval_dmean_pallbase = theano.function([theta], T.jacobian(pallbase.mean(axis=1),[theta])[0])
def cal_marginal_effect(thetahat, TX, X1, X2):
    X0 = TX.get_value()
    TX.set_value(X1)
    share1 = eval_mean_pallbase(thetahat)
    dshare1 = eval_dmean_pallbase(thetahat)
    TX.set_value(X2)
    share2 = eval_mean_pallbase(thetahat)
    dshare2 = eval_dmean_pallbase(thetahat)
    TX.set_value(X0)
    effecthat = share2 - share1
    deffect = dshare2 - dshare1
    coveffect = np.dot(np.dot(deffect, covhat), deffect.transpose())
    effectse = np.sqrt(np.diag(coveffect))
    return [effecthat, effectse]
    
def cal_marginal_effect_dummies(thetahat, dummyset):
    X1df = df.loc[:, Xlbls].copy()
    for d in dummyset: X1df[d] = 0.0
    X1 = X1df.as_matrix().astype(np.float64).transpose()
    mfx = {}
    for d in dummyset:
        X2df = X1df.copy()
        X2df[d] = 1.0
        X2 = X2df.as_matrix().astype(np.float64).transpose()
        print('mfx of ' + d)
        mfx[d] = cal_marginal_effect(thetahat, TX, X1, X2)
    return mfx
    
def cal_marginal_effect_continuous(thetajat, varset):
    mfx = {}
    for varname, val in varset.items():
        if varname in Xlbls:
            X1df = df.loc[:, Xlbls].copy()
            tx = TX
        else:
            X1df = df.loc[:, pricelbls].copy()
            tx = Tprice
        X1 = X1df.as_matrix().astype(np.float64).transpose()
        X2df = X1df.copy()
        X2df[varname] += val
        X2 = X2df.as_matrix().astype(np.float64).transpose()
        mfx[varname] = cal_marginal_effect(thetahat, tx, X1, X2)
        mfx[varname][0] /= val
        mfx[varname][1] /= val
    return mfx

#%%
if purpose != 'mfx':
    thetahat = solve_unconstr(theta0)
        
    if use_fe and use_share_moments:
        result = solve_constr(thetahat)
        thetahat2 = result[0]
    
    with open('./results/' + specname + '_result.json', 'w') as outfile:
        json.dump({'thetahat':thetahat.tolist(), 'specname': specname}, outfile, indent=2)
else:
    with open('./results/' + specname + '_result.json', 'r') as outfile:
    	results = json.load(outfile)
    	thetahat = np.array(results['thetahat'])

    
#%%
#
#
#eval_pallbase = theano.function([theta], pallbase)
#
##%%
#
#offset = nalpha + nbeta
#Sidx = range(offset, offset + nallsigma)
#Shat = thetahat[Sidx]
#S_split = np.split(np.hstack([[1], Shat]), nsigma+1)
#
#pratiocf = np.linspace(0.7, 1.3)
#pcfmean = []
#
#for pr in pratiocf:
#    pricecf= np.array(price)
#    pricecf[0,:] = np.log(pr)
#    Tprice.set_value(pricecf)
#    pcf = []
#    for g in range(ngroup):
#        Scf = np.tile(S_split[g], ngroup)
#        thetacf = np.array(thetahat)
#        thetacf[Sidx] = Scf[1:]
#        thetacf /= Scf[0]
#        
#        pcf.append(eval_pallbase(thetacf))
#        
#    pcfmean.append([pp.mean(axis=1) for pp in pcf])
#    
#pcfmeannp = np.array(pcfmean)
#
##%%
#
#import matplotlib.pyplot as plt
#
#plt.plot(pratiocf, pcfmeannp[:,0,0])
#plt.plot(pratiocf, pcfmeannp[:,1,0])
#plt.show()
#
#plt.plot(pratiocf, pcfmeannp[:,1,0] - pcfmeannp[:,0,0])
#plt.show()

#%%
covhat = np.linalg.pinv(eval_hess(thetahat))
lnprob = T.log(prob0) + T.log(prob1.mean(axis=0))
#dlnprob = T.jacobian(lnprob, [theta])[0]

jacobian = theano.gradient.jacobian(nlogl_i, theta)
eval_jab = theano.function([theta], jacobian)
Jhat = eval_jab(thetahat)

GG = Jhat.transpose().dot(Jhat)
GGclustered = np.zeros_like(GG)
for stid in df.stationid.unique():
    Jsubhat = Jhat[(df.stationid==stid).nonzero()].sum(axis=0)
    GGclustered += np.outer(Jsubhat, Jsubhat)

covhatclustered = np.matmul(covhat, np.matmul(GGclustered, covhat))

covhat = covhatclustered
#%%
#G = theano.function([theta], dlnprob)(thetahat)
#covhat = np.linalg.pinv(np.dot(G.transpose(), G))
sehat = np.sqrt(np.diag(covhat))
tstat = thetahat/sehat

#%%
#marginal_effect_set = {
#    'female': [{'dv_female':0}, 
#               {'dv_female':1}],
#    'college': [{'dv_somesecondary':0, 'dv_somecollege':0},
#                {'dv_somesecondary':0, 'dv_somecollege':1}]}

if 'marginal_effect_set' in spec:
    marginal_effect_set = spec['marginal_effect_set']                
    marginal_effects = {}
    for mfxcase in marginal_effect_set:
        if mfxcase['type'] == 'dv':
            marginal_effects.update(cal_marginal_effect_dummies(thetahat, mfxcase['vars']))
        else:
            marginal_effects.update(cal_marginal_effect_continuous(thetahat, mfxcase['vars']))
            
    X1df = df.loc[:, pricelbls].copy()
    X2df = X1df.copy()
    X2df -= 0.01
    X1 = X1df.as_matrix().astype(np.float64).transpose()
    X2 = X2df.as_matrix().astype(np.float64).transpose()
    ehat, ese = cal_marginal_effect(thetahat, Tprice, X1, X2)
    marginal_effects['rel_lpg_km_adj'] = [ehat/0.01, ese/0.01]
    
    marginal_effects_serialized = {}
    for k, v in marginal_effects.items():
        marginal_effects_serialized[k] = [x.tolist() for x in v]

    with open('./results/' + specname + '_mfx.json', 'w') as outfile:
        json.dump(marginal_effects_serialized, outfile, indent=2)


#%%
#if use_fe and use_share_moments:
#    dnlogf = T.grad(nlogl, [theta])[0]
#    jac = T.jacobian(pstation, [theta])[0].transpose()
#    dnlogfstar = dnlogf[:ntheta1] - T.dot(T.dot(jac[:ntheta1,:],T.nlinalg.matrix_inverse(jac[ntheta1:,:])), dnlogf[ntheta1:])
#    d2nlogfstar = T.jacobian(dnlogfstar, [theta])[0]    
#    d2nlogfstarf = theano.function([theta], d2nlogfstar)

def get_stat(f, thetahat):
    fhat = theano.function([theta], f)(thetahat)
    dfhat = theano.function([theta], T.jacobian(f, [theta])[0])(thetahat)
    fhatcov = np.dot(np.dot(dfhat, covhat), dfhat.transpose())
    try:
        fse = np.sqrt(np.diag(fhatcov))
    except:
        fse = np.sqrt(fhatcov)
    ftstat = fhat/fse
    return fhat, fse, ftstat
    
#%%

alphahat, alphase, alphatstat = get_stat(alpha, thetahat)
betahat, betase, betatstat = get_stat(beta.flatten(), thetahat)

i1 = np.zeros(ngroup*(nchoice-1)-1, dtype=int) # base alternative = 0
i2 = np.tile(np.arange(ngroup, dtype=int), nchoice-1)[1:] # groupid
i3 = np.repeat(np.arange(nchoice-1, dtype=int), ngroup)[1:] # variance (diagonal element of each choice)
iii = (i1,i2,i3,i3)
Sigmamain = Sigma[iii]

mm = np.hstack((-np.ones((2,1)), np.eye(2)))
mm = scipy.linalg.block_diag(np.eye(2), *([mm]*(ngroup*2//3-1)))
effect = T.dot(mm, T.log(Sigmamain))

Sigmahat, Sigmase, Sigmatstat = get_stat(Sigmamain, thetahat)
effecthat, effectse, effecttstat = get_stat(effect, thetahat)

choicelbls = ['ethanol', 'midgrade gasoline']

formatstr = "%30s%10.3f%10.3f%10.3f"
formatstr2 = "%30s%10.3f%10s%10s"
divider = '*'*(30+10+10+10)
divider2 = '-'*(30+10+10+10)
print(divider)
print("%30s%10s%10s%10s" % ('', 'coeff', 'se', 't'))
print(' '*30 + '-'*30)
print('*** price sensitiviy ***')
for i in range(len(Xplbls)):
    print(formatstr % (Xplbls[i], thetahat[i], sehat[i], tstat[i]))
print(divider)

def print_result_row(coeff, se, t, label):
    print(formatstr % (label, coeff, se, t))
    
def print_result_group(grouplabel, coeffs, ses, ts, labels):
    print(grouplabel)
    for i in range(len(coeffs)):
        print_result_row(coeffs[i], ses[i], ts[i], labels[i])
    print(divider2)
    
for j in range(nchoice-1):
    idx = range(j*nX, (j+1)*nX)
    print_result_group("utiltiy of " + choicelbls[j],
                       betahat[idx], betase[idx], betatstat[idx], Xlbls)

print_result_group("variance of random utility, " + choicelbls[0],
                   Sigmahat[:ngroup-1], Sigmase[:ngroup-1], Sigmatstat[:ngroup-1],
                   ["Treatment " + str(j) for j in range(1,ngroup)])

for j in range(1,nchoice-1):
    idx = range(j*ngroup-1, (j+1)*ngroup-1)
    print_result_group("variance of random utility, " + choicelbls[j],
                       Sigmahat[idx], Sigmase[idx], Sigmatstat[idx],
                       ["Treatment " + str(j) for j in range(ngroup)])

for j in range(nchoice-1):
    idx = range(j*(ngroup*2//3), (j+1)*(ngroup*2//3))
    print_result_group("effect on log(variance of random utility, " + choicelbls[j] + ")",
                       effecthat[idx], effectse[idx], effecttstat[idx],
                       ["Treatment " + str(j) for j in range(1,ngroup)])

