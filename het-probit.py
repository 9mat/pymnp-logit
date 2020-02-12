import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import theano.gradient
import theano.tensor.slinalg
import json
import sys

#%% READ INPUT AND SPECIFICATIONS

# Spec file:
#     specifications, input file and output are specified in a spec file
#     example of a spec file:
#     {
#       "inputfile": "path to input file",
#       "price": ["list price variables"],
#       "X": ["list of covariates in the mean utiltiy"],
#       "Xp": ["list of covariates the shift price sensitivity"],
#       "resultfile": "path to result file",
#       "other spec": "values"
#      }
#     Specify the path to the spec file in the first argument of the command
#     The program will ask for the spec file if no argument is given
specfile = sys.argv[1] if len(sys.argv) > 1 else input("Path to spec: ")


# purpose: 
#     specify whether to solve for estiamtes, to calculate the marginal effects
#     or to just display the results. purposed can be combined
#     specify purposes in the second argument of the command, or when prompted      
purpose = sys.argv[2] if len(sys.argv) > 2 else input("Purpose (solve/mfx/display): ")


# read the spec file
with open(specfile, 'r') as f:
    spec = json.load(f)
    
# unpack the spefications
    
# path to input file
inputfile = spec['inputfile'] if 'inputfile' in spec else input("Path to input: ")
    
# variable names that will be in the mean utility equation
Xlbls = spec['X']

# variable names that will be in the price sensitivity equation
Xplbls = spec['Xp']

# variable names that will be in the covariance equation
Xsigmalbls = spec['Xsigma'] if 'Xsigma' in spec else []

# names of (relative) price variables
pricelbls = spec['price']

# name of the treatment variable
grouplbls = spec['group']

# whether to use station-fuel fixed effects or not
use_fe = spec['use_fe'] if 'use_fe' in spec else False

# whether to use share moment (contraction mapping to solve for xi) or not
use_share_moments = spec['use_share_moment'] if 'use_share_moment' in spec else False

# whether to use a subsample instead of the full sample
subsample = spec['subsample'] if 'subsample' in spec else None

# whether to allow the covariance to shift with the price sensitivity
use_price_sensitity_in_sigma = ('price_sensitity_in_sigma' in spec and spec['price_sensitity_in_sigma'])


# read the input
with open(inputfile, 'rb') as fi:
    df = pd.read_stata(fi)

# convert some variables to integer
for x in ['treattype', 'choice', 'consumerid', 'stationid']:
    df[x] = df[x].astype(int)


# if using past price, need to drop missing values because of the gap weeks
all_Xlbls = Xlbls + Xplbls + Xsigmalbls
if any(x in all_Xlbls for x in ['dl_pedivpg', 'abs_dl_pedivpg', 'pos_dl_pedivpg', 'neg_dl_pedivpg']):
    df = df[~pd.isna(df.dl_pedivpg)]

# # old coding: 2 = midgrade gasoline, 3 = ethanol
# # new coding: 2 = ethanol, 3 = midgrad gasoline
# choice = df.choice.values
# if 'recoding' in spec and spec['recoding']:
#     df.loc[choice==3, 'choice']=2
#     df.loc[choice==2, 'choice']=3

# # drop RJ, drop midgrade ethanol and treatment 3 and 4
# df = df[df.dv_rj==0]
# df = df[df.choice < 4]
# df = df.loc[df.treattype < 3]


# df['const'] = 1

# # impute missing prices
# df['pgmidgrade_km_adj'].fillna(value=1e3, inplace=True)
# df['pemidgrade_km_adj'].fillna(value=1e3, inplace=True)

# # relative prices
# df['rel_lpgmidgrade_km_adj'] = np.log(df.pgmidgrade_km_adj) - np.log(df.pg_km_adj)
# df['rel_lpe_km_adj'] = np.log(df.pe_km_adj) - np.log(df.pg_km_adj)

# df['rel_pgmidgrade_km_adj'] = (df.pgmidgrade_km_adj) - (df.pg_km_adj)
# df['rel_pe_km_adj'] = (df.pe_km_adj) - (df.pg_km_adj)

# df['ltank'] = np.log(df.car_tank)

# # car brand dummies
# df['gm']   = df.car_make == "GM"
# df['vw']   = df.car_make == "VW"
# df['fiat'] = df.car_make == "FIAT"
# df['ford'] = df.car_make == "FORD"

# # car class
# df['dv_carclass_compact'] = df.car_class == "Compact"
# df['dv_carclass_fullsize'] = df.car_class == "Fullsize"
# df['dv_carclass_midsize'] = df.car_class == "Midsize"
# df['dv_carclass_minivan'] = df.car_class == "Minivan"
# df['dv_carclass_suv'] = df.car_class == "SUV"
# df['dv_carclass_smalltruck'] = df.car_class == "Smalltruck"
# df['dv_carclass_subcompact'] = df.car_class == "Subcompact"

# df['car_age'] = 2012 - df.car_model_year
# df['car_lprice'] = np.log(df.car_price_adj)
# df['car_lprice_normalized'] = df.car_lprice - df.car_lprice.mean()

# df['dv_carpriceadj_p0p75'] = 1 - df['dv_carpriceadj_p75p100']
# df['dv_usageveh_p0p75'] = 1 - df['dv_usageveh_p75p100']
# df['dv_nocollege'] = 1 - df['dv_somecollege']
# df['dv_atleast_secondary'] = df.dv_somecollege+df.dv_somesecondary

# df['p_ratio'] = df['pe_lt']/df['pg_lt']
# df['e_favor'] = df['p_ratio'] > 0.705
 

# if 'dl_pedivpg' in df:
#     df['abs_dl_pedivpg'] = np.abs(df.dl_pedivpg)
#     df['pos_dl_pedivpg'] = np.maximum(0, df.dl_pedivpg)
#     df['neg_dl_pedivpg'] = np.maximum(0, -df.dl_pedivpg)

# if `subsample` is not None, use only the corresponding subsample
# if `subsample` starts with `~`, use the complement subsample instead
# e.g. subsample = dv_somecollege to use college graduates
# while subsample = ~dv_somecoollege to use non-college
if subsample is not None:
    if subsample.startswith("~"):
        df = df.loc[df[subsample[1:]]==0,:]
    else:
        df = df.loc[df[subsample]==1,:]
        
print('Number of observations {}'.format(len(df)))

# generate day of week dummies
dow_dummies = pd.get_dummies(df['date'].dt.dayofweek, prefix='dv_dow')
df[dow_dummies.columns[1:]] = dow_dummies[dow_dummies.columns[1:]]

group_dummies = pd.get_dummies(df[grouplbls], prefix='treat', drop_first=True)
df[group_dummies.columns] = group_dummies


# covariates that shift the covariance matrix
# the covariates include variables specified in Xsigmalbls
# and their interactions with the treatement dummies
Xsigmatreatlbls = Xsigmalbls[:]
for x in group_dummies.columns:
    for y in Xsigmalbls:
        # interaction between Xsigma and treatment
        df[y+"*"+x] = df[x]*df[y] 
        Xsigmatreatlbls.append(y+"*"+x)
        
# the constant term for the base alternative's variance is not identified
# thus, the set of coveriates for it will not have const
Xsigmatreatlbls_noconst = [x for x in Xsigmatreatlbls if x != 'const']


#%% UNPACK DATA, PREPARE PARAMETERS


# unpack the data from the dataframe to individual vectors/matrices
# for ease of implementation
choice  = df['choice'].values
price   = df.loc[:, pricelbls].values.transpose()
X       = df.loc[:, Xlbls].values.transpose()
Xp      = df.loc[:, Xplbls].values.transpose()
Xsigma  = df.loc[:, Xsigmalbls].values.transpose()
groupid = df.loc[:, grouplbls].values.transpose()

# number of covariates in the mean utility and the price sensitivity equation
# and the number of alternatives
nX, nXp, nchoice  = len(Xlbls), len(Xplbls), len(pricelbls) + 1

# number of parameters to be estimated
# note: the covariance matrix sigma is (nchoice-1) by (nchoice-1) matrix
# the first variance is not identified
# thus, we have (nchoice-1)*nchoice//2 - 1 identified parameters in the matrix
nalpha  = nXp
nbeta   = nX*(nchoice-1)
nsigma  = (nchoice-1)*nchoice//2 - 1

nobs    = df.shape[0]
ngroup  = np.unique(groupid).size


# indicates whether to allow the price sensitivity to be heterogenous
use_dprice = nalpha > 1


ngamma = len(Xsigmalbls) + use_price_sensitity_in_sigma
ngamma_e = ngamma_m = ngamma_em = ngamma*(ngroup-1) + ngamma - use_price_sensitity_in_sigma
ngamma_e -= ('const' in Xsigmalbls)
 

stationidold = df['stationid'].astype(int).values
uniquestationid = np.unique(stationidold)
nstation = uniquestationid.shape[0]

stationid = np.zeros(stationidold.shape, dtype=int)
for i in range(nstation):
    stationid[stationidold==uniquestationid[i]] = i
    
dv_choice = np.arange(nchoice).reshape((nchoice,1)) == (choice-1)
nobsstation = np.array([(stationid==sid).sum() for sid in range(nstation)])


if use_fe:
    sumchoice_station = np.array([dv_choice[1:, stationid == i].sum(axis = 1) for i in range(nstation)]).transpose()
    nuisancexi = sumchoice_station == 0
    nxi = (1-nuisancexi).sum()
    
    xi_idx = np.zeros(nuisancexi.shape, dtype = int) + nxi
    xi_idx[~nuisancexi] = np.arange(nxi)
else:
    nxi = 0
    
theta0 = np.zeros((nalpha + nbeta + ngamma_e + ngamma_m + ngamma_em + 2 + nxi,))

if 'rel_pe_km_adj' in pricelbls:
    theta0[0] = -30

#%%

# indices to help pick out the elements in the S matrix to be estimated
# aka the all the elements in the lower triangle, except the first element,
# which is set to 1 to normalized the scale of the utilit

# There will be ngroup matrices of size (nchoice-1, nchoice-1)
# There will be nallsigma parameters to be estimated for these matrices
# The rest will be zero (the upper triangle except diagonal) or 1 (the first 
# element for first group)
# I map all the zero-elements to index (nallsigma+1), the one-element to
# index-zero, and the rest to index from 1 to nallsigma

# initialized with all (nallsigma+1) indices
#tril_index_matrix = np.zeros((ngroup+1, nchoice-1, nchoice-1), dtype=int) + nallsigma + 1

# list of tuples (g,j,k) -- g group, and j,k index of the lower triangles
#tril_index = tuple(np.vstack((np.repeat(np.arange(ngroup+1), nsigma+1), 
#              np.tile(np.tril_indices(nchoice-1), ngroup+1))).tolist())

# set the indices of the lower triangles
# for 3 groups, 3 choices, this matrix will be
#[[[0, 9],
#  [1, 2]],
# [[3, 9],
#  [4, 5]],
# [[6, 9],
#  [7, 8]]])
#tril_index_matrix[tril_index] = np.arange(nallsigma+1)

np.random.seed(1234)

#%% DEFINE THEANO DATA TO REPRESENT PARAMETERS, DATA, AND LIKELHOOD

# use double precision in all calculation
floatX = 'float64'

# vector of paremeters to be estimated
theta  = T.dvector('theta')

# unpack parameters
#   alpha: price sensisivity paramater ()
#   beta: choice-specific coefficients
#   sraw: parameters to construct the covariance matrix
#   xiraw: fuel-station fixed effects that can be identified

offset = 0
alpha  = theta[offset: offset + nalpha]

offset += nalpha
beta   = theta[offset: offset + nbeta].reshape((nchoice-1, nX))

offset += nbeta
gamma_e = theta[offset: offset + ngamma_e]

offset += ngamma_e
gamma_m = theta[offset: offset + ngamma_m]

offset += ngamma_m
gamma_em = theta[offset: offset + ngamma_em]

offset += ngamma_em
xiraw = theta[offset:offset+nxi] if use_fe else 0

# fuel-station fixed effects
if use_fe:
    xiraw_padded = T.concatenate([xiraw, [-10000.]])
    xi = xiraw_padded[xi_idx]
    xi_i = xi[:, stationid]
else:
    xi_i = 0

# To be used in the GHK simulator:
# transformation matrix M will change the base alternative to another base
# M consists of `nchoice` matrices of size `(nchoice-1, nchoice-1)`
# Suppose V is the mean utility using choice 0 as the base alternative 
# M_j*V will be the mean utility if using choice j as the base alternative
M = np.stack([[[1,0],[0,1]], [[-1,0],[-1,1]], [[0,-1],[1,-1]]])

# For each motorist, transform the mean utity using base-zero to the mean 
# utility using the chosen choice as the base
# Mi will be N matrices of size (nchoice-1, nchoice-1), each matrix for
# each motorist
Mi = M[choice-1, :, :]


# convert the data to theano data
Tprice  = theano.shared(price.astype(floatX),  name='price')
TX      = theano.shared(X.astype(floatX),  name='X')
TXp     = theano.shared(Xp.astype(floatX), name='Xp')


# mean utility equation
# V_ijt = alpha_i*p_jt + beta_j*X_ijt + xi_jt
alpha_i = T.dot(alpha, TXp)
V = alpha_i*Tprice + T.dot(beta,TX)  + xi_i

# For GHK simulator, we need to use the chosen choice as the base alternative
# for each individual, and subtract it from other non-choice 
# The transformation matrix M is used to carry out the subtraction
Vnonchoice = T.batched_dot(Mi,V.transpose()).transpose()  


if use_fe:
    # Due to the base-alternative transformation, we need to recalculate
    # the set of station-fuel FEs that cannot be identified
    # Assuming the first choice (gasoline) is always identified
    # Applying the same transformation M to the set of dummies for identification
    # will give use the new set of dummies for identification
    mask = np.einsum('ijk,ki->ji', Mi, nuisancexi[:,stationid]) == 0
else:
    # Even without fixed effects, we still have some stations with missing 
    # fuel, and the prices are imputed with a very high prices
    # The log likelihood of those will be zero, so either including then or not
    # will not affect identification
    # However, due to extreme values, these calculation can be unstable
    # so, if possible, we exclude these observations (where the price of a fuel 
    # is missing)
    mask = np.abs(np.einsum('ijk,ki->ji', Mi, price)) < 10

#%%
#esigma_11 = T.exp(sigma_11)
#var_zcontrol11 = sigma_10**2 + esigma_11**2
    

# covariates in the covariance equation
TXsigma = theano.shared(df[Xsigmatreatlbls].values.transpose().astype(floatX))

# covariates in the covariance equation, exclduing const
TXsigma_noconst = theano.shared(df[Xsigmatreatlbls_noconst].values.transpose().astype(floatX))

# group dummies
Tgroup_dummies = theano.shared(group_dummies.values.transpose().astype(floatX))

# add the price sensitivity of allowing price sentivity to shift the covariance matrix
TXsigmafull = TXsigma
TXsigmafull_noconst = TXsigma_noconst
if use_price_sensitity_in_sigma:
    TXsigmafull = T.concatenate([TXsigmafull, (alpha_i-alpha_i.mean())*Tgroup_dummies])
    TXsigmafull_noconst = T.concatenate([TXsigma_noconst, (alpha_i-alpha_i.mean())*Tgroup_dummies])
    

var_z00 = T.exp(gamma_e.dot(TXsigmafull_noconst))
var_z11 = T.exp(gamma_m.dot(TXsigmafull))
cov_z10 = T.tanh(gamma_em.dot(TXsigmafull))*T.sqrt(var_z00*var_z11)

# note that the above covariance matrix correponds to mean utiltity relative
# to the first alternative

#%%


# Cholesky decomposition for 2x2 matrix
# V = [v00 v01, v10 v11], S = [s00 0, s10 s11], S'S = V
# then s00 = sqrt(v00), s10 = v10/s00, s11 = sqrt(v11 - s10^2)
s00 = T.sqrt(var_z00)
s10 = cov_z10/s00
s11 = T.sqrt(var_z11 - s10**2)
s01 = T.zeros((nobs,))

S = T.stack([s00, s01, s10, s11])
S = S.transpose().reshape((nobs,2,2))

#%%

# Calculat the covariance matirx of the mean utility relative to each motorist
# chosen alternative

MS = T.batched_dot(Mi, S)
Sigma = T.batched_dot(MS, MS.dimshuffle((0,2,1)))

# Cholesky decomposition (see the note above)
c00 = T.sqrt(Sigma[:,0,0])
c10 = Sigma[:,1,0]/c00
c11 = T.sqrt(Sigma[:,1,1] - c10**2)

#%%

normcdf = lambda x: 0.5 + 0.5*T.erf(x/np.sqrt(2))
norminv = lambda p: np.sqrt(2)*T.erfinv(2*p-1)

ndraws = spec['ndraw'] if 'ndraw' in spec else 100
#draws = np.random.random((ndraws,nobs))

draws = (np.tile(np.arange(ndraws), (nobs,1)).transpose() + 0.5)/ndraws

# GHK simulator for 3 choices
prob0 = normcdf(-Vnonchoice[0,:]/c00)
prob1 = normcdf(-(Vnonchoice[1,:] + c10*norminv(draws*prob0))/c11).mean(axis=0)

nlogl_i = -T.log(prob0)*mask[0] - T.log(prob1)*mask[1]
nlogl = -T.log(prob0).sum() - T.log(prob1[mask[1]]).sum()


# theano function to calculate the log likelihood
eval_f = theano.function([theta], outputs = nlogl)

# automatic firs and second derivatives
grad = theano.function([theta], outputs = T.grad(nlogl, [theta]))
hess = theano.function([theta], outputs = theano.gradient.hessian(nlogl, [theta]))


# theano gradient returns a matrix instead of a vector, convert it to a vector
eval_grad = lambda t: np.squeeze(grad(t))

# theano hessian returns a 3D tensor, convert it to a matrix
eval_hess = lambda t: np.squeeze(hess(t))

#%% 

dv_control = groupid==0
nlog_control = nlogl_i[dv_control].sum()

eval_f_control = theano.function([theta], outputs = nlog_control)
grad_control = theano.function([theta], outputs = T.grad(nlog_control, [theta]))
hess_control = theano.function([theta], outputs = theano.gradient.hessian(nlog_control, [theta]))

eval_grad_control = lambda t: np.squeeze(grad_control(t))
eval_hess_control = lambda t: np.squeeze(hess_control(t))

#theta0 = solve_unconstr(theta0, eval_f_control, eval_grad_control, eval_hess_control)
#print(theta0)

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

#%%
    
#%%

# mean utility, using each of the alternative as base
# i.e. Vallabse is a N*J*(J-1) matrix: Vallbase(ijk) =  V(ij) - V(ik)
Vallbase        = T.dot(M, V)

# GHK simulator to approximate the likelihood
p0allbase       = T.maximum(normcdf(-Vallbase[:,0,:]/c00), 1e-8)
#drawsallbase    = np.random.random((ndraws,nchoice,nobs)) # uniform random draws
drawsallbase    =  (np.tile(np.arange(ndraws), (nobs,nchoice,1)).transpose() + 0.5)/ndraws # hammersley draws
draws1allbase   = norminv(drawsallbase*p0allbase)
p1allbase    = normcdf(-(Vallbase[:,1,:] + c10*draws1allbase)/c11).mean(axis=0)

# probability of choosing each of the alternative
pallbase = p0allbase*p1allbase
#
#if use_fe and use_share_moments:    
#    pstation = T.stack([pallbase[1:,np.where(stationid==i)[0]].mean(axis=1) for i in range(nstation)]).transpose().flatten()[(~nuisancexi).flatten().nonzero()[0]]
#    pstationtrue = np.stack([dv_choice[1:,stationid==i].mean(axis=1) for i in range(nstation)]).transpose().flatten()[~nuisancexi.flatten()]
#                   
#    obj_multiplier = T.dscalar('obj_multiplier')
#    lagrange_multiplier = T.dvector('lagrange_multiplier')
#    lagrange = obj_multiplier*obj + (lagrange_multiplier*pstation).sum()
#    
#    constr = theano.function([theta], pstation)
#    jab = theano.function([theta], T.jacobian(pstation, [theta]))
#    hess_constr = theano.function([theta, lagrange_multiplier, obj_multiplier], 
#                                  outputs=theano.gradient.hessian(lagrange, [theta]))
#    
#    ntheta1 = nalpha + nbeta + nallsigma
#    nxifull = (nchoice-1)*nstation
#    mask00 = np.ones((ntheta1, ntheta1), dtype = bool)
#    mask01 = np.ones((ntheta1, nxi), dtype = bool)
#    mask10 = np.ones((nxi, ntheta1), dtype = bool)
#    mask11 = np.tile(np.eye(nstation, dtype = bool), (nchoice-1, nchoice-1))[~nuisancexi.flatten(),:][:,~nuisancexi.flatten()]
#    
#    maskj = np.hstack((mask10, mask11))
#    maskh = np.hstack((np.vstack((mask00, mask10)), np.vstack((mask01, mask11))))
#    
#    def solve_constr(theta0, use_hess = False):
#        pyipopt.set_loglevel(1)    
#        n = theta0.size    
#        x_L = np.array([pyipopt.NLP_LOWER_BOUND_INF]*n, dtype=float)
#        x_U = np.array([pyipopt.NLP_UPPER_BOUND_INF]*n, dtype=float)        
#        ncon = pstationtrue.size
#        g_L = g_U = pstationtrue    
#        nnzj = maskj.sum()
#        nnzh = maskh.sum()    
#        idxrj, idxcj = np.mgrid[:ncon, :n]
#        idxrh, idxch = np.mgrid[:n, :n]    
#        eval_c = lambda t: constr(t)
#        eval_j = lambda t, f: (idxrj[maskj], idxcj[maskj]) if f else np.squeeze(jab(t))[maskj]
#        if use_hess:
#            eval_h = lambda t, l, o, f: (idxrh[maskh], idxch[maskh]) if f else np.squeeze(hess_constr(t,l,o))[maskh]
#            nlp = pyipopt.create(theta0.size, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad, eval_c, eval_j, eval_h)
#        else:
#            nlp = pyipopt.create(theta0.size, x_L, x_U, ncon, g_L, g_U, nnzj, nnzh, eval_f, eval_grad, eval_c, eval_j)
#        results = nlp.solve(theta0)
#        nlp.close()
#        return results
#    
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
        
        df.append(np.array(f(x2)-f(x1)).flatten()/(2*dx[i]))
        
    return np.stack(df)


#%%
    
if any(x in purpose for x in ['mfx', 'cf']): 
    eval_mean_pallbase = theano.function([theta], pallbase.mean(axis=1))
    eval_dmean_pallbase = theano.function([theta], T.jacobian(pallbase.mean(axis=1),[theta])[0])


tensor_lbls_dict = {Tprice: pricelbls, TX: Xlbls, TXp: Xplbls, TXsigma: Xsigmatreatlbls,
                    TXsigma_noconst: Xsigmatreatlbls_noconst, Tgroup_dummies: group_dummies.columns}

df2tensor = lambda x: df[x].values.transpose().astype(floatX)

def reset_values():
    for tensor, lbls in tensor_lbls_dict.items():
        tensor.set_value(df2tensor(lbls))        

def set_values_cf(df_cf):
    df_saved = df[df_cf.columns].copy()
    df[df_cf.columns] = df_cf
    reset_values()
    df[df_cf.columns] = df_saved
      
def cal_share_cf(thetahat, df_cf):
    set_values_cf(df_cf)
    return eval_mean_pallbase(thetahat), eval_dmean_pallbase(thetahat)

def cal_marginal_effect(thetahat, df1, df2):    
    share1, dshare1 = cal_share_cf(thetahat, df1)
    share2, dshare2 = cal_share_cf(thetahat, df2)
    reset_values()
    
    effecthat = share2 - share1
    deffect = dshare2 - dshare1
    coveffect = np.dot(np.dot(deffect, covhat), deffect.transpose())
    effectse = np.sqrt(np.diag(coveffect))
    return [effecthat, effectse]
    
def cal_marginal_effect_dummies(thetahat, dummyset):
    df1 = df.loc[:, Xlbls].copy()
    df1[dummyset] = 0.0
    
    mfx = {}
    for d in dummyset:
        df2 = df1.copy()
        df2[d] = 1.0
        print('mfx of ' + d)
        mfx[d] = cal_marginal_effect(thetahat, df1, df2)
    return mfx
    
#%%
def cal_marginal_price_elasticity(thetahat, treattype=None):
    df1 = df[pricelbls].copy()
    if treattype is not None:
        df1['treat_1'] = df1['const*treat_1'] = treattype == 'treat_1'
        df1['treat_2'] = df1['const*treat_2'] = treattype == 'treat_2'

    pg_km_adj_cf = df.pg_km_adj*1.01
    df2 = df1.copy()
    df2['rel_pe_km_adj'] = df.pe_km_adj - pg_km_adj_cf
    df2['rel_pgmidgrade_km_adj'] = df.pgmidgrade_km_adj - pg_km_adj_cf    
    pg_mfx, pg_mfx_se = cal_marginal_effect(thetahat, df1, df2)

    pe_km_adj_cf = df.pe_km_adj*1.01    
    df3= df1.copy()
    df3['rel_pe_km_adj'] = pe_km_adj_cf - df.pg_km_adj
    df3['rel_pgmidgrade_km_adj'] = df.pgmidgrade_km_adj - df.pg_km_adj
    pe_mfx, pe_mfx_se = cal_marginal_effect(thetahat, df1, df3)
    
    pgmidgrade_km_adj_cf = df.pgmidgrade_km_adj*1.01    
    df4= df1.copy()
    df4['rel_pe_km_adj'] = df.pe_km_adj - df.pg_km_adj
    df4['rel_pgmidgrade_km_adj'] = pgmidgrade_km_adj_cf - df.pg_km_adj
    pmg_mfx, pmg_mfx_se = cal_marginal_effect(thetahat, df1, df4)
   
    return pg_mfx, pg_mfx_se, pe_mfx, pe_mfx_se, pmg_mfx, pmg_mfx_se
    
    
#%%
    
    
def cal_marginal_effect_continuous(thetahat, varname, val):
    print('mfx of ' + varname)
    df1 = df[[varname]].copy()
    df2 = df[[varname]].copy()
    df2[varname] += val
    mfx = cal_marginal_effect(thetahat, df1, df2)
    mfx[0] /= val
    mfx[1] /= val
    return {varname: mfx}

#%%

resultfile = spec['resultfile'] if 'resultfile' in spec else input('Path to result: ')

if 'solve' in purpose:
    from solver import solve_unconstr
    thetahat = solve_unconstr(theta0, eval_f, eval_grad, eval_hess)
    with open(resultfile, 'w') as outfile:
        json.dump({'thetahat':thetahat.tolist(), 'specfile': specfile}, outfile, indent=2)

with open(resultfile, 'r') as outfile:
	results = json.load(outfile)
	
thetahat = np.array(results['thetahat'])


#%%

if 'cf' in purpose:
    pg_mfx, pg_mfx_se, pe_mfx, pe_mfx_se, pmg_mfx, pmg_mfx_se = cal_marginal_price_elasticity(thetahat)
    print("Type as per data")
    print("\t".join("{:>8.4f} ({:.4f})".format(b, se) for b, se in zip(pg_mfx, pg_mfx_se)))
    print("\t".join("{:>8.4f} ({:.4f})".format(b, se) for b, se in zip(pe_mfx, pe_mfx_se)))
    print("\t".join("{:>8.4f} ({:.4f})".format(b, se) for b, se in zip(pmg_mfx, pmg_mfx_se)))
    
    pg_mfx, pg_mfx_se, pe_mfx, pe_mfx_se, pmg_mfx, pmg_mfx_se = cal_marginal_price_elasticity(thetahat, 'control')
    print("Control")
    print("\t".join("{:>8.4f} ({:.4f})".format(b, se) for b, se in zip(pg_mfx, pg_mfx_se)))
    print("\t".join("{:>8.4f} ({:.4f})".format(b, se) for b, se in zip(pe_mfx, pe_mfx_se)))
    print("\t".join("{:>8.4f} ({:.4f})".format(b, se) for b, se in zip(pmg_mfx, pmg_mfx_se)))
    
    pg_mfx, pg_mfx_se, pe_mfx, pe_mfx_se, pmg_mfx, pmg_mfx_se = cal_marginal_price_elasticity(thetahat, 'treat_1')
    print("Treat 1")
    print("\t".join("{:>8.4f} ({:.4f})".format(b, se) for b, se in zip(pg_mfx, pg_mfx_se)))
    print("\t".join("{:>8.4f} ({:.4f})".format(b, se) for b, se in zip(pe_mfx, pe_mfx_se)))
    print("\t".join("{:>8.4f} ({:.4f})".format(b, se) for b, se in zip(pmg_mfx, pmg_mfx_se)))

    
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

if any(p in purpose for p in ['se', 'mfx', 'display', 'cf']):
    jacobian = theano.gradient.jacobian(nlogl_i, theta)
    eval_jab = theano.function([theta], jacobian)
    
    
    Jhat = eval_jab(thetahat)
    covhat = np.linalg.pinv(eval_hess(thetahat))
    
    GG = Jhat.transpose().dot(Jhat)
    GGclustered = np.zeros_like(GG)
    for stid in df.stationid.unique():
        Jsubhat = Jhat[(df.stationid==stid).nonzero()].sum(axis=0)
        GGclustered += np.outer(Jsubhat, Jsubhat)
    
    covhatclustered = np.matmul(covhat, np.matmul(GGclustered, covhat))
    
    covhat = covhatclustered
    sehat = np.sqrt(np.diag(covhat))
    tstat = thetahat/sehat

#%%

if 'mfx' in purpose: 
    marginal_effects = {}

    for mfx in filter(lambda k: k.startswith("mfx_dv_"), spec):
        marginal_effects.update(cal_marginal_effect_dummies(thetahat, spec[mfx]))

    for mfx in filter(lambda k: k.startswith("mfx_ct_"), spec):
        marginal_effects.update(cal_marginal_effect_continuous(thetahat, mfx[7:], spec[mfx]))

    df1 = df.loc[:, pricelbls].copy()
    df2 = df1.copy()
    df2 -= 0.01
    ehat, ese = cal_marginal_effect(thetahat, df1, df2)
    marginal_effects[pricelbls[1].replace('pgmidgrade', 'pg')] = [ehat/0.01, ese/0.01]
    
    marginal_effects_serialized = {}
    for k, v in marginal_effects.items():
        marginal_effects_serialized[k] = [x.tolist() for x in v]

    mfxfile = spec['mfxfile'] if 'mfxfile' in spec else input('Path to mfx (empty to skip saving): ')

    if mfxfile:
        with open(mfxfile, 'w') as outfile:
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
    
if 'display' in purpose:
    alphahat, alphase, alphatstat = get_stat(alpha, thetahat)
    betahat, betase, betatstat = get_stat(beta.flatten(), thetahat)
    gamma_ehat, gamma_ese, gamma_et = get_stat(gamma_e, thetahat)
    gamma_mhat, gamma_mse, gamma_mt = get_stat(gamma_m, thetahat)
    gamma_emhat, gamma_emse, gamma_emt = get_stat(gamma_em, thetahat)
    
    #
    #i1 = np.zeros(ngroup*(nchoice-1)-1, dtype=int) # base alternative = 0
    #i2 = np.tile(np.arange(ngroup, dtype=int), nchoice-1)[1:] # groupid
    #i3 = np.repeat(np.arange(nchoice-1, dtype=int), ngroup)[1:] # variance (diagonal element of each choice)
    #iii = (i1,i2,i3,i3)
    #Sigmamain = Sigma[iii]
    #
    #mm = np.hstack((-np.ones((2,1)), np.eye(2)))
    #mm = scipy.linalg.block_diag(np.eye(2), *([mm]*(ngroup*2//3-1)))
    #effect = T.dot(mm, T.log(Sigmamain))
    #
    #Sigmahat, Sigmase, Sigmatstat = get_stat(Sigmamain, thetahat)
    #effecthat, effectse, effecttstat = get_stat(effect, thetahat)
    
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
        for coef, se, t, lbl in zip(coeffs, ses, ts, labels):
            print_result_row(coef, se, t, lbl)
        print(divider2)
        
    for j in range(nchoice-1):
        idx = range(j*nX, (j+1)*nX)
        print_result_group("utiltiy of " + choicelbls[j],
                           betahat[idx], betase[idx], betatstat[idx], Xlbls)
    #
    #print_result_group("variance of random utility, " + choicelbls[0],
    #                   Sigmahat[:ngroup-1], Sigmase[:ngroup-1], Sigmatstat[:ngroup-1],
    #                   ["Treatment " + str(j) for j in range(1,ngroup)])
    #
    #for j in range(1,nchoice-1):
    #    idx = range(j*ngroup-1, (j+1)*ngroup-1)
    #    print_result_group("variance of random utility, " + choicelbls[j],
    #                       Sigmahat[idx], Sigmase[idx], Sigmatstat[idx],
    #                       ["Treatment " + str(j) for j in range(ngroup)])
    #
    #for j in range(nchoice-1):
    #    idx = range(j*(ngroup*2//3), (j+1)*(ngroup*2//3))
    #    print_result_group("effect on log(variance of random utility, " + choicelbls[j] + ")",
    #                       effecthat[idx], effectse[idx], effecttstat[idx],
    #                       ["Treatment " + str(j) for j in range(1,ngroup)])
    #
    
    Xsigmatreatlbls_extra = ["alpha_i*" + x for x in group_dummies.columns] if use_price_sensitity_in_sigma else []
        
    print_result_group("variance of ethanol random utility (log)",
                       gamma_ehat, gamma_ese, gamma_et,
                       Xsigmatreatlbls_noconst + Xsigmatreatlbls_extra)
        
    print_result_group("variance of midgrade-g random utility (log)",
                       gamma_mhat, gamma_mse, gamma_mt,
                       Xsigmatreatlbls + Xsigmatreatlbls_extra)
    
    print_result_group("corr of midgrade-g random utility (atanh)",
                       gamma_emhat, gamma_emse, gamma_emt,
                       Xsigmatreatlbls + Xsigmatreatlbls_extra)
    
    print("Log-likelihood:", eval_f(thetahat))
    print(divider)


#TXsigma = theano.shared(df[Xsigmatreatlbls].values.transpose().astype(floatX))

#Xsigmatreatlbls = []
#for x in group_dummies.columns:
#    for y in Xsigmalbls:
#        df[y+"*"+x] = df[x]*df[y]
#        Xsigmatreatlbls.append(y+"*"+x)
        
#%%
#if 'cf' in purpose:
#    group_dummies_cf = {x: group_dummies.copy() for x in ['control'] + group_dummies.columns.tolist()}
#    Xsigma_cf = {x: df[Xsigmatreatlbls].copy() for x in ['control'] + group_dummies.columns.tolist()}
#    
#    for x_cf in group_dummies.columns:
#        for x in group_dummies.columns:
#            group_dummies_cf[x_cf][x] = x_cf == x
#        
#        for x in group_dummies.columns:
#            for y in Xsigmalbls:
#                Xsigma_cf[x_cf][y+"*"+x] = df[y]*(x == 'treat_1')
#    
#        
#        TXsigma.set_value(Xsigma_cf[x_cf].values.transpose().astype(floatX))
#        Tgroup_dummies.set_values(group_dummies_cf[x_cf].values.transpose().astype(floatX))
    
#%%    
if 'se' in purpose:
    with open(resultfile, 'w') as outfile:
        json.dump({'thetahat':thetahat.tolist(), 'thetase': sehat.tolist(), 'specfile': specfile}, outfile, indent=2)

#%%
        
if 'sharecf' in purpose:
    pratio_range = np.arange(50,90)/100.0
    share_cf = []
    for pratio in pratio_range:
        pe_lt_cf = df.pg_lt*pratio
        pe_km_adj_cf = df.pe_km_adj/df.pe_lt*pe_lt_cf
        df_cf = df[pricelbls].copy()
        df_cf['rel_pe_km_adj'] = pe_km_adj_cf - df.pg_km_adj
        share_cf_i = []
        for treattype in ['control'] + group_dummies.columns.tolist():
            df_cf['treat_1'] = df_cf['const*treat_1'] = (treattype=='treat_1')*1
            df_cf['treat_2'] = df_cf['const*treat_2'] = (treattype=='treat_2')*1
            share_hat, share_se = cal_share_cf(thetahat, df_cf)
            share_cf_i.append(share_hat)
        share_cf.append(np.array(share_cf_i))
        
    # share_cf.shape = (cf, group, fuel) 
    share_cf = np.array(share_cf)
    
    sharecf_results_cf = pd.DataFrame(data = {'pratio': pratio_range, 
                                              'share_g_control': share_cf[:,0,0],
                                              'share_g_treat1': share_cf[:,1,0],
                                              'share_g_treat2': share_cf[:,2,0],
                                              'share_e_control': share_cf[:,0,1],
                                              'share_e_treat1': share_cf[:,1,1],
                                              'share_e_treat2': share_cf[:,2,1],
                                              'share_mg_control': share_cf[:,0,2],
                                              'share_mg_treat1': share_cf[:,1,2],
                                              'share_mg_treat2': share_cf[:,2,2]})
    
    sharecf_results_cf['id'] = np.arange(sharecf_results_cf.shape[0])    
    sharecf_results_cf_long = pd.wide_to_long(sharecf_results_cf, ['share_g', 'share_e', 'share_mg'], i='id', j='treattype', sep='_', suffix='\w+').reset_index()

#%%

from ggplot import ggplot, aes, geom_line, xlab, ylab, save, scale_linetype_discrete, theme, element_blank, geom_histogram, facet_wrap

plot = ggplot(sharecf_results_cf_long[sharecf_results_cf_long.treattype != 'treat2']) + \
    aes(x='share_e', y='pratio', group='treattype') + \
    geom_line(aes(linetype='treattype')) + \
    ylab("$p_e/p_g$") + xlab("ethanol share") + \
    scale_linetype_discrete(labels=("Control", "Price-ratio treatment")) + \
    theme(legend_title=element_blank(), legend_position = (0.7, 0.75))

# plot.save(resultfile.replace('_results.json', 'sharecfe.pdf'))
plot.save(resultfile.replace('_results.json', 'sharecfe.pdf'))

#%%
plot = ggplot(sharecf_results_cf_long[sharecf_results_cf_long.treattype != 'treat2']) + \
    aes(x='share_g', y='pratio', group='treattype') + \
    geom_line(aes(linetype='treattype')) + \
    ylab("$p_e/p_g$") + xlab("gasoline share") + \
    scale_linetype_discrete(labels=("Control", "Price-ratio treatment")) + \
    theme(legend_title=element_blank(), legend_position = (0.3, 0.75))
# plot.save(resultfile.replace('_results.json', 'sharecfg.pdf'))
plot.save(resultfile.replace('_results.json', 'sharecfg.pdf'))

#%%
sharecf_results_cf['d1share_e'] = sharecf_results_cf.share_e_treat1 - sharecf_results_cf.share_e_control
sharecf_results_cf['d1share_g'] = sharecf_results_cf.share_g_treat1 - sharecf_results_cf.share_g_control

plot = ggplot(sharecf_results_cf) + \
    aes(y='d1share_e', x='pratio') + \
    geom_line() + \
    xlab("$p_e/p_g$") + ylab("Change in ethanol share (ppts)")
plot.save(resultfile.replace('_results.json', 'dsharecfe.pdf'))

plot = ggplot(sharecf_results_cf) + \
    aes(y='d1share_g', x='pratio') + \
    geom_line() + \
    xlab("$p_e/p_g$") + ylab("Change in gasoline share (ppts)")
plot.save(resultfile.replace('_results.json', 'dsharecfg.pdf'))

#%%
def binsearch(f,  target, lb, ub, toler=1e-6):
    mid = (lb+ub)/2.0
    while np.abs(lb-ub)>toler:
        mid = (lb+ub)/2.0
        gap = f(mid) - target
        if np.abs(gap) < toler:
            return mid
        if gap < 0:
            lb = mid
        else:
            ub = mid
            
    return mid

def binsearch_vector(f, target, lb, ub, toler=1e-6):
    iter = 0
    while True:
        mid = (lb+ub)/2.0
        gap = f(mid) - target
        iter += 1
#        print('Bin search iter {}, error = {}'.format(iter, np.abs(gap).max()))
        if np.abs(gap).max() < toler or np.abs(ub-lb).max()<toler:
            return mid
        lb[gap<0] = mid[gap<0]
        ub[gap>0] = mid[gap>0]
    return mid

#%%
    
if 'choicesetcf' in purpose:
    ndraws_cf = 50
    draws_cf = np.random.normal(size=(ndraws_cf, nobs,2))
    
    
            
    eval_V = theano.function([theta], V)
    eval_S = theano.function([theta], S)
    
    def mean_utility_cf(pe_km_adj_cf, treattype = 'control', choiceset = [0,1,2], axes = None, theta=thetahat.copy()):
        df_cf = df[pricelbls].copy()
        df_cf['treat_1'] = df_cf['const*treat_1'] = (treattype=='treat_1')*1
        df_cf['treat_2'] = df_cf['const*treat_2'] = (treattype=='treat_2')*1
        
        df_cf['pe_km_adj'] = pe_km_adj_cf
        df_cf['rel_pe_km_adj'] = df_cf.pe_km_adj - df.pg_km_adj
        df_cf['rel_lpe_km_adj'] = np.log(df_cf.pe_km_adj) - np.log(df.pg_km_adj)
        
        set_values_cf(df_cf)
        Vmean = eval_V(theta)
        Shat = eval_S(theta)
        random_utility = np.einsum('ijk,rik->rji', Shat, draws_cf)
        
        Vrandom = Vmean + random_utility
        Vmax = Vrandom[:,[j-1 for j in choiceset if j > 0],:].max(axis=1)
        
        if 0 in choiceset:
            Vmax = np.maximum(Vmax, 0)
            
        return Vmax.mean(axis=axes)
       
        
    choiceset = {'no_g':[1,2], 'no_fossil': [1], 'no_mg': [0,1]}
    
    pratio_range = np.arange(55,71,5)/100
    choisetcf_commondiscount_df = pd.DataFrame()
    
    for pratio_cf in pratio_range:
        print("Counterfactual: pe/pg = {}".format(pratio_cf))
        pe_lt_cf = df.pg_lt*pratio_cf
        pe_km_adj_cf = df.pe_km_adj/df.pe_lt*pe_lt_cf
                
        for treattype in ['control', 'treat_1']:
            mean_utility_fullchoiceset = mean_utility_cf(pe_km_adj_cf, treattype)
            
            for choicesetcode in ['no_g', 'no_fossil', 'no_mg']:
                mean_utility_newchoiceset = lambda discount: mean_utility_cf(pe_km_adj_cf*(1-discount), treattype, choiceset[choicesetcode])
                pe_discount_solution = binsearch(mean_utility_newchoiceset, mean_utility_fullchoiceset, 0, 1)
                
                choisetcf_commondiscount_df = choisetcf_commondiscount_df.append({
                        'pratio': pratio_cf,
                        'treattype': treattype, 
                        'choiceset': choicesetcode,
                        'pe_discount': pe_discount_solution}, ignore_index=True)
                
    
    choisetcf_indivdiscount_df = pd.DataFrame()
    
    for pratio_cf in pratio_range:
        print("Counterfactual: pe/pg = {}".format(pratio_cf))
        pe_lt_cf = df.pg_lt*pratio_cf
        pe_km_adj_cf = df.pe_km_adj/df.pe_lt*pe_lt_cf
                    
        for treattype in ['control', 'treat_1']:
            mean_utility_fullchoiceset = mean_utility_cf(pe_km_adj_cf, treattype, axes=0)
            
            for choicesetcode in ['no_g', 'no_fossil', 'no_mg']:
                mean_utility_newchoiceset = lambda discount: mean_utility_cf(pe_km_adj_cf*(1-discount), treattype, choiceset[choicesetcode], axes=0)
                pe_discount_solution = binsearch_vector(mean_utility_newchoiceset, mean_utility_fullchoiceset, np.ones_like(mean_utility_fullchoiceset)*(-3), np.ones_like(mean_utility_fullchoiceset)*10)
                
                choisetcf_indivdiscount_df = choisetcf_indivdiscount_df.append({
                        'pratio': pratio_cf,
                        'treattype': treattype, 
                        'choiceset': choicesetcode,
                        'pe_discount': pe_discount_solution.mean()}, ignore_index=True)

    
#%%
choisetcf2_commondiscount_df = pd.DataFrame()
pe_km_adj_cf = df.pe_km_adj.copy()
for treattype in ['control', 'treat_1']:
    mean_utility_fullchoiceset = mean_utility_cf(pe_km_adj_cf, treattype)
    
    for choicesetcode in ['no_g', 'no_fossil', 'no_mg']:
        print('Counterfacture for {}, choice set {}'.format(treattype, choicesetcode))
        mean_utility_newchoiceset = lambda discount, theta: mean_utility_cf(pe_km_adj_cf*(1-discount), treattype, choiceset[choicesetcode], theta=theta)
        find_pe_discount = lambda theta: binsearch(lambda discount: mean_utility_newchoiceset(discount, theta), mean_utility_fullchoiceset, 0, 1)
        pe_discount_solution = find_pe_discount(thetahat)
        grad_pe_solution = findiff(find_pe_discount, thetahat)
        pe_discount_se = np.squeeze(np.sqrt(np.dot(np.dot(grad_pe_solution.transpose(), covhat), grad_pe_solution)))
        choisetcf2_commondiscount_df = choisetcf2_commondiscount_df.append({
                'treattype': treattype, 
                'choiceset': choicesetcode,
                'pe_discount': pe_discount_solution,
                'pe_discount_se': pe_discount_se}, ignore_index=True)

choisetcf2_commondiscount_df.to_csv(resultfile.replace('_results.json', '_choicesetcf_commondiscount.csv'))

#%%
choisetcf2_indivdiscount_df = pd.DataFrame()
pe_km_adj_cf = df.pe_km_adj.copy()
for treattype in ['control', 'treat_1']:
    mean_utility_fullchoiceset = mean_utility_cf(pe_km_adj_cf, treattype, axes=0)
    
    for choicesetcode in ['no_g', 'no_fossil', 'no_mg']:
        mean_utility_newchoiceset = lambda discount, theta: mean_utility_cf(pe_km_adj_cf*(1-discount), treattype, choiceset[choicesetcode], axes=0, theta=theta)
        find_pe_discount = lambda theta: binsearch_vector(lambda discount: mean_utility_newchoiceset(discount, theta), mean_utility_fullchoiceset, np.ones_like(mean_utility_fullchoiceset)*(-3), np.ones_like(mean_utility_fullchoiceset)*10).mean()
#        pe_discount_solution = binsearch_vector(mean_utility_newchoiceset, mean_utility_fullchoiceset, np.ones_like(mean_utility_fullchoiceset)*(-3), np.ones_like(mean_utility_fullchoiceset)*10).mean()
        pe_discount_solution = find_pe_discount(thetahat)
        grad_pe_solution = findiff(find_pe_discount, thetahat)
        pe_discount_se = np.squeeze(np.sqrt(np.dot(np.dot(grad_pe_solution.transpose(), covhat), grad_pe_solution)))
        
        print('discount = {:.5f}, se = {:.5f}'.format(pe_discount_solution, pe_discount_se))
        
        choisetcf2_indivdiscount_df = choisetcf2_indivdiscount_df.append({
                'treattype': treattype, 
                'choiceset': choicesetcode,
                'pe_discount': pe_discount_solution,
                'pe_discount_se': pe_discount_se}, ignore_index=True)

choisetcf2_indivdiscount_df.to_csv(resultfile.replace('_results.json', '_choisetcf_indivdiscount.csv'))

#%%
choisetcf2_indivdiscount_70pg_df = pd.DataFrame()
pe_km_adj_cf = df.pe_km_adj/df.pe_lt*(df.pg_lt*0.7)
for treattype in ['control', 'treat_1']:
    mean_utility_fullchoiceset = mean_utility_cf(pe_km_adj_cf, treattype, axes=0)
    
    for choicesetcode in ['no_g', 'no_fossil', 'no_mg']:
        mean_utility_newchoiceset = lambda discount, theta: mean_utility_cf(pe_km_adj_cf*(1-discount), treattype, choiceset[choicesetcode], axes=0, theta=theta)
        find_pe_discount = lambda theta: binsearch_vector(lambda discount: mean_utility_newchoiceset(discount, theta), mean_utility_fullchoiceset, np.ones_like(mean_utility_fullchoiceset)*(-3), np.ones_like(mean_utility_fullchoiceset)*10).mean()
#        pe_discount_solution = binsearch_vector(mean_utility_newchoiceset, mean_utility_fullchoiceset, np.ones_like(mean_utility_fullchoiceset)*(-3), np.ones_like(mean_utility_fullchoiceset)*10).mean()
        pe_discount_solution = find_pe_discount(thetahat)
        grad_pe_solution = findiff(find_pe_discount, thetahat)
        pe_discount_se = np.squeeze(np.sqrt(np.dot(np.dot(grad_pe_solution.transpose(), covhat), grad_pe_solution)))
        
        print('discount = {:.5f}, se = {:.5f}'.format(pe_discount_solution, pe_discount_se))
        
        choisetcf2_indivdiscount_70pg_df = choisetcf2_indivdiscount_70pg_df.append({
                'treattype': treattype, 
                'choiceset': choicesetcode,
                'pe_discount': pe_discount_solution,
                'pe_discount_se': pe_discount_se}, ignore_index=True)

choisetcf2_indivdiscount_70pg_df.to_csv(resultfile.replace('_results.json', '_choisetcf_indivdiscount_70pg.csv'))

#%%
choisetcf2_indivdiscount_60pg_df = pd.DataFrame()
pe_km_adj_cf = df.pe_km_adj/df.pe_lt*(df.pg_lt*0.63)
for treattype in ['control', 'treat_1']:
    mean_utility_fullchoiceset = mean_utility_cf(pe_km_adj_cf, treattype, axes=0)
    
    for choicesetcode in ['no_g', 'no_fossil', 'no_mg']:
        mean_utility_newchoiceset = lambda discount, theta: mean_utility_cf(pe_km_adj_cf*(1-discount), treattype, choiceset[choicesetcode], axes=0, theta=theta)
        find_pe_discount = lambda theta: binsearch_vector(lambda discount: mean_utility_newchoiceset(discount, theta), mean_utility_fullchoiceset, np.ones_like(mean_utility_fullchoiceset)*(-3), np.ones_like(mean_utility_fullchoiceset)*10).mean()
#        pe_discount_solution = binsearch_vector(mean_utility_newchoiceset, mean_utility_fullchoiceset, np.ones_like(mean_utility_fullchoiceset)*(-3), np.ones_like(mean_utility_fullchoiceset)*10).mean()
        pe_discount_solution = find_pe_discount(thetahat)
        grad_pe_solution = findiff(find_pe_discount, thetahat)
        pe_discount_se = np.squeeze(np.sqrt(np.dot(np.dot(grad_pe_solution.transpose(), covhat), grad_pe_solution)))
        
        print('discount = {:.5f}, se = {:.5f}'.format(pe_discount_solution, pe_discount_se))
        
        choisetcf2_indivdiscount_60pg_df = choisetcf2_indivdiscount_60pg_df.append({
                'treattype': treattype, 
                'choiceset': choicesetcode,
                'pe_discount': pe_discount_solution,
                'pe_discount_se': pe_discount_se}, ignore_index=True)

choisetcf2_indivdiscount_60pg_df.to_csv(resultfile.replace('_results.json', '_choisetcf_indivdiscount_60pg.csv'))

#%%
pe_km_adj_cf = df.pe_km_adj.copy()
mean_utility_fullchoiceset = mean_utility_cf(pe_km_adj_cf, 'treat_1', axes=0)
mean_utility_newchoiceset = lambda discount: mean_utility_cf(pe_km_adj_cf*(1-discount), 'treat_1', [1], axes=0, theta=thetahat)
pe_discount_dist = binsearch_vector(mean_utility_newchoiceset, mean_utility_fullchoiceset, np.ones_like(mean_utility_fullchoiceset)*(-3), np.ones_like(mean_utility_fullchoiceset)*10)
pe_discount_dist_df = pd.DataFrame({'pe_discount': pe_discount_dist})
plot = ggplot(pe_discount_dist_df) + aes(x='pe_discount') + geom_histogram() + xlab('ethanol discount') + ylab('count')
plot.save(resultfile.replace('_results.json', 'pe_discount_hist.pdf'))

#%%
pe_km_adj_cf = df.pe_km_adj/df.pe_lt*(df.pg_lt*0.7)
mean_utility_fullchoiceset = mean_utility_cf(pe_km_adj_cf, 'treat_1', axes=0)
mean_utility_newchoiceset = lambda discount: mean_utility_cf(pe_km_adj_cf*(1-discount), 'treat_1', [1], axes=0, theta=thetahat)
pe_discount_dist = binsearch_vector(mean_utility_newchoiceset, mean_utility_fullchoiceset, np.ones_like(mean_utility_fullchoiceset)*(-3), np.ones_like(mean_utility_fullchoiceset)*10)
pe_discount_dist_70pg_df = pd.DataFrame({'pe_discount': pe_discount_dist})
plot = ggplot(pe_discount_dist_70pg_df) + aes(x='pe_discount') + geom_histogram() + xlab('ethanol discount') + ylab('count')
plot.save(resultfile.replace('_results.json', 'pe_discount_70pg_hist.pdf'))

#%%
pe_km_adj_cf = df.pe_km_adj/df.pe_lt*(df.pg_lt*0.6)
mean_utility_fullchoiceset = mean_utility_cf(pe_km_adj_cf, 'treat_1', axes=0)
mean_utility_newchoiceset = lambda discount: mean_utility_cf(pe_km_adj_cf*(1-discount), 'treat_1', [1], axes=0, theta=thetahat)
pe_discount_dist = binsearch_vector(mean_utility_newchoiceset, mean_utility_fullchoiceset, np.ones_like(mean_utility_fullchoiceset)*(-3), np.ones_like(mean_utility_fullchoiceset)*10)
pe_discount_dist_60pg_df = pd.DataFrame({'pe_discount': pe_discount_dist})
plot = ggplot(pe_discount_dist_60pg_df) + aes(x='pe_discount') + geom_histogram() + xlab('ethanol discount') + ylab('count')
plot.save(resultfile.replace('_results.json', 'pe_discount_60pg_hist.pdf'))

#%%
#pe_discount_dist_df = pe_discount_dist_60pg_df.copy()
#pe_discount_dist_df['pg_discount_60'] = 1-0.6*(1-pe_discount_dist_df.pe_discount)
#pe_discount_dist_df['pg_discount_70'] = 1-0.7*(1-pe_discount_dist_70pg_df.pe_discount)
#pe_discount_dist_df['diff_discount'] = pe_discount_dist_df.pg_discount_60 - pe_discount_dist_df.pg_discount_70
#ggplot(pe_discount_dist_df) + aes(x='diff_discount') + geom_histogram() + xlab('ethanol discount') + ylab('count')

#%%
pe_discount_dist_df['case'] = 'existing prices'
pe_discount_dist_df['dollar_discount'] = pe_discount_dist_df.pe_discount*df.pe_km_adj
pe_discount_dist_70pg_df['case'] = 'pe/pg=0.7'
pe_discount_dist_70pg_df['dollar_discount'] = pe_discount_dist_70pg_df.pe_discount*df.pe_km_adj/df.pe_lt*(df.pg_lt*0.7)
pe_discount_dist_60pg_df['case'] = 'pe/pg=0.63'
pe_discount_dist_60pg_df['dollar_discount'] = pe_discount_dist_60pg_df.pe_discount*df.pe_km_adj/df.pe_lt*(df.pg_lt*0.6)

#%%
pe_discount_combined = pe_discount_dist_df.append(pe_discount_dist_70pg_df).append(pe_discount_dist_60pg_df)
labels = {'existing prices': 'existing $p_e$', 'pe/pg=0.7': '$p_e/p_g=0.7$', 'pe/pg=0.63': '$p_e/p_g=0.63$'}
plot = ggplot(pe_discount_combined) + aes(x='dollar_discount', group='case') + geom_histogram() + facet_wrap('case', ncol=1, labeller=lambda x: labels[x]) + xlab('Ethanol compensating discount in R\$/km')
plot.save(resultfile.replace('_results.json', 'pe_discount_hist_combined.pdf'))

#choisetcf2_commondiscount_df = pd.DataFrame()
#pe_km_adj_cf = df.pe_km_adj.copy()
#
#V_fullchoiceset_control = mean_utility_cf(pe_km_adj_cf, 'control')
#V_fullchoiceset_treat1 = mean_utility_cf(pe_km_adj_cf, 'treat_1')
#
#V_no_g_control = lambda discount, theta: mean_utility_cf(pe_km_adj_cf*(1-discount), choiceset=[1,2], theta=theta)
#V_no_g_treat1 = lambda discount, theta: mean_utility_cf(pe_km_adj_cf*(1-discount), 'treat_1', choiceset=[1,2], theta=theta)
#
#V_only_e_control = lambda discount, theta: mean_utility_cf(pe_km_adj_cf*(1-discount), choiceset=[1], theta=thetahat)
#V_only_e_treat1 = lambda discount, theta: mean_utility_cf(pe_km_adj_cf*(1-discount), 'treat_1', choiceset=[1], theta=theta)
#
#pe_discount_no_g_control = lambda theta: binsearch(lambda discount: V_no_g_control(discount, theta), V_fullchoiceset_control, 0, 1)
#pe_discount_no_g_treat1 = lambda theta: binsearch(lambda discount: V_no_g_treat1(discount, theta), V_fullchoiceset_treat1, 0, 1)
#
#pe_discount_only_e_control = lambda theta: binsearch(lambda discount: V_only_e_control(discount, theta), V_fullchoiceset_control, 0, 1)
#pe_discount_only_e_treat1 = lambda theta: binsearch(lambda discount: V_only_e_treat1(discount, theta), V_fullchoiceset_treat1, 0, 1)
#
#diff_pe_discount_no_g = lambda theta: pe_discount_no_g_control(theta) - pe_discount_no_g_treat1(theta)
#
#grad_diff_pe_discount_no_g = findiff(diff_pe_discount_no_g, thetahat.copy())
#
#diff_pe_discount_no_g_hat = diff_pe_discount_no_g(thetahat)
#
#diff_pe_discount_no_g_se = np.dot(np.dot(grad_diff_pe_discount_no_g.transpose(), covhat), grad_diff_pe_discount_no_g)
#diff_pe_discount_no_g_se = np.squeeze(diff_pe_discount_no_g_se)
##%%
#
#for treattype in ['control', 'treat_1']:
#    mean_utility_fullchoiceset = mean_utility_cf(pe_km_adj_cf, treattype)
#    
#    for choicesetcode in ['no_g', 'no_fossil', 'no_mg']:
#        mean_utility_newchoiceset = lambda discount, thetahat: mean_utility_cf(pe_km_adj_cf*(1-discount), treattype, choiceset[choicesetcode], thetahat=thetahat)
#        find_pe_discount = lambda thetahat: binsearch(lambda discount: mean_utility_newchoiceset(discount, thetahat), mean_utility_fullchoiceset, 0, 1)
#        pe_discount_solution = find_pe_discount(thetahat)
#        grad_pe_solution = findiff(find_pe_discount, thetahat)
#        pe_discount_se = np.squeeze(np.sqrt(np.dot(np.dot(grad_pe_solution.transpose(), covhat), grad_pe_solution)))
#        choisetcf2_commondiscount_df = choisetcf2_commondiscount_df.append({
#                'treattype': treattype, 
#                'choiceset': choicesetcode,
#                'pe_discount': pe_discount_solution,
#                'pe_discount_se': pe_discount_se}, ignore_index=True)
