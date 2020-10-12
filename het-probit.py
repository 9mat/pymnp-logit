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
#     specify whether to solve for estimates, to calculate the marginal effects
#     or to just display the results. purposes can be combined
#     specify purposes in the second argument of the command, or when prompted      
purpose = sys.argv[2] if len(sys.argv) > 2 else input("Purpose (solve/mfx/display): ")

theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

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

# if `subsample` is not None, use only the corresponding subsample
# if `subsample` starts with `~`, use the complement subsample instead
# e.g. subsample = dv_somecollege to use only college graduates
#      subsample = ~dv_somecollege to use motorists with no college degree
if subsample is not None:
    df = df[df[subsample] == (not subsample.startswith("~"))]
        
# print('Number of observations {}'.format(len(df)))

# generate day of week dummies
dow_dummies = pd.get_dummies(df['date'].dt.dayofweek, prefix='dv_dow')
df[dow_dummies.columns[1:]] = dow_dummies[dow_dummies.columns[1:]]

treat_dummies = pd.get_dummies(df.treattype, prefix='treat', drop_first=True)
treatlbls = treat_dummies.columns
df[treatlbls] = treat_dummies


# covariates that shift the covariance matrix
# the covariates include variables specified in Xsigmalbls
# and their interactions with the treatement dummies
Xsigmatreatlbls = Xsigmalbls[:]
for x in treatlbls:
    for y in Xsigmalbls:
        # interaction between Xsigma and treatment
        df[y+"*"+x] = df[x]*df[y] 
        Xsigmatreatlbls.append(y+"*"+x)
        
# the constant term for the base alternative's variance is not identified
# thus, the set of coveriates for it will not have const
Xsigmatreatlbls_noconst = [x for x in Xsigmatreatlbls if x != 'const']


#%% UNPACK DATA, PREPARE PARAMETERS


# number of covariates in the mean utility and the price sensitivity equation
# and the number of alternatives
nX, nXp, J  = len(Xlbls), len(Xplbls), len(pricelbls) + 1

# number of parameters to be estimated
# note: the covariance matrix sigma is (J-1) by (J-1) matrix
# the first variance is not identified
# thus, we have (J-1)*J//2 - 1 identified parameters in the matrix
nalpha  = nXp
nbeta   = nX*(J-1)
nsigma  = (J-1)*J//2 - 1

N       = len(df)
ngroup  = df.treattype.unique().size

# indicates whether to allow the price sensitivity to shift with demographics
use_dprice = nalpha > 1


# ngamma = len(Xsigmalbls) + use_price_sensitity_in_sigma
# ngamma_e = ngamma_m = ngamma_em = ngamma*(ngroup-1) + ngamma - use_price_sensitity_in_sigma
# ngamma_e -= ('const' in Xsigmalbls)
ngamma_e = ngamma_m = ngamma_em = len(Xsigmalbls)

# number of stations (= number of unique station id)
M = df.stationid.unique().size

# recode the station id so that it is in consecutive order from 0 to (M-1)
df.stationid = df.stationid.astype('category').cat.rename_categories(range(M))
stationid = df.stationid.values

choice  = df.choice.values

# dummies for fuel choices
dv_choice = pd.get_dummies(choice).values.transpose()

#%%
if use_fe:
    # If an alternative is never chosen at a station, the corresponding xi
    # is not identified. We need to count and mark the number of 
    # the xi's that can be identified
    
    # count the frequency of each choice by station
    choicestation_count = np.zeros((J, M))
    for j, sid in zip(choice, stationid):
        choicestation_count[j-1, sid] += 1

    
    # Note that, with `J` alternatives, there are `J-1` xi's in
    # each station (the xi of the base alternative is set to 0)
    # Create boolean matrix of size (J-1, M) to mark if the xi's
    # is identified or not <=> the alternative is ever chosen or not
    xi_identified = choicestation_count[1:] > 0
    
    # the number of xi's that can be identified
    nxi = xi_identified.sum()
    
    # Index the identified xi's from 0 to (nxi-1) and index all the other
    # non-identified xi's by nxi
    xi_idx = np.zeros(xi_identified.shape, dtype = int)
    xi_idx[xi_identified] = np.arange(nxi)
    xi_idx[~xi_identified] = nxi
else:
    nxi = 0
    
#theta0 = np.zeros((nalpha + nbeta + ngamma_e + ngamma_m + ngamma_em + 2 + nxi,))
    
    
gamma_ehat = [-0.38133455, -0.18611983, -6.01431427]
gamma_mhat = [ -5.87281899,  -2.37829492, -52.92545266]
gamma_emhat = [1.08950418e-05,  3.37502157e-05,  1.02565212e-05]
lsigma_mhat = 0.326445991950642
atsigma_emhat = -0.667286566721085
#alphahat = [-196.14601568,   36.31973006,   14.31932179,   26.09967419,
#         37.65339836,    0.79270248,    9.07639863]
alphahat = [0]*nalpha
alphahat[0] = -30
betahat = [0.0]*nbeta

theta0 = np.array(alphahat + betahat + gamma_ehat + gamma_mhat + gamma_emhat + [lsigma_mhat, atsigma_emhat])

#theta0 = np.array([
#        0, 0, 0, 0, 0,
#        -3.36951638e+01, -2.74197526e+01, -1.86185845e-01, -9.44574978e-02,
#       -2.05805368e-01, -4.67825877e-01, -5.11601818e-01, -3.30847195e-01,
#       -6.32480976e+00, -4.65350655e+00, -1.76412592e+00, -2.85187053e+01,
#        2.67363491e-05, -3.29968552e-05, -7.01768875e-04,  2.80919394e-01,
#        6.08997501e-01])

#if 'rel_pe_km_adj' in pricelbls:
#    theta0[0] = -30

np.random.seed(1234)

#%% DEFINE THEANO DATA TO REPRESENT PARAMETERS, DATA, AND LIKELHOOD
# use double precision in all calculation
floatX = 'float64'

# the vector of paremeters to be estimated
theta  = T.dvector('theta')

# unpack parameters
#   alpha: price sensisivity paramater
#   beta: choice-specific coefficients of the mean utility equation
#   gamma_e: coefficients in the ethanol-variance equation
#   gamma_m: coefficients in the mgasoline-variance equation
#   gamma_em: coefficients in the correlation equation
#   xiraw: fuel-station fixed effects that can be identified

offset = 0
alpha  = theta[offset: offset + nalpha]

offset += nalpha
beta   = theta[offset: offset + nbeta].reshape((J-1, nX))

offset += nbeta
gamma_e = theta[offset: offset + ngamma_e]

offset += ngamma_e
gamma_m = theta[offset: offset + ngamma_m]

offset += ngamma_m
gamma_em = theta[offset: offset + ngamma_em]

offset += ngamma_em
lsigma_m = theta[offset]

offset +=1 
atsigma_em = theta[offset]

offset += 1
xiraw = theta[offset:offset+nxi] if use_fe else 0

# fuel-station fixed effects
if use_fe:
    # Impute all the unidentified xi's by -10000
    # append this number to the end of the xiraw vector
    # so that this value maps to the index nxi of the vector 
    xiraw_padded = T.concatenate([xiraw, [-10000.]])
    
    # map the vector of raw xi's to the correspoinding station and choice
    # using the index matrix xi_idx, create a matrix of (J-1, M)
    # values of xi
    xi = xiraw_padded[xi_idx]
    
    # find the xi corresponding to all individuals by looking up the stationid
    # that they are in, xi_i is a matrix of size (J-1, N)
    xi_i = xi[:, stationid]
else:
    xi_i = 0

# To be used in the GHK simulator:
# Matrix M is used to transform the vector of mean utility from the current
# the base alternative (gasoline, choice 0) to another base alternative.
# M consists of `J` matrices of size `(J-1, J-1)`
# Suppose V is the mean utility using choice 0 as the base alternative 
# M_j*V will be the mean utility using choice j as the base alternative
M = np.stack([[[1,0],
               [0,1]],  # base 0 to base 0 aka identity matrix
              [[-1,0],
               [-1,1]], # base 0 to base 1
              [[0,-1],
               [1,-1]]]) # base 0 to base 2

# For each motorist, transform the mean utity using gasoline (choice 0) as the 
# base alternative to the mean utility using the chosen fuel as the base.
# Mi will be a set of N matrices of size (J-1, J-1),
# so that Mi[i]*V[i] will be the  i-th motorist's mean utility using his/her
# chosen fuel as the base alternative.
Mi = M[choice-1, :, :]

def df2tensor(x, name):
    return theano.shared(df[x].values.transpose().astype(floatX), name=name)

# unpack the data from the dataframe to individual vectors/matrices
# for ease of implementation
price   = df[pricelbls].values.transpose()
X       = df[Xlbls].values.transpose()
Xp      = df[Xplbls].values.transpose()
Xsigma  = df[Xsigmalbls].values.transpose()
groupid = df[grouplbls].values.transpose()


# convert the data to theano data
price   = df2tensor(pricelbls, 'price')
X       = df2tensor(Xlbls, 'X')
Xp      = df2tensor(Xplbls, 'Xp')


# mean utility equation, using gasoline as the base alternative
# V_ijt = alpha_i*p_jt + beta_j*X_ijt + xi_jt
alpha_i = T.dot(alpha, Xp)
V = alpha_i*price + T.dot(beta, X)  + xi_i

# mean utility, using each motorist's chosen fuel as the base alternative
# this is to be used for the GHK simulator
Vnonchoice = T.batched_dot(Mi,V.transpose()).transpose()  



# In principle, with the imputation of missing prices (10000) and unidentified
# xi's (-10000) the likelihood can be calculated as normal.
# This sometimes however creates extreme large values and can make some
# computation (e.g. taking exponential) unstable (in numpy, exp(-1000) = 0 and exp(1000) = inf)
# In such situation, we need to pick out and discard the log likelihood
# corresponding to those prices and xi's (which should be zero anyway)
# The boolean vector `mask` of size (J-1, N) is to pick out the
# indidual likelihood that do not correpsond to those prices and xi's
if use_fe:
    # Assuming the first choice (gasoline) is always identified
    # Applying the same transformation M to the set of dummies for identified xi's
    # will give use the new set of dummies for non-zero likelihood
    mask = np.einsum('ijk,ki->ji', Mi, ~xi_identified[:,stationid]) == 0
else:
    # Even without fixed effects, we still have some stations with missing 
    # fuel, and the prices are imputed with a very high value
    # The log likelihood of those will be zero, so including them or not
    # will not affect total log likelihood
    # However, due to extreme values, the calculations can become unstable
    # so, if possible, we exclude these observations (where the price of a fuel 
    # is missing)
    mask = np.abs(np.einsum('ijk,ki->ji', Mi, df[pricelbls].values.transpose())) < 10

#%%
    
# covariates in the covariance equation
# Xsigma = df2tensor(Xsigmatreatlbls, 'Xsigma')
Xsigma = df2tensor(Xsigmalbls, 'Xsigma')


# # covariates in the covariance equation, excluding the const
# Xsigma_noconst = df2tensor(Xsigmatreatlbls_noconst, 'Xsigma_noconst')

# # group dummies
# treat_dummies = df2tensor(treatlbls, 'treat')

# # add the price sensitivity to Xsigma if allowing price sentivity to shift 
# # the covariance matrix
# if use_price_sensitity_in_sigma:
#     alpha_treat = (alpha_i-alpha_i.mean())*treat_dummies
#     Xsigmafull = T.concatenate([Xsigma, alpha_treat])
#     Xsigmafull_noconst = T.concatenate([Xsigma_noconst, alpha_treat])
# else:
#     Xsigmafull = Xsigma
#     Xsigmafull_noconst = Xsigma_noconst
    
    

var_z00 = T.exp(gamma_e.dot(Xsigma))*alpha_i*alpha_i + 1
var_z11 = T.exp(gamma_m.dot(Xsigma))*alpha_i*alpha_i + T.exp(lsigma_m*2)
cov_z10 = T.tanh(gamma_em.dot(Xsigma))*T.sqrt(var_z00*var_z11)*alpha_i*alpha_i + T.tanh(atsigma_em)*T.exp(lsigma_m)

# note that the above covariance matrix correponds to mean utiltity relative
# to the first alternative (gasoline)

#%%


# Cholesky decomposition for 2x2 matrix
# see https://algowiki-project.org/en/Cholesky_method
# V = [v00 v01, v10 v11], S = [s00 0, s10 s11], S'S = V
# then s00 = sqrt(v00), s10 = v10/s00, s11 = sqrt(v11 - s10^2)
# each of the following is a vector of the length N
s00 = T.sqrt(var_z00)
s10 = cov_z10/s00
s11 = T.sqrt(var_z11 - s10**2 + 1e-6)
s01 = T.zeros((N,))

S = (T.stack([s00, s01, s10, s11]) # size (4, N)
     .transpose() # size (N, 4)
     .reshape((N,2,2))) # size (N, 2, 2)

#%%

# Calculat the covariance matirx of the mean utility relative to each motorist
# chosen alternative

# Transformed Cholesky matrices
MS = T.batched_dot(Mi, S)   # MS_i = Mi_i*S_i
                            # Mi is (N, 2, 2), S is (N, 2, 2)
                            # batched_dot: matrices multiplication in batches 
                            #              along the first dimension


# Calculate the transformed variance matrices for each motorist i:
# Sigma_i = (M_i*S_i)'*(M_i*S_i) = MS_i'*MS_i
# MS is (N, 2, 2) or MS.dimshuffle((0,2,1)) is (N, 2, 2) but transpose 
# the second and third dimension
Sigma = T.batched_dot(MS, MS.dimshuffle((0,2,1)))

# Cholesky decomposition (see the note above)
# These vectors will be used in the GHK simulator
c00 = T.sqrt(Sigma[:,0,0]) + 1e-4
c10 = Sigma[:,1,0]/c00
c11 = T.sqrt(Sigma[:,1,1] - c10**2) + 1e-4

#%%

normcdf = lambda x: 0.5 + 0.5*T.erf(x/np.sqrt(2))
norminv = lambda p: np.sqrt(2)*T.erfinv(2*p-1)

ndraws = spec['ndraw'] if 'ndraw' in spec else 100

# random draws
#draws = np.random.random((ndraws,N)) 

# hammersley draws
draws = (np.tile(np.arange(ndraws), (N,1)).transpose() + 0.5)/ndraws 

# GHK simulator for 3 choices
prob0 = normcdf(-Vnonchoice[0,:]/c00)
prob1 = normcdf(-(Vnonchoice[1,:] + c10*norminv(draws*prob0))/c11).mean(axis=0)

if use_fe:
    nlogl_i = -T.log(prob0)*mask[0] - T.log(prob1)*mask[1]
    nlogl = -T.log(prob0).sum() - T.log(prob1[mask[1]]).sum()
else:
    nlogl_i = -T.log(prob0) - T.log(prob1)
    nlogl = -T.log(prob0).sum() - T.log(prob1).sum()


# theano function to calculate the log likelihood
eval_f = theano.function([theta], outputs = nlogl)

# automatic first and second derivatives
grad = theano.function([theta], outputs = T.grad(nlogl, [theta]))
hess = theano.function([theta], outputs = theano.gradient.hessian(nlogl, [theta]))


# theano gradient returns a matrix instead of a vector, convert it to a vector
eval_grad = lambda t: np.squeeze(grad(t))

# theano hessian returns a 3D tensor, convert it to a matrix
eval_hess = lambda t: np.squeeze(hess(t))

#%% 

# dv_control = df.treattype.values == 0
# nlog_control = nlogl_i[dv_control].sum()

# eval_f_control = theano.function([theta], outputs = nlog_control)
# grad_control = theano.function([theta], outputs = T.grad(nlog_control, [theta]))
# hess_control = theano.function([theta], outputs = theano.gradient.hessian(nlog_control, [theta]))

# eval_grad_control = lambda t: np.squeeze(grad_control(t))
# eval_hess_control = lambda t: np.squeeze(hess_control(t))

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

# mean utility, using each of the alternatives as base
# i.e. Vallabse is a N*J*(J-1) matrix: Vallbase(ijk) =  V(ij) - V(ik)
Vallbase        = T.dot(M, V)

# GHK simulator to approximate the likelihood
p0allbase       = T.maximum(normcdf(-Vallbase[:,0,:]/c00), 1e-8)
#drawsallbase    = np.random.random((ndraws,J,N)) # uniform random draws
drawsallbase    =  (np.tile(np.arange(ndraws), (N,J,1)).transpose() + 0.5)/ndraws # hammersley draws
draws1allbase   = norminv(drawsallbase*p0allbase)
p1allbase       = normcdf(-(Vallbase[:,1,:] + c10*draws1allbase)/c11).mean(axis=0)

# probability of choosing each of the alternative
pallbase = p0allbase*p1allbase
#
#if use_fe and use_share_moments:    
#    pstation = T.stack([pallbase[1:,np.where(stationid==i)[0]].mean(axis=1) for i in range(M)]).transpose().flatten()[(xi_identified).flatten().nonzero()[0]]
#    pstationtrue = np.stack([dv_choice[1:,stationid==i].mean(axis=1) for i in range(M)]).transpose().flatten()[xi_identified.flatten()]
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
#    nxifull = (J-1)*M
#    mask00 = np.ones((ntheta1, ntheta1), dtype = bool)
#    mask01 = np.ones((ntheta1, nxi), dtype = bool)
#    mask10 = np.ones((nxi, ntheta1), dtype = bool)
#    mask11 = np.tile(np.eye(M, dtype = bool), (J-1, J-1))[xi_identified.flatten(),:][:,xi_identified.flatten()]
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

if any(p in purpose for p in ['se', 'mfx', 'display', 'cf']):
    print("Calculating clustered standard errors")
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
    
if any(x in purpose for x in ['mfx', 'cf']): 
    eval_mean_pallbase = theano.function([theta], pallbase.mean(axis=1))
    eval_dmean_pallbase = theano.function([theta], T.jacobian(pallbase.mean(axis=1),[theta])[0])


tensor_lbls_dict = {price: pricelbls, X: Xlbls, Xp: Xplbls, Xsigma: Xsigmatreatlbls,
                    Xsigma_noconst: Xsigmatreatlbls_noconst, treat_dummies: treatlbls}

def df2tensor2(x):
    return df[x].values.transpose().astype(floatX)


def reset_values():
    for tensor, lbls in tensor_lbls_dict.items():
        tensor.set_value(df2tensor2(lbls))        

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
#    pricecf= np.array(price) # note: to be fixed due to refactoring Tprice to price
#    pricecf[0,:] = np.log(pr)
#    price.set_value(pricecf)
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
    #i1 = np.zeros(ngroup*(J-1)-1, dtype=int) # base alternative = 0
    #i2 = np.tile(np.arange(ngroup, dtype=int), J-1)[1:] # groupid
    #i3 = np.repeat(np.arange(J-1, dtype=int), ngroup)[1:] # variance (diagonal element of each choice)
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
        
    for j in range(J-1):
        idx = range(j*nX, (j+1)*nX)
        print_result_group("utiltiy of " + choicelbls[j],
                           betahat[idx], betase[idx], betatstat[idx], Xlbls)
    #
    #print_result_group("variance of random utility, " + choicelbls[0],
    #                   Sigmahat[:ngroup-1], Sigmase[:ngroup-1], Sigmatstat[:ngroup-1],
    #                   ["Treatment " + str(j) for j in range(1,ngroup)])
    #
    #for j in range(1,J-1):
    #    idx = range(j*ngroup-1, (j+1)*ngroup-1)
    #    print_result_group("variance of random utility, " + choicelbls[j],
    #                       Sigmahat[idx], Sigmase[idx], Sigmatstat[idx],
    #                       ["Treatment " + str(j) for j in range(ngroup)])
    #
    #for j in range(J-1):
    #    idx = range(j*(ngroup*2//3), (j+1)*(ngroup*2//3))
    #    print_result_group("effect on log(variance of random utility, " + choicelbls[j] + ")",
    #                       effecthat[idx], effectse[idx], effecttstat[idx],
    #                       ["Treatment " + str(j) for j in range(1,ngroup)])
    #
    
    Xsigmatreatlbls_extra = ["alpha_i*" + x for x in treatlbls] if use_price_sensitity_in_sigma else []
        
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


#Xsigma = theano.shared(df[Xsigmatreatlbls].values.transpose().astype(floatX))

#Xsigmatreatlbls = []
#for x in treatlbls:
#    for y in Xsigmalbls:
#        df[y+"*"+x] = df[x]*df[y]
#        Xsigmatreatlbls.append(y+"*"+x)
        
#%%
#if 'cf' in purpose:
#    group_dummies_cf = {x: treat_dummies.copy() for x in ['control'] + treatlbls.tolist()}
#    Xsigma_cf = {x: df[Xsigmatreatlbls].copy() for x in ['control'] + treatlbls.tolist()}
#    
#    for x_cf in treatlbls:
#        for x in treatlbls:
#            group_dummies_cf[x_cf][x] = x_cf == x
#        
#        for x in treatlbls:
#            for y in Xsigmalbls:
#                Xsigma_cf[x_cf][y+"*"+x] = df[y]*(x == 'treat_1')
#    
#        
#        Xsigma.set_value(Xsigma_cf[x_cf].values.transpose().astype(floatX))
#        treat_dummies.set_values(group_dummies_cf[x_cf].values.transpose().astype(floatX))
    
#%%    
if 'se' in purpose:
    with open(resultfile, 'w') as outfile:
        json.dump({'thetahat':thetahat.tolist(), 'thetase': sehat.tolist(), 'specfile': specfile}, outfile, indent=2)

#%%
from plotnine import ggplot, aes, geom_line, xlab, ylab, scale_linetype_discrete, theme, element_blank, geom_histogram, facet_wrap

if 'sharecf' in purpose:
    pratio_range = np.arange(50,90)/100.0
    share_cf = []
    for pratio in pratio_range:
        pe_lt_cf = df.pg_lt*pratio
        pe_km_adj_cf = df.pe_km_adj/df.pe_lt*pe_lt_cf
        df_cf = df[pricelbls].copy()
        df_cf['rel_pe_km_adj'] = pe_km_adj_cf - df.pg_km_adj
        share_cf_i = []
        for treattype in ['control'] + treatlbls.tolist():
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



    plot = ggplot(sharecf_results_cf_long[sharecf_results_cf_long.treattype != 'treat2']) + \
        aes(x='share_e', y='pratio', group='treattype') + \
        geom_line(aes(linetype='treattype')) + \
        ylab("$p_e/p_g$") + xlab("ethanol share") + \
        scale_linetype_discrete(labels=("Control", "Price-ratio treatment")) + \
        theme(legend_title=element_blank(), legend_position = (0.7, 0.75))
    
    # plot.save(resultfile.replace('_results.json', 'sharecfe.pdf'))
    plot.save(resultfile.replace('_results.json', 'sharecfe.pdf'))
    
    plot = ggplot(sharecf_results_cf_long[sharecf_results_cf_long.treattype != 'treat2']) + \
        aes(x='share_g', y='pratio', group='treattype') + \
        geom_line(aes(linetype='treattype')) + \
        ylab("$p_e/p_g$") + xlab("gasoline share") + \
        scale_linetype_discrete(labels=("Control", "Price-ratio treatment")) + \
        theme(legend_title=element_blank(), legend_position = (0.3, 0.75))
    # plot.save(resultfile.replace('_results.json', 'sharecfg.pdf'))
    plot.save(resultfile.replace('_results.json', 'sharecfg.pdf'))

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
    draws_cf = np.random.normal(size=(ndraws_cf, N, 2))
    
    
            
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

#%% TRANSFER


choisetcf2_indivtransfer_70pg_df = pd.DataFrame()
pe_km_adj_cf = df.pe_km_adj/df.pe_lt*(df.pg_lt*0.7)
for treattype in ['control', 'treat_1']:
    mean_utility_fullchoiceset = mean_utility_cf(pe_km_adj_cf, treattype, axes=0)
    
    mean_utility_newchoiceset = lambda transfer, theta: mean_utility_cf(pe_km_adj_cf-transfer, treattype, choiceset['no_fossil'], axes=0, theta=theta)
    find_transfer = lambda theta: binsearch_vector(lambda transfer: mean_utility_newchoiceset(transfer, theta), mean_utility_fullchoiceset, np.ones_like(mean_utility_fullchoiceset)*(-3), np.ones_like(mean_utility_fullchoiceset)*10).mean()
    transfer_solution = find_transfer(thetahat)
    grad_transfer_solution = findiff(find_transfer, thetahat)
    transfer_se = np.squeeze(np.sqrt(np.dot(np.dot(grad_transfer_solution.transpose(), covhat), grad_transfer_solution)))
    
    print('transfer = {:.5f}, se = {:.5f}'.format(transfer_solution, transfer_se))
    
    choisetcf2_indivtransfer_70pg_df = choisetcf2_indivtransfer_70pg_df.append({
            'treattype': treattype, 
            'choiceset': choicesetcode,
            'pe_transfer': transfer_solution,
            'pe_transfer_se': transfer_se}, ignore_index=True)

choisetcf2_indivtransfer_70pg_df.to_csv(resultfile.replace('_results.json', 'choisetcf2_indivtransfer_70pg.csv'))

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
