import numpy as np
import pandas as pd
import theano.gradient
import numpy.random
import glob
import json
import sys
import pyipopt
import os
from time import time
from logittheano import DataStructure, DataContent, buildtheano, buildeval, getthetarange


if len(sys.argv) <= 1:
    specname = 'spec1'
    multistartidx = 0
else:
    specname = sys.argv[1]
    multistartidx = int(sys.argv[2])

print '** reading specifications from ' + specname + '.json'

with open(specname + '.json', 'r') as f:
    spec = json.load(f)
    lbls = spec['label']
    draw = spec['draw']
    inputfile = spec['inputfile']

print '** reading input from ' + inputfile

df = pd.read_csv(inputfile)

# generate const and treatment dummies
df['const'] = 1
df = pd.concat([df, pd.get_dummies(df['treattype'], prefix='dv_treat')], axis=1)

print '** prepare data'
        
n = DataStructure(df, lbls, draw)
data = DataContent(df, lbls, n)

print '** building theano'

theta, nloglf = buildtheano(data, n)
eval_f, eval_grad, eval_hess = buildeval(theta, nloglf)


outdir  = './result/' + specname 

if multistartidx < 0:
    print '** combine data'

    pricelbls, covarlbls, pcovarlbls, zcovarlbls = lbls
    collbls = ['nlogl', '|gradient|']

    for l in pcovarlbls: collbls.append('price*' + l)
    for j in range(n.choice-1):
        for x in covarlbls:
            collbls.append('choice_' + str(j+1) +'*' + x)
    for j in range(n.choice-1):
        for x in zcovarlbls:
            collbls.append('ln(sigma)/choice_' + str(j+1) + '*' + x)

    for j in range(n.choice-2):
        collbls.append('mu/choice_' + str(j+2))

    print ','.join(collbls)

    for fname in glob.glob(outdir + '/result*.npz'):
        result = np.load(fname)


        thetahat = result['theta']
        nlogl = float(result['nlogl'])
        sehat = result['se']
        grad = np.absolute(eval_grad(thetahat)).max()

        print str(nlogl) + ',' + str(grad) + ',' + ','.join(str(thetahat[i]) for i in xrange(thetahat.size))
        print  ' , ,' + ','.join(str(sehat[i]) for i in xrange(sehat.size))

        exit()



print '** generate a random starting point'

thetalower, thetaupper = getthetarange(n)
assert (len(thetalower) == len(thetaupper))

multistart = 1000
np.random.seed(1234)
r = np.random.rand(multistart, len(thetalower))
r = r[multistartidx,:]



print '** invoke ipopt'

start = time()

pyipopt.set_loglevel(1)
thetahatnew , _, _, _, _, fval = pyipopt.fmin_unconstrained(
    eval_f,
    thetalower*r + thetaupper*(1-r),
    fprime=eval_grad,
    fhess=eval_hess,
    )

print '** calculating nlogl and se'

nllnew = eval_f(thetahatnew)
try:
    sehatnew = np.diag(np.linalg.inv(eval_hess(thetahatnew)))**0.5
except Exception, e:
    sehatnew = np.nan*np.ones(thetahatnew.shape)


print '-- Time =', time()-start, 's'

outname = 'result'+ '_' + specname +'_' "{:03d}".format(int(multistartidx))

try: os.makedirs(outdir)
except OSError: 
    if not os.path.isdir(outdir): raise

np.savez_compressed(outdir + '/' + outname, 
    theta=thetahatnew, nlogl=nllnew, se=sehatnew)
