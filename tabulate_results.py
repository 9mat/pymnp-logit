# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:56:26 2019

@author: long
"""
import json
import numpy as np

specfolder = '../spec'
specfiles = ['spec6.json', 'spec13.json']

Xlbls = []
Xplbls = []
Xsigmalbls = []
use_price_sensitity_in_sigma = False

for specfile in specfiles:
    with open(specfolder + '/' + specfile, 'r') as f:
        spec = json.load(f)
                
    Xlbls += [x for x in spec['X'] if x not in Xlbls]
    Xplbls += [x for x in spec['Xp'] if x not in Xplbls]
    Xsigmalbls += [x for x in spec['Xsigma'] if x not in Xsigmalbls]
    
    use_price_sensitity_in_sigma |= ('price_sensitity_in_sigma' in spec and spec['price_sensitity_in_sigma'])
    
    
def unpack_params(theta, spec):
    theta = np.array(theta)
    use_price_sensitity_in_sigma = ('price_sensitity_in_sigma' in spec and spec['price_sensitity_in_sigma'])
        
    nX, nXp, nchoice  = len(spec['X']), len(spec['Xp']), len(spec['price']) + 1

    nalpha  = nXp
    nbeta   = nX*(nchoice-1)
    
    ngroup  = 3
    
    ngamma = len(spec['Xsigma']) + use_price_sensitity_in_sigma
    ngamma_e = ngamma_m = ngamma_em = ngamma*(ngroup-1)

    offset = 0
    alpha  = theta[offset: offset + nalpha]
    
    offset += nalpha
    beta   = theta[offset: offset + nbeta].reshape((nchoice-1, nX))
    
    offset += nbeta
    sigma_10 = theta[offset]
    sigma_11 = theta[offset+1]
    
    v_m = sigma_10**2 + np.exp(sigma_11)**2
    c_em = sigma_10
    
    offset += 2
    gamma_e = theta[offset: offset + ngamma_e]
    
    offset += ngamma_e
    gamma_m = theta[offset: offset + ngamma_m]
    
    offset += ngamma_m
    gamma_em = theta[offset: offset + ngamma_em]
    
    offset += ngamma_em
    
    return {'a': alpha, 'b_e': beta[0],  'b_m': beta[1], 'g_e': gamma_e, 'g_m': gamma_m, 'g_em': gamma_em, 'v_m': [v_m], 'c_em': [c_em]}

Xsigmatreatlbls = []
for x in ['treat_1', 'treat_2']:
    for y in Xsigmalbls:
        Xsigmatreatlbls.append(x+"*"+y)


Xsigmatreatlbls_extra = ["alpha_i*" + x for x in ['treat_1', 'treat_2']] if use_price_sensitity_in_sigma else []
Xsigmatreatlbls += Xsigmatreatlbls_extra
param_sets = {'a': Xplbls, 'b_e': Xlbls, 'b_m': Xlbls, 'g_e':Xsigmatreatlbls, 'g_m': Xsigmatreatlbls, 'g_em': Xsigmatreatlbls, 'v_m': ['const'], 'c_em': ['const']}
estimates = {param: {x: [] for x in covar_set} for param, covar_set in param_sets.items()}



for specfile in specfiles:
    with open(specfolder + '/' + specfile, 'r') as f:
        spec = json.load(f)
        
    with open(spec['resultfile'], 'r') as f:
        results = json.load(f)
        
    theta = results['thetahat']
    se = results['thetase']
#    se = np.ones(len(theta))

    Xlbls = spec['X']
    Xplbls = spec['Xp']
    Xsigmalbls = spec['Xsigma']
    use_price_sensitity_in_sigma = ('price_sensitity_in_sigma' in spec and spec['price_sensitity_in_sigma'])
    
    Xsigmatreatlbls = []
    for x in ['treat_1', 'treat_2']:
        for y in Xsigmalbls:
            Xsigmatreatlbls.append(x+"*"+y)
        
    Xsigmatreatlbls_extra = ["alpha_i*" + x for x in ['treat_1', 'treat_2']] if use_price_sensitity_in_sigma else []
    Xsigmatreatlbls += Xsigmatreatlbls_extra
    
    param_sets = {'a': Xplbls, 'b_e': Xlbls, 'b_m': Xlbls, 'g_e':Xsigmatreatlbls, 'g_m': Xsigmatreatlbls, 'g_em': Xsigmatreatlbls, 'v_m': ['const'], 'c_em': ['const']}
    
    unpacked_theta = unpack_params(theta, spec)
    unpacked_se = unpack_params(se, spec)
    
    for param in estimates:
        for x in estimates[param]:
            estimates[param][x].append((np.nan, np.nan))
            
        for i, x in enumerate(param_sets[param]):
            estimates[param][x][-1] = ((unpacked_theta[param][i], unpacked_se[param][i]))
    
        
    
param_set_title ={'a': 'price sensitity', 
                  'b_e': 'ethanol mean utility',
                  'b_m': 'midgrade-g mean utility',
                  'g_e': 'variance of ethanol random utility (log)',
                  'g_m': 'variance of midgrade-g random utility (log)',
                  'g_em': 'correlation between random utilities of two fuels (atanh)',
                  'v_m': 'variance of ethanol random utility for control group',
                  'c_em': 'covariance between random utilities of two fuels for control group'}

def sig_level(t):
    return 3 if t > 2.56 else 2 if t > 1.96 else 1 if t > 1.65 else 0

def display_txt():
    label_col_size = max(max(len(lbl) for lbl in lbls) for param, lbls in param_sets.items())+5
    est_col_width = 16
    
    n_cols = len(specfiles)
    table_width = label_col_size + n_cols*est_col_width + 5
    
    print("="*table_width)
    est_line_format = " "*5 + "{:<" + str(label_col_size) + "s}" + ("{:>" + str(est_col_width) + "s}")*n_cols

    star_symbols = ["   ", "*  ", "** ", "***"]
    
    for param, x_list in estimates.items():
        print(param_set_title[param])
        print('.'*table_width)
        for x, estimate_list in x_list.items():
            ests, ses = list(zip(*estimate_list))
            ts = np.abs(np.array(ests)/np.array(ses))
            stars = [star_symbols[sig_level(t)] for t in ts]
            eststrs = ["{:.3f}".format(est)+star if not np.isnan(est) else "" for est, star in zip(ests, stars)]
            sestrs = ["[" + "{:.3f}".format(se) + "]  " if not np.isnan(se) else "" for se in ses]
            print(est_line_format.format(x, *eststrs))
            print(est_line_format.format("", *sestrs))
        print('-'*table_width)
    

def display_csv():    
    for param, x_list in estimates.items():
        print(param_set_title[param])
        for x, estimate_list in x_list.items():
            ests, ses = list(zip(*estimate_list))
            ts = np.abs(np.array(ests)/np.array(ses))
            stars = ['***' if t > 2.56 else '** ' if t > 1.96 else '*  ' if t > 1.65 else '   ' for t in ts]
            eststrs = ["{:.3f}".format(est)+star if not np.isnan(est) else "" for est, star in zip(ests, stars)]
            sestrs = ["[{:.3f}]".format(se)  if not np.isnan(se) else "" for se in ses]
            print(','.join([x] + eststrs))
            print(','.join([""] + sestrs))


def display_tex():
    n_cols = len(specfiles)
    print("{")
    print("\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}")
    print("\\begin{tabular}{l" + "c"*n_cols + "}")
    print("\\toprule")
    print("&" + "&".join("({})".format(i) for i in range(1, n_cols+1))+"\\\\")
    print("\\midrule")
    star_symbols = ["", "\\sym{*}", "\\sym{**}", "\\sym{***}"]
    for param, x_list in estimates.items():
        print("\\multicolumn{{{}}}{{l}}{{{}}}".format(n_cols+1, param_set_title[param]) + "\\\\")
        for x, estimate_list in x_list.items():
            ests, ses = list(zip(*estimate_list))
            ts = np.abs(np.array(ests)/np.array(ses))
            stars = [star_symbols[sig_level(t)] for t in ts]
            eststrs = [("{:.3f}".format(est) + star) if not np.isnan(est) else "" for est, star in zip(ests, stars)]
            sestrs = ["({:.3f})".format(se)  if not np.isnan(se) else "" for se in ses]
            print('&'.join([x] + eststrs) + "\\\\")
            print('&'.join([""] + sestrs) + "\\\\")
        print("\\midrule")
        print("\\bottomrule")
    print("\\end{tabular}")    
    print("}")
    
display_txt()
            
