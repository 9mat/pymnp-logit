# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:56:26 2019

@author: long
"""
import json
import numpy as np
import sys

specfolder = '../../spec/absprice/run-20191008-10pm'
specfiles = ['spec{}.json'.format(i) for i in ['1nofe',1,4,7]]

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
    ngamma_e = ngamma_m = ngamma_em = ngamma*(ngroup-1) + ngamma - use_price_sensitity_in_sigma
    ngamma_e -= ('const' in spec['Xsigma'])

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
    
    return {'a': alpha, 'b_e': beta[0],  'b_m': beta[1], 'g_e': gamma_e, 'g_m': gamma_m, 'g_em': gamma_em}

Xsigmatreatlbls = Xsigmalbls[:]
for x in ['treat_1', 'treat_2']:
    for y in Xsigmalbls:
        Xsigmatreatlbls.append(x+"*"+y)

Xsigmatreatlbls_noconst = [x for x in Xsigmatreatlbls if x != 'const']
Xsigmatreatlbls_extra = ["alpha_i*" + x for x in ['treat_1', 'treat_2']] if use_price_sensitity_in_sigma else []
Xsigmatreatlbls += Xsigmatreatlbls_extra
Xsigmatreatlbls_noconst += Xsigmatreatlbls_extra

param_sets = {'a': Xplbls, 'b_e': Xlbls, 'b_m': Xlbls, 'g_e':Xsigmatreatlbls_noconst, 'g_m': Xsigmatreatlbls, 'g_em': Xsigmatreatlbls}
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
    
    Xsigmatreatlbls = Xsigmalbls[:]
    for x in ['treat_1', 'treat_2']:
        for y in Xsigmalbls:
            Xsigmatreatlbls.append(x+"*"+y)
        
    Xsigmatreatlbls_noconst = [x for x in Xsigmatreatlbls if x != 'const']
    Xsigmatreatlbls_extra = ["alpha_i*" + x for x in ['treat_1', 'treat_2']] if use_price_sensitity_in_sigma else []
    Xsigmatreatlbls += Xsigmatreatlbls_extra
    Xsigmatreatlbls_noconst += Xsigmatreatlbls_extra

    param_sets = {'a': Xplbls, 'b_e': Xlbls, 'b_m': Xlbls, 'g_e':Xsigmatreatlbls_noconst, 'g_m': Xsigmatreatlbls, 'g_em': Xsigmatreatlbls}
    
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
                  'g_em': 'correlation between random utilities of two fuels (atanh)',}

def sig_level(t):
    t = np.abs(t)
    return 3 if t > 2.56 else 2 if t > 1.96 else 1 if t > 1.65 else 0

def display_txt():
    label_col_size = max(max(len(lbl) for lbl in lbls) for param, lbls in param_sets.items())+5
    est_col_width = 16
    
    n_cols = len(specfiles)
    table_width = label_col_size + n_cols*est_col_width + 5
    
    print("="*table_width)
    est_line_format = " "*5 + "{:<" + str(label_col_size) + "s}" + ("{:>" + str(est_col_width) + "s}")*n_cols

    print(est_line_format.format("", *[f.replace('.json', '')+"   " for f in specfiles]))

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

var_lbls = {
    'dv_age_25to40y': 'Aged 25-40 years',
    'dv_age_40to65y': 'Aged 40-65 years',
    'dv_age_morethan65y': "Aged more than 65 years",
    'dv_somesecondary': 'Secondary School',
    'dv_somecollege': "College educated",
    'car_lprice': 'Vehicle price (1000\\$, log)',
    'treat_1': 'Price-ratio treatment',
    'treat_2': 'km-per-R\\$50 treatment',
    'dv_female': 'Female consumer',
    'dv_usageveh_p75p100': 'Top-quartile vehicle usage',
    'treat_1*const': 'Price-ratio treatment',
    'treat_2*const': 'km-per-R\\$50 treatment',
    'alpha_i*treat_1': 'Price-ratio treatment \\(\\times\\) (\\(\\alpha_i-\\alpha\\)',
    'alpha_i*treat_2': 'km-per-R\\$50 treatment \\(\\times\\) (\\(\\alpha_i-\\alpha\\)',
    'const': 'Constant'
}

def display_tex():
    n_cols = len(specfiles)
    print("\\documentclass{article}")
    print("\\usepackage{booktabs}")
    print("\\begin{document}")
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
            eststrs = [x.replace('_', '\\textunderscore ')] + [("{:.3f}".format(est) + star) if not np.isnan(est) else "" for est, star in zip(ests, stars)]
            sestrs = [""] + ["({:.3f})".format(se)  if not np.isnan(se) else "" for se in ses]
            print('&'.join(map("{:>16}".format, eststrs)) + "\\\\")
            print('&'.join(map("{:>16}".format, sestrs)) + "\\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")    
    print("}")
    print("\\end{document}")


def display_tex_wide():
    n_cols = len(specfiles)
    print("\\documentclass{article}")
    print("\\usepackage{booktabs}")
    print("\\usepackage{fullpage}")
    print("\\begin{document}")
    print("{\\footnotesize\\def\\sym#1{\\ifmmode^{#1}\\else\\(^{#1}\\)\\fi}")
    print("\\begin{tabular}{l" + "lc"*n_cols + "}")
    print("\\toprule")
    print("&" + "&".join("\\multicolumn{{2}}{{c}}{{({})}}".format(i) for i in range(1, n_cols+1))+"\\\\")
    print(" ".join("\\cmidrule(lr){{{}-{}}}".format(2*i,2*i+1) for i in range(1, n_cols+1)))
    print("& Coef.  &   S.E.  "*n_cols + "\\\\")
    print("\\midrule")
    star_symbols = ["", "\\sym{*}", "\\sym{**}", "\\sym{***}"]
    for param, x_list in estimates.items():
        print("\\multicolumn{{{}}}{{l}}{{{}}}".format(n_cols*2+1, param_set_title[param]) + "\\\\")
        for x, estimate_list in x_list.items():
            ests, ses = list(zip(*estimate_list))
            ts = np.abs(np.array(ests)/np.array(ses))
            stars = [star_symbols[sig_level(t)] for t in ts]
            eststrs =  [("\({:.2f}\)".format(est) + star) if not np.isnan(est) else "" for est, star in zip(ests, stars)]
            sestrs = ["({:.2f})".format(se)  if not np.isnan(se) else "" for se in ses]
            items = sum(map(list, zip(eststrs, sestrs)), [])
            label = var_lbls[x] if x in var_lbls else x.replace('_', '\\textunderscore ')
            print("  {:<30}&".format(label) + '&'.join(map("{:>16}".format, items)) + "\\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")    
    print("}")
    print("\\end{document}")
    
            
display_tex_wide()