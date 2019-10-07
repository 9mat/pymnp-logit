# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 23:21:11 2019

@author: long
"""

try:
    #from knitro.numpy import Variables, Callback, KN_RC_EVALFC, KN_RC_EVALGA, KN_RC_EVALH, KN_RC_EVALH_NO_F, KN_DENSE, KN_DENSE_ROWMAJOR, optimize, KN_OUTLEV_ALL
    
    from knitro.numpy import *
    import os.path
    
    def solve_unconstr(theta0, eval_f, eval_grad, eval_hess=None):

        def callbackEvalF (kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != KN_RC_EVALFC:
                print ("*** callbackEvalF incorrectly called with eval type %d" % evalRequest.type)
                return -1
            
            evalResult.obj = eval_f(evalRequest.x)
            return 0

        
        def callbackEvalG (kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != KN_RC_EVALGA:
                print ("*** callbackEvalG incorrectly called with eval type %d" % evalRequest.type)
                return -1
        
            evalResult.objGrad = eval_grad(evalRequest.x)
            return 0

        
        def callbackEvalH (kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != KN_RC_EVALH and evalRequest.type != KN_RC_EVALH_NO_F:
                print ("*** callbackEvalH incorrectly called with eval type %d" % evalRequest.type)
                return -1
            
            h = eval_hess(evalRequest.x)
        
            offset = 0
            n = len(evalRequest.x)
            
            for row in range(n):
                evalResult.hess[offset:offset+(n-row)] = h[row, row:]*evalRequest.sigma
                offset += n-row
        
            return 0

        
#        variables = Variables(len(theta0), xInitVals = theta0)
#
#        if eval_hess is None:
#            callbacks = Callback(evalObj=True,
#                                 funcCallback=callbackEvalF,
#                                 objGradIndexVars=KN_DENSE,
#                                 gradCallback=callbackEvalG)
#        else:
#            callbacks = Callback(evalObj=True,
#                                 funcCallback=callbackEvalF,
#                                 objGradIndexVars=KN_DENSE,
#                                 gradCallback=callbackEvalG,
#                                 hessIndexVars1=KN_DENSE_ROWMAJOR,
#                                 hessCallback=callbackEvalH,
#                                 hessianNoFAllow=True)
#        
#        options = {'outlev': KN_OUTLEV_ALL}
#        solution = optimize(variables=variables,
#                            callbacks=callbacks,
#                            options=options)
            
        try:
            kc = KN_new ()
        except:
            print ("Failed to find a valid license.")
            quit ()
            
        if os.path.isfile("knitro.opt"):
            KN_load_param_file(kc, "kitro.opt")
        
        KN_add_vars (kc, len(theta0))
        KN_set_var_primal_init_values (kc, xInitVals = theta0)        
        cb = KN_add_eval_callback (kc, evalObj = True, funcCallback = callbackEvalF)        
        KN_set_cb_grad (kc, cb, objGradIndexVars = KN_DENSE, gradCallback = callbackEvalG)
        KN_set_cb_hess (kc, cb, hessIndexVars1 = KN_DENSE_ROWMAJOR, hessCallback = callbackEvalH)
        nStatus = KN_solve (kc)        
        nStatus, objSol, x, lambda_ = KN_get_solution (kc)
            
        return x

except ImportError:
    from pyipopt import set_loglevel, fmin_unconstrained
    
    def solve_unconstr(theta0, eval_f, eval_grad, eval_hess=None):
        set_loglevel(1)
        thetahat , _, _, _, _, fval = fmin_unconstrained(
            eval_f,
            theta0,
            fprime=eval_grad,
            fhess=eval_hess,
        )
    
        return thetahat