import numpy as np
from sympy import *
# from tbcontrol.symbolic import routh
import mpmath as mp


def local_stability(parameters,mica_micro_intensity,ex_template,inh_template):
    
    gei = parameters["gei"]  # excitatory-inhibitory synaptic conductance as ratio of E-E syn
    gii = parameters["gii"]  # inhibitory-inhibitory synaptic conductance as ratio of E-E syn
    gee = 1
    st = 0
    
    for i in range(68):
        gee = ex_template[i]
        gii = parameters["gii"]*inh_template[i]
        tau_e = parameters["tau_e"]*mica_micro_intensity[i]/1000
        tau_i = parameters["tau_i"]*mica_micro_intensity[i]/1000

        fe = float(1/tau_e)
        fi = float(1/tau_i)
        s = Symbol('s')


        a = poly(
            ( (s * (s+fe)**2 * (s+fi)**2 + gee * fe**3 * (s+fi)**2) *  
            (s * (s+fe)**2 * (s+fi)**2 + gii * fi**3 * (s+fe)**2) + 
            gei**2 * fe**5 * fi**5 ) / (fe**5 * fi**5), 
            s
        )


        b = a.all_coeffs()
        roots = np.roots(b)
        

        for result in roots:
            if result.real>0:
                st += 1
    
    for i in range(14):
        gee = ex_template[68+i]
        gii = parameters["gii"]*inh_template[68+i]
        tau_e = parameters["tau_e"]
        tau_i = parameters["tau_i"]

        fe = float(1/tau_e)
        fi = float(1/tau_i)
        s = Symbol('s')

        a = poly(
            ( (s * (s+fe)**2 * (s+fi)**2 + gee * fe**3 * (s+fi)**2) *  
            (s * (s+fe)**2 * (s+fi)**2 + gii * fi**3 * (s+fe)**2) + 
            gei**2 * fe**5 * fi**5 ) / (fe**5 * fi**5), 
            s
        )


        b = a.all_coeffs()
        roots = np.roots(b)
        

        for result in roots:
            if result.real>0:
                st += 1
    return st
    # return roots
