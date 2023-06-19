import sys
sys.path.append("../../")

from spectrome.utils import functions, path
import numpy as np
import xarray as xr
from scipy.io import loadmat

from spectrome.optim import sgmglobaloptim, sgmglobaloptim_pearson
from scipy.optimize import dual_annealing
from spectrome.brain import Brain
# from spectrome.stability import localstability_microintensity_allrois
from spectrome.stability import localstability

import time

import itertools
import multiprocessing
from functools import partial

start = time.time()


sleep = loadmat('/protected/data/rajlab1/user_data/parul/sleep_files/PSD_pmtm_RP.mat')
print("Loaded sleep data", flush= True)

Data = sleep['PSD']

N1 = Data[:,0]
N2 = Data[:,1]
wake = Data[:,3]

temp = np.squeeze(N1[0][1])
psd_N1 = temp['REL_psdx']

fvec = temp['freqs'][0][:,0]

print("shape of N1:", psd_N1.shape)
print("shape of fvec:", fvec.shape)

# Take frequencies till 35 Hz
fvec = fvec[0:30]

print("after cutting off till 30 Hz fvec shape:", fvec.shape)
print("Last frequency in fvec:", fvec[-1])

nsubs = len(psd_N1)
nroi = len(psd_N1[0])
print("nsubs len:", nsubs)
print("nroi len:", nroi)

temp = np.squeeze(N2[0][1])
psd_N2 = temp['REL_psdx']

print("shape of N2:", psd_N2.shape)

temp = np.squeeze(wake[0][1])
psd_wake = temp['REL_psdx']

print("shape of wake:", psd_wake.shape)

start = time.time()

# cortical areas with MEG collected + source reconstructed
rois_with_MEG = np.arange(0,nroi)

paramlist = range(nsubs)

# sc_bn = loadmat("/protected/data/rajlab1/shared_data/datasets/brainnetome_template/SC_HC_BN_template.mat")
sc_bn = loadmat("/protected/data/rajlab1/shared_data/datasets/brainnetome_template/SC_HC_BN_roivolnorm_template.mat")

# conn = sc_bn['SC_template']
# dist = sc_bn['Dist_template']

conn = sc_bn['SC_template_volnorm']
dist = sc_bn['Dist_template']

# create spectrome brain:
brain = Brain.Brain()
# brain.reducedConnectome = conn
brain.connectome = conn
brain.distance_matrix = dist
brain.bi_symmetric_c_BN()
brain.reduce_extreme_dir()

print(brain.reducedConnectome.shape)
print(brain.distance_matrix.shape)

mica_micro_intensity = np.squeeze(loadmat('/protected/data/rajlab1/shared_data/datasets/MICA/micro_intensity_mean_BN_subcort.mat')['micro_intensity_mean_subcort'])

# Parameter bounds for optimization
# bnds = ((5.0,20.0), (5.0,20.0), (0.1,1.0), (5.0,20.0), (0.5,5.0), (0.5,5.0), (5.0,20.0))
# bnds = ((5.0,30.0), (5.0,200.0), (0.1,1.0), (0.001,10.0), (0.5,10.0), (0.5,10.0), (5.0,30.0))

#  AD bounds
# v_lower = 3.5-1.8
# v_upper = 3.5+1.8
# bnds = ((5.0,30.0), (5.0,200.0), (0.1,1.0), (v_lower,v_upper), (0.5,10.0), (0.5,10.0), (5.0,30.0))
v_lower = 5
v_upper = 20
# bnds = ((5.0,30.0), (5.0,200.0), (0.1,1.0), (v_lower,v_upper), (0.001,0.5), (0.001,1.5), (5.0,30.0))
bnds1 = ((5.0,30.0), (5.0,200.0), (0.1,1.0), (v_lower,v_upper), (0.001,0.7), (0.001,2.0), (5.0,30.0))
bnds2 = ((5.0,30.0), (5.0,200.0), (0.1,1.0), (v_lower,v_upper), (0.001,0.5), (0.001,1.5), (5.0,30.0))
bnds3 = ((5.0,30.0), (5.0,200.0), (0.1,1.0), (v_lower,v_upper), (0.001,0.4), (0.001,1.5), (5.0,30.0))
bnds4 = ((5.0,30.0), (5.0,200.0), (0.1,1.0), (v_lower,v_upper), (0.001,0.3), (0.001,1.5), (5.0,30.0))

# Trying stable bounds for g
# bnds = ((5.0,30.0), (5.0,200.0), (0.1,1.0), (v_lower,v_upper), (0.001,0.6), (0.5,1.5), (5.0,30.0))


# Initial guesses taue, taui, alpha, speed, gei, gii, beta
# Changing initial bound of taui from 3 to 5.

def inguess(i):
    if i==0:
        allx0 = np.array([15.0, 10.0, 1.0, 5, 0.3, 0.6, 6.0])
    if i==1:
        allx0 = np.array([25.0, 80.0, 0.5, 10, 0.2, 0.1, 15.0])
    if i==2:
        allx0 = np.array([6.0, 150.0, 0.1, 18, 0.1, 1.2, 25.0])
    return allx0

# Function for calculating how many times the parameters hit the bounds
def flagout(optres,bnd):
    return sum(optres["x"][i] in [bnd[i][0], bnd[i][1]] for i in range(7))
    
# Optimization function
def optsgm(brain,psd,rois_with_MEG,fvec,s,bnds):
    print('optsgm', os.getpid(), s)

    F_ind = psd[s][:,0:30] # grab current subject's MEG

    F_ind_db = 10*np.log10(F_ind)

    opt_res0 = dual_annealing(
        sgmglobaloptim.global_corr,
        x0=inguess(0),
        bounds=bnds,
        args=(brain,F_ind_db,F_ind,rois_with_MEG,fvec),
        # maxiter=1000,
        maxiter=500,
        initial_temp=5230.0,
        seed=24,
        visit=2.62,
        no_local_search=False
    )

    # print('sub:',s,'opt_res0:',opt_res0["message"])

    opt_res1 = dual_annealing(
        sgmglobaloptim.global_corr,
        x0=inguess(1),
        bounds=bnds,
        args=(brain,F_ind_db,F_ind,rois_with_MEG,fvec),
        # maxiter=1000,
        maxiter=500,
        initial_temp=5230.0,
        seed=24,
        visit=2.62,
        no_local_search=False
    )

    # print('sub:',s,'opt_res1:',opt_res1["message"])

    opt_res = opt_res0 if opt_res0["fun"] < opt_res1["fun"] else opt_res1

    opt_res2 = dual_annealing(
        sgmglobaloptim.global_corr,
        x0=inguess(2),
        bounds=bnds,
        args=(brain,F_ind_db,F_ind,rois_with_MEG,fvec),
        # maxiter=1000,
        maxiter=500,
        initial_temp=5230.0,
        seed=24,
        visit=2.62,
        no_local_search=False
    )

    # print('sub:',s,'opt_res2:',opt_res2["message"])

    if opt_res2["fun"] < opt_res["fun"]:
        opt_res = opt_res2

            # accept=-1e4,

    print('printing optim result')
    # When only using first set of initial guesses
    # opt_res = opt_res2
    print(opt_res)

    tau_e = opt_res["x"][0]
    tau_i = opt_res["x"][1]
    alpha = opt_res["x"][2]
    speed = opt_res["x"][3]
    gei = opt_res["x"][4]
    gii = opt_res["x"][5]
    tauC = opt_res["x"][6]

    f = flagout(opt_res,bnds)

    # print(opt_res["message"])
    
    
    rcorr, spcorr = sgmglobaloptim_pearson.global_corr(opt_res["x"], brain, F_ind_db, F_ind, rois_with_MEG, fvec)
    

    return [
        tau_e,
        tau_i,
        alpha,
        speed,
        gei,
        gii,
        tauC,
        -opt_res["fun"],
        rcorr,
        spcorr,
        s,
        f,
        opt_res["status"],
        opt_res["success"],
    ]

def optsgm_st(brain,psd,rois_with_MEG,fvec,mica_micro_intensity,s):
    
    res = optsgm(brain,psd,rois_with_MEG,fvec,s,bnds1)
    
    brain.ntf_params["tau_e"] = res[0]
    brain.ntf_params["tau_i"] = res[1]
    brain.ntf_params["gei"] = res[4]
    brain.ntf_params["gii"] = res[5]
    
    st = localstability.local_stability(brain.ntf_params,mica_micro_intensity)
    
    if st>0:
        res = optsgm(brain,psd,rois_with_MEG,fvec,s,bnds2)

    brain.ntf_params["tau_e"] = res[0]
    brain.ntf_params["tau_i"] = res[1]
    brain.ntf_params["gei"] = res[4]
    brain.ntf_params["gii"] = res[5]

    st = localstability.local_stability(brain.ntf_params,mica_micro_intensity)

    if st>0:
        res = optsgm(brain,psd,rois_with_MEG,fvec,s,bnds3)

    brain.ntf_params["tau_e"] = res[0]
    brain.ntf_params["tau_i"] = res[1]
    brain.ntf_params["gei"] = res[4]
    brain.ntf_params["gii"] = res[5]

    st = localstability.local_stability(brain.ntf_params,mica_micro_intensity)

    if st>0:
        res = optsgm(brain,psd,rois_with_MEG,fvec,s,bnds4)

    return res

if __name__ == '__main__':
    print('Starting main')
    import os
    # print('Available cores:', os.sched_getaffinity(0))

    # print('PID', os.getpid())

    #Generate processes equal to the number of cores
    pool = multiprocessing.Pool(14)

    # print('doing param optimization with 01 correlation weighted with structural connectome directly with model out')
    print('this is with original model but macro stable')
    # Distribute the parameter sets evenly across the cores

    # print(res)
    func = partial(optsgm_st, brain, psd_wake, rois_with_MEG, fvec, mica_micro_intensity)
    res  = pool.map(func,paramlist)
    # pool.close()
    res2 = np.array(res)
    np.savetxt("/protected/data/rajlab1/user_data/parul/spectromeP_results/results_globalSGM/alpha_experiments/sleep_wake_orgSGM_relspec_roivolnorm_reducedconnectome_distancecost.csv", res2, delimiter=",",header="taue, taui, alpha, speed, gei, gii, tauC, r_tot, r_psd, r_sp, sub, flag, status, success")

    print("Finished Chang data optimization for MSGM")
  
    end = time.time()
    print("T is 5230 and visit is 2.62 and maxiter is 500 and dual annealing with three initial conditions")

    print(f"Runtime of the global sgm optimization with simulated annealing for all subjects is {(end - start)/3600} hours")



