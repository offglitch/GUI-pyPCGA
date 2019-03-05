import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
#from scipy.io import savemat, loadmat
import numpy as np
#import csim as cs TO IMPORT CSIM NEEDS TO NOT HAVE ERRORS
import csim as csim 
from pyPCGA import PCGA
import math
import sys
import os
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:01:08 2018

@author: Amalia Kokkinaki

Example file for running PCGA with COMPSIM for DNAPL identification
"""

if __name__ == '__main__':  # for windows application
    # model domain and discretization
    
    #import pdb
    #pdb.set_trace()

    nx = [80, 40]
    m=np.prod(nx)
    
    dx = [0.0625, 0.0625]
    Lx = nx[0]*dx[0]; Ly = nx[1]*dx[1]; nrow = nx[1]; ncol = nx[0]
    N=np.array([ncol,nrow])
    xmin = np.array([0. + dx[0] / 2., 0. + dx[1] / 2.])
    xmax = np.array([Lx - dx[0] / 2., Ly - dx[1] / 2.])
    
    #pts needed for generation of covariance - see tough 2 example
    x = np.linspace(0. + dx[0] / 2., Lx - dx[0] / 2., N[0])
    y = np.linspace(0. + dx[1] / 2., Ly - dx[1] / 2., N[1])
    XX, YY = np.meshgrid(x, y)
    pts = np.hstack((XX.ravel()[:, np.newaxis], YY.ravel()[:, np.newaxis]))
    
    # Observations
    obs = np.loadtxt('obs_true.txt')
    # std_obs = 0.001
    std_obs = float(sys.argv[1])
    case_id = sys.argv[2]
    os.mkdir(case_id)
    std_obs = std_obs * np.ones_like(obs)

    xlocs = np.arange(15,70,5)
    ylocs = np.arange(5,40,5)
    
    
    csim_exec = 'mphMay20e'
    input_dir = "./support_files"
    sim_dir = './simul'
    
    forward_model_params = {'csim_exec': csim_exec, 'input_dir': input_dir,
          'sim_dir': sim_dir,
          'Lx': Lx, 'Ly': Ly,'m':m,'dx':dx,'nx':nx,
          #'InjQ': InjQ, 'InjL': InjL,
          'nrow': nrow, 'ncol': ncol,
          'xlocs':xlocs,'ylocs':ylocs,
          'obs_type':['head']}
          #'obs_locmat': obs_locmat, 'Q_locs': Q_locs}
    


    def kernel(r): return (prior_std ** 2) * np.exp(-r)


    # load true value for comparison purpose
    s_true = np.loadtxt('s_true.txt')
    s_true = np.array(s_true).reshape(-1, 1)  # make it 2D array
    
    # covariance kernel and scale parameters
    prior_std = 1.0
    prior_cov_scale = np.array([5.0, 2.5])
    
    
    #for i in range(5, 71, 16):
    #for j in range(9, 96, 16):
        #obs_locmat[0, i, j] = 1
    
    
    # prepare interface to run as a function
    def forward_model(s, parallelization, ncores=None):
        model = csim.Model(forward_model_params)

        if parallelization:
            simul_obs = model.run(s, parallelization, ncores)
        else:
            simul_obs = model.run(s, parallelization)
        return simul_obs
    
    # parameters for the inversion 
    params = {'R': std_obs ** 2, 'n_pc': 50,
            'maxiter': 5, 'restol': 0.01,
            'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N,
            'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
            'kernel': kernel, 'post_cov': "diag",
            'precond': True, 'LM': True,
            'parallel': True, 'linesearch': True,
            'forward_model_verbose': False, 'verbose': False,
            'iter_save': True,'precision': 1e-4,
            'LM_smin': -34, 'LM_smax': -14}

    # params['objeval'] = False, if true, it will compute accurate objective function
    # params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified

    init_guess = -24.718 # background log permeability average, perm in m2
    s_init = init_guess * np.ones((m, 1))
    # s_init = np.copy(s_true) # you can try with s_true!

    # initialize
    prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
    # prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X

    # run inversion
    s_hat, simul_obs, post_diagv, iter_best = prob.Run()
    
    
    # Post processing of results
    s_hat2d = s_hat.reshape(nrow, ncol)
    s_true2d = s_true.reshape(nrow, ncol)

    post_diagv[post_diagv < 0.] = 0.  # just in case
    post_std = np.sqrt(post_diagv)
    post_std2d = post_std.reshape(nrow, ncol)

    #fig = plt.figure()
    #plot properties
    fig, ax = plt.subplots(1,1)
    fig.suptitle('{}'.format('Estimated logk'), fontsize = 12)
    cax = ax.imshow(s_hat2d.T, extent=[0,5,0,2.5], origin='upper')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    cbar = fig.colorbar(cax, orientation='vertical') 
    #plt.plot(x,s_hat,'k-',label='estimated')
    #plt.plot(x,s_hat + 2.*post_std,'k--',label='95%')
    #plt.plot(x,s_hat - 2.*post_std,'k--',label='')
    #plt.plot(x,s_true,'r-',label='true')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    #plt.legend()
    fig.savefig(case_id + '/best.png', dpi = 300)
    plt.close(fig)

    #fit = plt.figure()
    figs, axs = plt.subplots(1,1)
    figs.suptitle('{}'.format('Posterior uncertainty in logk'), fontsize = 12)
    cax = axs.imshow(post_std2d.T, extent=[0,5,0,2.5], origin='upper')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    figs.savefig(case_id + '/unc.png', dpi = 300)
    plt.close(figs)

    nobs = prob.obs.shape[0]
    fig = plt.figure()
    plt.title('obs. vs simul.')
    plt.plot(prob.obs, simul_obs, '.')
    plt.xlabel('observation')
    plt.ylabel('simulation')
    minobs = np.vstack((prob.obs, simul_obs)).min(0)
    maxobs = np.vstack((prob.obs, simul_obs)).max(0)
    plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
    plt.axis('equal')
    axes = plt.gca()
    axes.set_xlim([math.floor(minobs), math.ceil(maxobs)])
    axes.set_ylim([math.floor(minobs), math.ceil(maxobs)])
    fig.savefig(case_id + '/obs.png', dpi = 300)
    # plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.semilogy(np.linspace(1,len(prob.objvals),len(prob.objvals)), prob.objvals, 'r-')
    plt.xticks(np.linspace(1,len(prob.objvals),len(prob.objvals)))
    plt.title('obj values over iterations')
    plt.axis('tight')
    fig.savefig(case_id + '/obj.png')
    plt.close(fig)
    

    # moving result files to folder case_id
    import glob
    import shutil 
    for f in glob.glob(r'shat*'):
        shutil.move(f, case_id)

    for f in glob.glob(r'simulobs*'):
        shutil.move(f, case_id)

    shutil.move('postv.txt',case_id)



