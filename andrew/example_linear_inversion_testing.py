import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import numpy as np
import drawdown as dd
from pyPCGA import PCGA
import math

if __name__ == '__main__':  # for windows application
    # model domain and discretization

    import pdb
    pdb.set_trace()
    
    # This is a 1D case, therefore should be used to test the 1D scenario
    
    ####### BEGINNING OF MODULE 1 #################### 

    # M1 parameters are: Lx, Ly, Lz, x0, y0, z0, dx, dy, dz, s_true, s_init

    x0 = 0    	# M1: Origin of x dimension 
    Lx = 1000 	# M1: Total length in the x direction
    dxx = 0.1 	# M1: Discretization (cell length) in the x direction, assumes cells of equal size

    # This simulation is 1D, therefore default to y_origin = z_origin = 0, Ly = Lz = 1, dy = dz = 1

    y0 = 0    	# M1: Origin of y dimension 
    Ly = 1 	# M1: Total length in the y direction
    dyy = 1 	# M1: Discretization (cell length) in the y direction, assumes cells of equal size

    z0 = 0    	# M1: Origin of y dimension 
    Lz = 1 	# M1: Total length in the y direction
    dzz = 1 	# M1: Discretization (cell length) in the z direction, assumes cells of equal size

    xmin = np.array([x0]) 
    xmax = np.array([Lx])
    m= int(Lx/dxx + 1) 
    N = np.array([m])
    dx = np.array([dxx])
    x = np.linspace(xmin, xmax, m)
    pts = np.copy(x)
 

    s_true = np.loadtxt('true.txt') # M1: input for file "true.txt"  

    # s_init, three options (drop down menu) 
    # option 1: user inputs a constant which gets assigned to variable s_constant 

    s_constant = 1		    # M1: User selects constant checkbox from drop down, and inputs number in box 
    s_init = s_constant * np.ones((m, 1))

    # option 2: s_init automatically calculated using s_true, if s_true provided
    # # M1: User selects Auto checkbox from drop down, and check is run to see if s_true was provided 
    print(m)
    s_init = np.mean(s_true) * np.ones((m, 1)) #M1 file input or constant input
    # s_init = np.copy(s_true) # you can try with s_true!
 
    
    ### PLOTTING FOR 1D MODULE 1 #############
    
    fig = plt.figure()
    plt.plot(x,s_init,'k-',label='initial')
    plt.plot(x,s_true,'r-',label='true')
    plt.title('Pumping history')
    plt.xlabel('Time (min)')
    plt.ylabel(r'Q ($m^3$/min)')
    plt.legend()
    fig.savefig('best.png')
    plt.close(fig)	

    ####### END OF MODULE 1 #################### 
   
    # xloc,yloc,zloc are uniformly distributed
    #xloc = [xmin:10:xmax]

    #import pdb
    #pdb.set_trace()
    # covarIance kernel and scale parameters

    #prior_std = 0.04 #Module 4 (R) 
    #prior_cov_scale = np.array([200.0]) #M4 lambdas, lx, ly, lz

    #def kernel(r): return (prior_std ** 2) * np.exp(-r)  # M4Kernel use switch function

   
    #obs = np.loadtxt('obs.txt') # M3 file input

    # prepare interface to run as a function
    #def forward_model(s, parallelization, ncores=None):
    #    params = {}
    #    model = dd.Model(params)
    #
    #    if parallelization:
    #        simul_obs = model.run(s, parallelization, ncores)
    #    else:
    #        simul_obs = model.run(s, parallelization)
    #    return simul_obs
    #M 4 parameters
    #params = {'R': (0.04) ** 2, 'n_pc': 50,
    #        'maxiter': 10, 'restol': 0.01,
    #        'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N,
    #        'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
    #        'kernel': kernel, 'post_cov': "diag",
    #        'precond': True, 'LM': True,
    #        'parallel': True, 'linesearch': True,
    #        'forward_model_verbose': False, 'verbose': False,
    #        'iter_save': True}

    # params['objeval'] = False, if true, it will compute accurate objective function
    # params['ncores'] = 36, with parallell True, it will determine maximum physcial core unless specified



    # initialize
    #prob = PCGA(forward_model, s_init, pts, params, s_true, obs)
    # prob = PCGA(forward_model, s_init, pts, params, s_true, obs, X = X) #if you want to add your own drift X

    # run inversion
    #s_hat, simul_obs, post_diagv, iter_best = prob.Run()

    #post_diagv[post_diagv < 0.] = 0.  # just in case
    #post_std = np.sqrt(post_diagv)


    ### BEGINNING OF PLOTTING #############
    #fig = plt.figure()
    #plt.plot(x,s_hat,'k-',label='estimated')
    #plt.plot(x,s_hat + 2.*post_std,'k--',label='95%')
    #plt.plot(x,s_hat - 2.*post_std,'k--',label='')
    #plt.plot(x,s_true,'r-',label='true')
    #plt.title('pumping history')
    #plt.xlabel('time (min)')
    #plt.ylabel(r's ($m^3$/min)')
    #plt.legend()
    #fig.savefig('best.png')
    #plt.close(fig)

    #nobs = prob.obs.shape[0]
    #fig = plt.figure()
    #plt.title('obs. vs simul.')
    #plt.plot(prob.obs, simul_obs, '.')
    #plt.xlabel('observation')
    #plt.ylabel('simulation')
    #minobs = np.vstack((prob.obs, simul_obs)).min(0)
    #maxobs = np.vstack((prob.obs, simul_obs)).max(0)
    #plt.plot(np.linspace(minobs, maxobs, 20), np.linspace(minobs, maxobs, 20), 'k-')
    #plt.axis('equal')
    #axes = plt.gca()
    #axes.set_xlim([math.floor(minobs), math.ceil(maxobs)])
    #axes.set_ylim([math.floor(minobs), math.ceil(maxobs)])
    #fig.savefig('obs.png')
    # plt.show()
    #plt.close(fig)

    #fig = plt.figure()
    #plt.semilogy(np.linspace(1,len(prob.objvals),len(prob.objvals)), prob.objvals, 'r-')
    #plt.xticks(np.linspace(1,len(prob.objvals),len(prob.objvals)))
    #plt.title('obj values over iterations')
    #plt.axis('tight')
    #fig.savefig('obj.png')
    #plt.close(fig)

