import datetime as dt
import os
import sys
from multiprocessing import Pool
import numpy as np
import shutil as sht
from scipy.io import loadmat, savemat
import pandas as pd
import subprocess

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 14:05:38 2018

@author: akokkinaki

Model definition file for PCGA and COMSPIM 

three operations
1. Write input to run COMPSIM; 
    -text file with permeabilities (cycle through z first and then through z, save as *.txt
    Replace the input file line of the dat file (line that follows inputm)
    Also needed: 
    - file for pc-s parameters. actual values don't matter since these will be run as non-multiphase simulations. Just have this in the main Csim directory does not need to be generated
    - Porosities. also not changing, same as above.
2. run simul

    Compsim is not a parallel program
    executable name is xmpMay20e input output &>screenoutput &
    
3. read input

    functions to read *.hd file read_HD

"""

class Model:
    def __init__(self,params = None):
        self.idx = 0
        self.homedir = os.path.abspath('./')
        self.inputdir = os.path.abspath(os.path.join(self.homedir,"./input_files"))
        self.deletedir = True
        self.outputdir = None
        self.parallel = False
        self.record_cobs = False

        from psutil import cpu_count  # physcial cpu counts
        self.ncores = cpu_count(logical=False)

        if params is not None: 
            if 'deletedir' in params:
                self.deletedir = params['deletedir']
            if 'homedir' in params:
                self.homedir = params['homedir']
                self.inputdir = os.path.abspath(os.path.join(self.homedir,"./input_files"))
            if 'inputdir' in params:
                self.inputdir = params['inputdir']
            if 'ncores' in params:
                self.ncores = params['ncores']
            if 'outputdir' in params:
                # note that outputdir is not used for now; pyPCGA forces outputdir in ./simul/simul0000
                self.outputdir = params['outputdir']
            if 'parallel' in params:
                self.parallel = params['parallel']
            
            
            
            # Need to adjust for COMPSIM - these are for TOUGH2
            #required_params = ("nx","dx","xlocs","ylocs","zlocs","obs_type","t_obs_interval", \
            #"max_timesteps","tstop","const_timestep","max_timestep","absolute_error","relative_error","print_interval", \
            #"relative_error","print_interval","timestep_reduction","gravity",'default_incons', \
            #"multi_params","solver_params","output_times_params")
            
            required_params = ("nx","dx","xlocs","ylocs","obs_type")

            if not params.keys() != required_params:
                raise ValueError("You need to provide all the required csim parameters")
            
            self.__dict__.update((k, v) for k, v in params.items() if k in required_params)
            
    def create_dir(self,idx=None):
        
        mydirbase = "./simul/simul"
        if idx is None:
            idx = self.idx
        
        mydir = mydirbase + "{0:04d}".format(idx)
        mydir = os.path.abspath(os.path.join(self.homedir, mydir))
        
        if not os.path.exists(mydir):
            os.makedirs(mydir)
        
        for filename in os.listdir(self.inputdir):
            sht.copy2(os.path.join(self.inputdir,filename),mydir)
        
        return mydir

    def cleanup(self,outputdir=None):
        """
        Removes outputdir if specified. Otherwise removes all output files
        in the current working directory.
        """
        #import shutil
        import glob
        log = "dummy.log"
        if os.path.exists(log):
            os.remove(log)
        if outputdir is not None and outputdir != os.getcwd():
            if os.path.exists(outputdir):
                sht.rmtree(outputdir)
        else:
            filelist = glob.glob("*.out")
            filelist += glob.glob("*.sim")
            
            for file in filelist:
                os.remove(file)


    def run_model(self,s,idx=0):
        
        # Create directory to run model
        
        case_filename='Case1'
        case_file=case_filename+'.dat'
        sim_dir = self.create_dir(idx)
        os.chdir(sim_dir)
        
        # copy directory with support files and folders
        
        support_files_dir=self.homedir+'/support_files/'  
        delete_csim_files = False
        
        for sp_file in os.listdir(support_files_dir):
            sht.copy2(os.path.join(support_files_dir,sp_file),sim_dir)
            
        os.mkdir('Data')
        os.mkdir('Out')
        sht.move('./'+case_file,'./Data')
        
        
        # s = log k, k=exp(s)
        
        # Create COMPSIM input data file:

        dx = self.dx # grid size
        nx = self.nx # number of grids in each dimension. 
        # m  = nx[0]*nx[1]
        
        # untransform log permeability vector
        k = np.exp(s)
        
        # where are the observations, only used for plotting
        
        xloc = self.xlocs 
        yloc = self.ylocs 
        
        
        # write permeability text file
        # need to give perm.text separate name for each run 
        # perm.txt needs to live in the main Csim folder
        # need to check if can call Csim from one location and have Data and Out in each realz
        
        perm_fn='perm'+str(idx)+'.txt'
             
        line1 = " ".join([str(ki[0]) for ki in k])
        line1 = line1 + " kx" + "\n"
        anisotropy = 1.0
        line2 = " ".join([str(anisotropy * ki[0]) for ki in k])
        line2 = line2 + " kz" + "\n"
        lines=line1,line2
        
        f=open(perm_fn,"w")
        f.writelines(lines)
        f.flush()
        f.close()
        
        # write input file
        # file only written to modify perm field
        # initial file provided 
        
        
        # modification of input file ti map to the right perm file
        self.modify_infile(perm_fn,case_file)
    
    
        # check that other txt files are present in main Csim
        # files needed for heterogeneous simulation: 
        # .dat file, folder Data, folder Out, permeability file, porosity file, pd file
        #self.check_csim
        
        
        from time import time
        stime = time()
        # running the Csim model
        
        csimerr = open("sterror.txt","w")
        csimout = open(os.devnull,"w")
        
        #exe_path = "/home2/hydro_local/" + sim_dir[-9:] + "/xmpMay20e"
        exe_path = os.getcwd() + "/xmpMay20e"   
        
        TestingLocally = False
        
        if TestingLocally:
            delete_csim_files = False
            sht.copy2(self.homedir+'/Case1.hd','./Out')
        else:
            delete_csim_files = True
            subprocess.call([exe_path,case_filename,case_filename], stdout=csimerr, stderr=csimout)
        
        # cleanup of not needed files to save space (vel, flux etc)
        if delete_csim_files:
            self.clean_csim(sim_dir,case_filename)
        

        # read simulation results with obs_type = 'Hw' and/or 'Permeability'
        #from time import time
        # need observation_model function
        
        
        measurements = []
        
        # we currently have only one observation type but use structure to be able to include permeabilities and pressures
        for str_obs_type in self.obs_type:
        
            #out_file = 'Case1.hd'
            
            #if str_obs_type is 'head':
            ext = '.hd'
            out_file = case_filename + ext
            #stime = time()
            measurements.extend(self.observation_model(out_file,str_obs_type,xloc,yloc))
            #print('measurement interpolation: %f sec' % (time() - stime))

        simul_obs = np.array(measurements).reshape(-1)
        
        os.chdir(self.homedir)
        
        if self.deletedir:
            sht.rmtree(sim_dir, ignore_errors=True)
            # self.cleanup(sim_dir)

        return simul_obs
    
    def modify_infile(self, perm_fn, case_file):
        ''' modify input file for Csim
        modify line that follows inputm line and add the correct *.txt file with permeability field 
        In AGU test case permeability file identifier is on line 32
        For more generic infile modification, the line to be modified is the one that follows "1 stis" line
        
        filename is the csim file to be modified
        
        perm_fn is the txt file holding the permeability values, should be formatted as s1 s2 s3 ... sn k in one line
        '''
        f = open('./Data/'+case_file, "r")
        contents = f.readlines()
        f.close()
        contents.insert(23, perm_fn+'\n')
        contents.pop(22)
        os.remove('./Data/'+case_file)
        
        f = open('./Data/'+case_file, "w")
        contents = "".join(contents)
        f.write(contents)
        f.close()
        
        return
     
    def observation_model(self,FileToRead,obs_type,xlocs,ylocs,outtime=3):
        ''' observation_model
        '''
        
        
        if obs_type != 'head' and  obs_type != 'permeability':
            raise ValueError('obs_type should be either Head of Permeability')
        #Permeability calculated based on saturation if we don't want to run the multiphase model
        
        csim_results = self.read_csim(FileToRead)
        
        csim_keys = list(csim_results.keys())
        csim_key = csim_keys[0]
        csim_obs = csim_results[csim_key].reshape(80,40,4)
        obs = []

        # if outtime does not exist replace default by last time
        #outtime = csim_obs[3]ndim
            
        for x in xlocs:
            for y in ylocs:
               # minus 1 to take care of indexing in python starting at 0
               obs.append(csim_obs[x-1,y-1,outtime])

        np.savetxt('obs.txt',obs,delimiter='')

        return np.array(obs)
    
    def read_csim(self, filename, var='var'):
        ''' read *.hd or *.sat file using pandas, output variable hydraulic head
        '''
        #global csim, var_name, file_name
        print(os.getcwd())
        os.chdir('./Out')
        file_name = filename
        csim = {}
        #read .hd file
        if '.hd' in filename:
            namesList = ['x','z','hw','pw','ho','po','hg','pg']
            data = pd.read_csv(filename, header = None, skiprows =6, error_bad_lines = False, delim_whitespace=True, names = namesList)
            data_drop = np.r_[3200:3203,6403:6406,9606:9609]
            data=data.drop(data.index[data_drop])
            
            #set default variable
            if var is 'var':
                var = 'hw'
            
            #read .sat file
        elif '.sat' in filename:
            namesList = ['x','z','lhw','lhst','sw','sgf','sgt','sof','sot']
            data = pd.read_csv(filename, header = None, skiprows =2, error_bad_lines = False, delim_whitespace=True, names = namesList)
            data_drop = np.r_[3200:3202,6402:6404,9604:9606]
            data=data.drop(data.index[data_drop])
        
            if var is 'var':
                var = 'sof'
            
        else:
            print('File type not supported')
            #call variables
        var_name = var
        time1 = slice(0,3200)
        time2 = slice(3200,6400)
        time3 = slice(6400,9600)
        time4 = slice(9600,12800)
        
        data_stack = np.dstack((data[var].iloc[time1].values.reshape(3200,1).astype(float),data[var].iloc[time2].values.reshape(3200,1).astype(float),data[var].iloc[time3].values.reshape(3200,1).astype(float),data[var].iloc[time4].values.reshape(3200,1).astype(float)))
        csim = {var:data_stack}

        return csim
    

    def plot_csim(self, time = 0, x = 80, y = 40, figname='Hd_t3_NAPLf.png'): #default call should be plot_csim.
    
        #define plot data
        csim_keys = list(csim.keys())
        csim_key = csim_keys[0]
        csim_plt = csim[csim_key][:,:,time].reshape(x,y)
        
        #plot properties
        
        fig, ax = plt.subplots(1,1)
        fig.suptitle('{}: {}, timestep = {}'.format(file_name,csim_key,time), fontsize = 12)
        cax = ax.imshow(csim_plt.T, extent=[0,5,0,2.5], origin='upper', cmap='tab20c')
        #ax.imshow(csim_plt.T, extent=[2.5,0,0,5], origin='lower')
        
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        cbar = fig.colorbar(cax, orientation='vertical')    
        
        plt.show()
        
        if figname is not None:
            fig.savefig(figname+'.png', dpi = 600)
        
        
        plt.close()
        
    def check_files(self,filename):
        
        # model run when in a simul directory
        # simul directory is like a Csim directory, and should have
        # copy entire directory with mph, xmp, ./Data, ./Out, por, pds, perm file
        
        # the folder also has the base case run input file *.dat
        
        shutil.copytree("/home2/hydro_local/Csim","./Csim")
        
  
    
    def clean_csim(self,sim_dir,sim_name):
        
        # remove filename.bl*, filename.cnv, filename.con, filename.flx3, filename.ifa, filename.rst, filename.vel
        # execute after each forward model run to save space
        # CAUTION will not allow for debugging
        os.chdir('./Out')
        
        
        output=['.bl1','.bl2','.bl3','.bl4','.blw','.blo','.blg','.cnv','.con','.flx3','.rst','.vel']
        
        for s in output:
        
            os.remove(sim_name+s)
        
        os.chdir('../') 
        
    def run(self,s,par,ncores=None):
        if ncores is None:
            ncores = self.ncores

        method_args = range(s.shape[1])
        args_map = [(s[:, arg:arg + 1], arg) for arg in method_args]

        if par:
            pool = Pool(processes=ncores)
            simul_obs = pool.map(self, args_map)
        else:
            simul_obs =[]
            for item in args_map:
                simul_obs.append(self(item))

        return np.array(simul_obs).T
    
    def __call__(self,args):
            return self.run_model(args[0],args[1])

if __name__ == '__main__':
    import csim as csim
    import numpy as np
    from time import time

    s = np.loadtxt('s_true.txt')
    s = s.reshape(-1, 1)
    nx = [40, 80]
    dx = [0.0625, 0.0625]
    
    # monitoring indices 
    xlocs = np.arange(15,70,5)
    ylocs = np.arange(5,40,5)

    params = {'nx':nx,'dx':dx, 'deletedir':False, 'xlocs': xlocs, 'ylocs':ylocs, \
    'obs_type':['head']}

    mymodel = csim.Model(params)

    par = True   #parallelization false
    
    if par:
        
        ncores = 25
        nrelzs = 50
    
        print('(2) parallel run with ncores = %d' % ncores)
        par = True # parallelization false
        srelz = np.zeros((np.size(s,0),nrelzs),'d')
        for i in range(nrelzs):
            srelz[:,i:i+1] = s + 0.1*np.random.randn(np.size(s,0),1)
        
        stime = time()
        
        simul_obs_all = mymodel.run(srelz,par,ncores = ncores)
    
        print('simulation run: %f sec' % (time() - stime))
        print(simul_obs_all.shape)
        print(simul_obs_all)
        
    else:

        print('(1) single run')
    
        from time import time
        stime = time()
        simul_obs = mymodel.run(s,par)
        print('simulation run: %f sec' % (time() - stime))
    
        obs_true = np.loadtxt('obs_true.txt')
        print(np.linalg.norm(obs_true.reshape(-1) - simul_obs.reshape(-1)))
        #import sys
        #sys.exit(0)

    #obs = np.copy(simul_obs)
    #nobs = obs.shape[0]
    #obs[:nobs/2] = simul_obs[:nobs/2] + 10000.*np.random.randn(nobs/2,1)
    #obs[nobs/2:] = simul_obs[nobs/2:] + 0.5*np.random.randn(nobs/2,1)
    
    #np.savetxt('obs.txt',obs)
    #np.savetxt('obs_pres.txt',obs[:nobs/2])
    #np.savetxt('obs_temp.txt',obs[nobs/2:])

    
        
    
        
        



