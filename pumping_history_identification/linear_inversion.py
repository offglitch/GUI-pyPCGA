from main_ui import Ui_MainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget
from pyPCGA import PCGA

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import numpy as np
import math
import drawdown as dd
  

class MainWindow(QMainWindow, Ui_MainWindow):

    def connectSlots(self):

        self.execute_button.clicked.connect(self.execute)

        self.restart_button.clicked.connect(self.restartFunction)

        self.check_button.clicked.connect(self.switchFunction)

        self.s_true_button.clicked.connect(self.openFileNameDialog)

        self.s_init_box.clicked.connect(self.openFileNameDialog)

        self.Obs_button.clicked.connect(self.openFileNameDialog)






    def switchFunction(self):
        passedVals = True
        x0 = int(self.x0_box.toPlainText())
        lx = int(self.lx_box.toPlainText())
        lambdax = int(self.precision_label.toPlainText())
        n_pc = int(self.n_pc_label.toPlainText())
        prior_std = float(self.r_label.toPlainText())
        maxiter = int(self.maxiter_label.toPlainText())
        restol = float(self.restol_label.toPlainText())
        # check if values are in their correct range
        if (x0 < 0):
            passedVals = False
        if (lx < 0):
            passedVals = False
        if (lambdax < 0):
            passedVals = False
        if (n_pc < 0 & n_pc > 200):
            passedVals = False
        if (prior_std < 0):
            passedVals = False
        if (maxiter < 0):
            passedVals = False
        if (restol < 0):
            passedVals = False
        
        
        if(passedVals):

            msg = QtWidgets.QMessageBox()

            msg.setIcon(QtWidgets.QMessageBox.Information)

            msg.setText("Correct!")
            msg.setInformativeText("Your values are within the correct range.")
            msg.setWindowTitle("Congratulations")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            retval = msg.exec_()
            print("value of pressed message box button:", retval)
        else:
            
            msg = QtWidgets.QMessageBox()
            
            msg.setIcon(QtWidgets.QMessageBox.Information)

            msg.setText("Error!")
            msg.setInformativeText("One or more of your values are incorrect. Please check that your values are correct.")
            msg.setWindowTitle("Incorrect Values")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
            retval = msg.exec_()
            print("value of pressed message box button:", retval)
        print("Checked")

    def restartFunction(self):
        global window
        global app

        msg = QtWidgets.QMessageBox()

        msg.setIcon(QtWidgets.QMessageBox.Information)

        msg.setText("Restart")
        msg.setInformativeText("Are you sure you want to restart the program?")
        msg.setWindowTitle("Restart Message")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        # gets the button value pressed
        retval = msg.exec_()
        print("value of pressed message box button:", retval)
        #if button == "ok"
        if (retval == 1024):
            print("Restarted")
            #restart the program
            python = sys.executable
            os.execl(python, python, * sys.argv)
        print("exited restart")


    # function to pop up open file dialog
    def openFileNameDialog(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.s_true_button,"QFileDialog.getOpenFileName()", "","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)

    def execute(self):
        value_2 = int(self.x0_box.toPlainText())
        value = int(self.lx_box.toPlainText())
        lambdax = int(self.precision_label.toPlainText())
        n_pc = int(self.n_pc_label.toPlainText())
        prior_std = float(self.r_label.toPlainText())
        maxiter = int(self.maxiter_label.toPlainText())
        restol = float(self.restol_label.toPlainText())
        # lm = bool(self.lm_check.isChecked())
        # linesearch= bool(self.line_search.isChecked())

        print("values: ", value, value_2)

        self.plot(value, value_2, n_pc, maxiter, restol, prior_std, lambdax)


    def plot(self, lx, x0, n_pc, maxiter, restol, prior_std, lambdax):

        # This is a 1D case, therefore should be used to test the 1D scenario
        
        # M1 parameters are: Lx, Ly, Lz, x0, y0, z0, dx, dy, dz, s_true, s_init

        x0 = x0      # M1: Origin of x dimension 
        Lx = lx   # M1: Total length in the x direction
        dxx = 0.1   # M1: Discretization (cell length) in the x direction, assumes cells of equal size

        # This simulation is 1D, therefore default to y_origin = z_origin = 0, Ly = Lz = 1, dy = dz = 1

        y0 = 0      # M1: Origin of y dimension 
        Ly = 1  # M1: Total length in the y direction
        dyy = 1     # M1: Discretization (cell length) in the y direction, assumes cells of equal size

        z0 = 0      # M1: Origin of y dimension 
        Lz = 1  # M1: Total length in the y direction
        dzz = 1     # M1: Discretization (cell length) in the z direction, assumes cells of equal size

        xmin = np.array([x0]) 
        xmax = np.array([Lx])

        m = int(Lx/dxx + 1) 
        N = np.array([m])


        _translate = QtCore.QCoreApplication.translate

        #self.m_output.setText(_translate("MainWindow", str(m)))
        #self.n_output.setText(_translate("MainWindow", str(N)))


        dx = np.array([dxx])
        x = np.linspace(xmin, xmax, m)
        pts = np.copy(x)
     

        s_true = np.loadtxt('true.txt') # input for file "true.txt" this can be changed to a default directory  
        obs = np.loadtxt('obs.txt')
        #obs = []



        # if(len(obs) != m):

        #    msg = QtWidgets.QMessageBox()

        #    msg.setIcon(QtWidgets.QMessageBox.Information)

        #    msg.setText("Error!")
        #    msg.setInformativeText("Something went wrong. Please check that your values are correct.")
        #    msg.setWindowTitle("Error Message")
        #    msg.setDetailedText("The details are as follows:")
        #    msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        #   # msg.buttonClicked.connect(msgbtn)
    
        #    retval = msg.exec_()
        #    print("value of pressed message box button:", retval)
    

        # print("the values in obs are not correct")
        # s_init, three options (drop down menu) 
        # option 1: user inputs a constant which gets assigned to variable s_constant 

        s_constant = 1          # M1: User selects constant checkbox from drop down, and inputs number in box 
        s_init = s_constant * np.ones((m, 1))

        # option 2: s_init automatically calculated using s_true, if s_true provided
        # # M1: User selects Auto checkbox from drop down, and check is run to see if s_true was provided 
        print(m)
        s_init = np.mean(s_true) * np.ones((m, 1)) #M1 file input or constant input
        # s_init = np.copy(s_true) # you can try with s_true!
     
        prior_std = prior_std #Module 4 (R) 
        lambdax = lambdax
        prior_cov_scale = np.array([lambdax]) #M4 lambdas, lx, ly, lz

        def kernel(r): return (prior_std ** 2) * np.exp(-r)  # M4Kernel use switch function

   
        def forward_model(s, parallelization, ncores=None):
            params = {}
            model = dd.Model(params)
    
            if parallelization:
              simul_obs = model.run(s, parallelization, ncores)
            else:
               simul_obs = model.run(s, parallelization)
            return simul_obs

        params = {'R': (prior_std) ** 2, 'n_pc': n_pc,
             'maxiter': maxiter, 'restol': restol,
             'matvec': 'FFT', 'xmin': xmin, 'xmax': xmax, 'N': N,
             'prior_std': prior_std, 'prior_cov_scale': prior_cov_scale,
             'kernel': kernel, 'post_cov': "diag",
             'precond': True, 'LM': True,
             'parallel': True, 'linesearch': True,
             'forward_model_verbose': False, 'verbose': False,
             'iter_save': True}

        #initialize
        prob = PCGA(forward_model, s_init, pts, params, s_true, obs)

        #run inversion
        s_hat, simul_obs, post_diagv, iter_best = prob.Run()

        post_diagv[post_diagv < 0.] = 0.  
        post_std = np.sqrt(post_diagv)
        ### PLOTTING FOR 1D MODULE 1 #############
        
        # fig = self.fig.add_subplot(111)
        fig = self.axs[0]
        fig.plot(x, s_init,'k-',label='initial')
        fig.plot(x, s_true,'r-',label='true')


        fig.set_title('Pumping history')
        fig.set_xlabel('Time (min)')
        fig.set_ylabel(r'Q ($m^3$/min)')
        fig.legend()
        
        ### PLOTTING FOR 1D MODULE 2, 3 & 4 #############

        # fig2 = self.fig.add_subplot(221)
        fig2 = self.axs[1]
        fig2.plot(x,s_hat,'k-',label='estimated')
        fig2.plot(x,s_hat + 2.*post_std,'k--',label='95%')
        fig2.plot(x,s_hat - 2.*post_std,'k--',label='')
        fig2.plot(x,s_true,'r-',label='true')


        fig2.set_title('Pumping history')
        fig2.set_xlabel('Time (min)')
        fig2.set_ylabel(r'Q ($m^3$/min)')
        fig2.legend()
        self.plotWidget.draw()

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow

app = QApplication(sys.argv)
window = QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(window)

window.show()
sys.exit(app.exec_())