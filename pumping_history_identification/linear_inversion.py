﻿import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget
from pyPCGA import PCGA

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import numpy as np
import math
import drawdown as dd



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("pyPCGA")
        MainWindow.resize(1440, 855)
        #----------------------------------------------
        # Setting the frame
        #----------------------------------------------
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(730, 20, 701, 511))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.frame.setFont(font)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(730, 545, 701, 91))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.export_settings = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.export_settings.setFont(font)
        self.export_settings.setObjectName("export_settings")
        self.gridLayout_6.addWidget(self.export_settings, 0, 1, 1, 1)
        self.import_settings = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.import_settings.setFont(font)
        self.import_settings.setObjectName("import_settings")
        self.gridLayout_6.addWidget(self.import_settings, 0, 2, 1, 1)
        self.execute_button = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.execute_button.setFont(font)
        self.execute_button.setObjectName("execute_button")
        self.gridLayout_6.addWidget(self.execute_button, 0, 0, 1, 1)
        #----------------------------------------------
        # added an execute button
        #----------------------------------------------
        self.execute_button.clicked.connect(self.execute)
        #----------------------------------------------
        # restart button
        #----------------------------------------------
        self.restart_button = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.restart_button.setFont(font)
        self.restart_button.setObjectName("restart_button")
        self.gridLayout_6.addWidget(self.restart_button, 1, 1, 1, 1)
        self.restart_button.clicked.connect(self.restartFunction)
        #----------------------------------------------
        # check values button
        #----------------------------------------------
        self.check_button = QtWidgets.QPushButton(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.check_button.setFont(font)
        self.check_button.setObjectName("check_button")
        self.gridLayout_6.addWidget(self.check_button, 1, 0, 1, 1)
        self.check_button.clicked.connect(self.switchFunction)
        #----------------------------------------------
        # Setting object names and sizing for main frame labels
        #----------------------------------------------
        self.fname_label = QtWidgets.QLabel(self.centralwidget)
        self.fname_label.setGeometry(QtCore.QRect(11, 60, 400, 16))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.fname_label.setFont(font)
        self.fname_label.setObjectName("fname_label")
        self.dimension_label = QtWidgets.QLabel(self.centralwidget)
        self.dimension_label.setGeometry(QtCore.QRect(569, 60, 72, 16))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.dimension_label.setFont(font)
        self.dimension_label.setObjectName("dimension_label")
        self.dim_box = QtWidgets.QComboBox(self.centralwidget)
        dimension_choices = ['1D', '2D', '3D']
        self.dim_box.addItems(dimension_choices)
        self.dim_box.setCurrentIndex(2)
        self.dim_box.setObjectName('3D')
        self.dim_box.setGeometry(QtCore.QRect(651, 60, 69, 24))
        self.dim_box.currentTextChanged.connect(self.dimension_changed)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.dim_box.setFont(font)
        self.dim_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly|QtCore.Qt.ImhPreferNumbers)
        self.dim_box.setObjectName("dim_box")
        self.module1_label = QtWidgets.QLabel(self.centralwidget)
        self.module1_label.setGeometry(QtCore.QRect(11, 94, 691, 16))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.module1_label.setFont(font)
        self.module1_label.setAutoFillBackground(False)
        self.module1_label.setObjectName("module1_label")
        #----------------------------------------------
        # Setting Module 1's frame and grid layout
        #----------------------------------------------
        self.Module1Frame = QtWidgets.QFrame(self.centralwidget)
        self.Module1Frame.setGeometry(QtCore.QRect(11, 114, 691, 181))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.Module1Frame.setFont(font)
        self.Module1Frame.setAutoFillBackground(True)
        self.Module1Frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Module1Frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Module1Frame.setObjectName("Module1Frame")
        self.gridLayout = QtWidgets.QGridLayout(self.Module1Frame)
        self.gridLayout.setObjectName("gridLayout")
        #----------------------------------------------
        # Module 1 labels and boxes start here
        #----------------------------------------------
        self.x0_label = QtWidgets.QLabel(self.Module1Frame)
        self.x0_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.x0_label.setFont(font)
        self.x0_label.setObjectName("x0_label")
        self.gridLayout.addWidget(self.x0_label, 0, 0, 1, 1)
        self.x0_box = QtWidgets.QTextEdit(self.Module1Frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.x0_box.sizePolicy().hasHeightForWidth())
        self.x0_box.setSizePolicy(sizePolicy)
        self.x0_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.x0_box.setFont(font)
        self.x0_box.setAcceptDrops(True)
        self.x0_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.x0_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.x0_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.x0_box.setTabStopWidth(5)
        self.x0_box.setObjectName("x0_box")
        self.gridLayout.addWidget(self.x0_box, 0, 1, 1, 1)
        self.y0_label = QtWidgets.QLabel(self.Module1Frame)
        self.y0_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.y0_label.setFont(font)
        self.y0_label.setObjectName("y0_label")
        self.gridLayout.addWidget(self.y0_label, 0, 2, 1, 1)
        self.y0_box = QtWidgets.QTextEdit(self.Module1Frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.y0_box.sizePolicy().hasHeightForWidth())
        self.y0_box.setSizePolicy(sizePolicy)
        self.y0_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.y0_box.setFont(font)
        self.y0_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.y0_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.y0_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.y0_box.setObjectName("y0_box")
        self.gridLayout.addWidget(self.y0_box, 0, 3, 1, 1)
        self.z0_label = QtWidgets.QLabel(self.Module1Frame)
        self.z0_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.z0_label.setFont(font)
        self.z0_label.setObjectName("z0_label")
        self.gridLayout.addWidget(self.z0_label, 0, 4, 1, 1)
        self.z0_box = QtWidgets.QTextEdit(self.Module1Frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.z0_box.sizePolicy().hasHeightForWidth())
        self.z0_box.setSizePolicy(sizePolicy)
        self.z0_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.z0_box.setFont(font)
        self.z0_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.z0_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.z0_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.z0_box.setObjectName("z0_box")
        self.gridLayout.addWidget(self.z0_box, 0, 5, 1, 1)
        self.lx_label = QtWidgets.QLabel(self.Module1Frame)
        self.lx_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.lx_label.setFont(font)
        self.lx_label.setObjectName("lx_label")
        self.gridLayout.addWidget(self.lx_label, 1, 0, 1, 1)
        self.lx_box = QtWidgets.QTextEdit(self.Module1Frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lx_box.sizePolicy().hasHeightForWidth())
        self.lx_box.setSizePolicy(sizePolicy)
        self.lx_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.lx_box.setFont(font)
        self.lx_box.setAcceptDrops(True)
        self.lx_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lx_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.lx_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.lx_box.setTabStopWidth(5)
        self.lx_box.setObjectName("lx_box")
        self.gridLayout.addWidget(self.lx_box, 1, 1, 1, 1)
        self.ly_label = QtWidgets.QLabel(self.Module1Frame)
        self.ly_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.ly_label.setFont(font)
        self.ly_label.setObjectName("ly_label")
        self.gridLayout.addWidget(self.ly_label, 1, 2, 1, 1)
        self.ly_box = QtWidgets.QTextEdit(self.Module1Frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ly_box.sizePolicy().hasHeightForWidth())
        self.ly_box.setSizePolicy(sizePolicy)
        self.ly_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.ly_box.setFont(font)
        self.ly_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.ly_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.ly_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.ly_box.setObjectName("ly_box")
        self.gridLayout.addWidget(self.ly_box, 1, 3, 1, 1)
        self.lz_label = QtWidgets.QLabel(self.Module1Frame)
        self.lz_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.lz_label.setFont(font)
        self.lz_label.setObjectName("lz_label")
        self.gridLayout.addWidget(self.lz_label, 1, 4, 1, 1)
        self.lz_box = QtWidgets.QTextEdit(self.Module1Frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lz_box.sizePolicy().hasHeightForWidth())
        self.lz_box.setSizePolicy(sizePolicy)
        self.lz_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.lz_box.setFont(font)
        self.lz_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.lz_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.lz_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.lz_box.setObjectName("lz_box")
        self.gridLayout.addWidget(self.lz_box, 1, 5, 1, 1)
        self.dxx_label = QtWidgets.QLabel(self.Module1Frame)
        self.dxx_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.dxx_label.setFont(font)
        self.dxx_label.setObjectName("dxx_label")
        self.gridLayout.addWidget(self.dxx_label, 2, 0, 1, 1)
        self.dxx_box = QtWidgets.QTextEdit(self.Module1Frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dxx_box.sizePolicy().hasHeightForWidth())
        self.dxx_box.setSizePolicy(sizePolicy)
        self.dxx_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.dxx_box.setFont(font)
        self.dxx_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.dxx_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.dxx_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.dxx_box.setObjectName("dxx_box")
        self.gridLayout.addWidget(self.dxx_box, 2, 1, 1, 1)
        self.dyy_label = QtWidgets.QLabel(self.Module1Frame)
        self.dyy_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.dyy_label.setFont(font)
        self.dyy_label.setObjectName("dyy_label")
        self.gridLayout.addWidget(self.dyy_label, 2, 2, 1, 1)
        self.dyy_box = QtWidgets.QTextEdit(self.Module1Frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dyy_box.sizePolicy().hasHeightForWidth())
        self.dyy_box.setSizePolicy(sizePolicy)
        self.dyy_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.dyy_box.setFont(font)
        self.dyy_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.dyy_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.dyy_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.dyy_box.setObjectName("dyy_box")
        self.gridLayout.addWidget(self.dyy_box, 2, 3, 1, 1)
        self.dzz_label = QtWidgets.QLabel(self.Module1Frame)
        self.dzz_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.dzz_label.setFont(font)
        self.dzz_label.setObjectName("dzz_label")
        self.gridLayout.addWidget(self.dzz_label, 2, 4, 1, 1)
        self.dz_box = QtWidgets.QTextEdit(self.Module1Frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dz_box.sizePolicy().hasHeightForWidth())
        self.dz_box.setSizePolicy(sizePolicy)
        self.dz_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.dz_box.setFont(font)
        self.dz_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.dz_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.dz_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.dz_box.setObjectName("dz_box")
        self.gridLayout.addWidget(self.dz_box, 2, 5, 1, 1)
        self.s_true_label = QtWidgets.QLabel(self.Module1Frame)
        self.s_true_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.s_true_label.setFont(font)
        self.s_true_label.setObjectName("s_true_label")
        self.gridLayout.addWidget(self.s_true_label, 3, 0, 1, 1)
        self.s_true_button = QtWidgets.QToolButton(self.Module1Frame)
        self.s_true_button.setMinimumSize(QtCore.QSize(130, 25))
        self.s_true_button.setMaximumSize(QtCore.QSize(130, 25))
        self.s_true_button.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.s_true_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.s_true_button.setAutoRaise(True)
        self.s_true_button.setObjectName("s_true_button")
        self.gridLayout.addWidget(self.s_true_button, 3, 1, 1, 1)
        self.s_true_button.clicked.connect(self.openFileNameDialog)
        # self.m_output_label = QtWidgets.QLabel(self.Module1Frame)
        # self.m_output_label.setMaximumSize(QtCore.QSize(50, 20))
        # font = QtGui.QFont()
        # font.setFamily("Helvetica")
        # font.setPointSize(15)
        # self.m_output_label.setFont(font)
        # self.m_output_label.setObjectName("m_output_label")
        # self.gridLayout.addWidget(self.m_output_label, 4, 2, 1, 1)
        # self.m_output = QtWidgets.QLabel(self.Module1Frame)
        # self.m_output.setMaximumSize(QtCore.QSize(130, 25))
        # font = QtGui.QFont()
        # font.setFamily("Helvetica")
        # font.setPointSize(15)
        # self.m_output.setFont(font)
        # self.m_output.setText("")
        # self.m_output.setObjectName("m_output")
        # self.gridLayout.addWidget(self.m_output, 4, 3, 1, 1)
        # self.s_init_box = QtWidgets.QTextEdit(self.Module1Frame)
        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        # sizePolicy.setHorizontalStretch(0)
        # sizePolicy.setVerticalStretch(0)
        # sizePolicy.setHeightForWidth(self.s_init_box.sizePolicy().hasHeightForWidth())
        # self.s_init_box.setSizePolicy(sizePolicy)
        # self.s_init_box.setMaximumSize(QtCore.QSize(130, 25))
        # font = QtGui.QFont()
        # font.setFamily("Helvetica")
        # self.s_init_box.setFont(font)
        # self.s_init_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        # self.s_init_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # self.s_init_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # self.s_init_box.setObjectName("s_init_box")
        self.s_init_type_box = QtWidgets.QComboBox(self.Module1Frame)
        self.s_init_type_box.addItems(['file', 'text'])
        self.s_init_type_box.setObjectName("s_init_type_box")
        self.s_init_type_box.currentTextChanged.connect(self.s_init_type_changed)
        self.gridLayout.addWidget(self.s_init_type_box, 4, 1, 1, 1)
        self.s_init_box = QtWidgets.QToolButton(self.Module1Frame)
        self.s_init_box.setMinimumSize(QtCore.QSize(130, 25))
        self.s_init_box.setMaximumSize(QtCore.QSize(130, 25))
        self.s_init_box.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.s_init_box.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.s_init_box.setAutoRaise(True)
        self.s_init_box.setObjectName("s_init_box")
        self.gridLayout.addWidget(self.s_init_box, 4, 2, 1, 1)
        self.s_init_box.clicked.connect(self.openFileNameDialog)
        self.s_init_text_box = QtWidgets.QTextEdit(self.Module1Frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.s_init_text_box.sizePolicy().hasHeightForWidth())
        self.s_init_text_box.setSizePolicy(sizePolicy)
        self.s_init_text_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.s_init_text_box.setFont(font)
        self.s_init_text_box.setAcceptDrops(True)
        self.s_init_text_box.setInputMethodHints(QtCore.Qt.ImhDigitsOnly)
        self.s_init_text_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.s_init_text_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.s_init_text_box.setTabStopWidth(5)
        self.s_init_text_box.setObjectName("s_init_text_box")
        self.s_init_text_box.setHidden(True)
        self.s_init_label = QtWidgets.QLabel(self.Module1Frame)
        self.s_init_label.setMaximumSize(QtCore.QSize(80, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.s_init_label.setFont(font)
        self.s_init_label.setObjectName("s_init_label")
        self.gridLayout.addWidget(self.s_init_label, 4, 0, 1, 1)
        self.fname_output = QtWidgets.QLabel(self.centralwidget)
        self.fname_output.setGeometry(QtCore.QRect(11, 150, 16, 16))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.fname_output.setFont(font)
        self.fname_output.setText("")
        self.fname_output.setObjectName("fname_output")
        #----------------------------------------------
        # Setting Module 2's labels and boxes
        #----------------------------------------------
        self.module2_label = QtWidgets.QLabel(self.centralwidget)
        # Adjusted to fix bleed issue
        self.module2_label.setGeometry(QtCore.QRect(11, 311, 300, 16))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.module2_label.setFont(font)
        self.module2_label.setAutoFillBackground(False)
        self.module2_label.setWordWrap(False)
        self.module2_label.setObjectName("module2_label")
        self.Module1Frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.Module1Frame_2.setGeometry(QtCore.QRect(11, 343, 689, 80))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.Module1Frame_2.setFont(font)
        self.Module1Frame_2.setAutoFillBackground(True)
        self.Module1Frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Module1Frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Module1Frame_2.setObjectName("Module1Frame_2")
        self.forward_model = QtWidgets.QComboBox(self.Module1Frame_2)
        self.forward_model.setGeometry(QtCore.QRect(164, 11, 123, 25))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.forward_model.sizePolicy().hasHeightForWidth())
        self.forward_model.setSizePolicy(sizePolicy)
        self.forward_model.setMaximumSize(QtCore.QSize(200, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.forward_model.setFont(font)
        self.forward_model.setObjectName("forward_model")
        self.forward_model.addItem("")
        self.forward_model.addItem("")
        self.forward_model.addItem("")
        self.source_label = QtWidgets.QLabel(self.Module1Frame_2)
        self.source_label.setGeometry(QtCore.QRect(13, 42, 50, 16))
        self.source_label.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.source_label.setFont(font)
        self.source_label.setObjectName("source_label")
        self.forward_model_label = QtWidgets.QLabel(self.Module1Frame_2)
        self.forward_model_label.setGeometry(QtCore.QRect(13, 13, 104, 16))
        self.forward_model_label.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.forward_model_label.setFont(font)
        self.forward_model_label.setObjectName("forward_model_label")
        self.log_check = QtWidgets.QCheckBox(self.Module1Frame_2)
        self.log_check.setGeometry(QtCore.QRect(605, 14, 19, 18))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.log_check.setFont(font)
        self.log_check.setText("")
        self.log_check.setObjectName("log_check")
        self.log_label = QtWidgets.QLabel(self.Module1Frame_2)
        self.log_label.setGeometry(QtCore.QRect(530, 13, 25, 16))
        self.log_label.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.log_label.setFont(font)
        self.log_label.setObjectName("log_label")
        self.source_button = QtWidgets.QToolButton(self.Module1Frame_2)
        self.source_button.setGeometry(QtCore.QRect(90, 42, 200, 25))
        self.source_button.setMinimumSize(QtCore.QSize(130, 25))
        self.source_button.setMaximumSize(QtCore.QSize(130, 25))
        self.source_button.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.source_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.source_button.setAutoRaise(True)
        self.source_button.setObjectName("source_button")

        #----------------------------------------------
        # Setting Module 3's labels and boxes
        #----------------------------------------------
        self.module3_label = QtWidgets.QLabel(self.centralwidget)
        self.module3_label.setGeometry(QtCore.QRect(11, 439, 691, 16))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.module3_label.setFont(font)
        self.module3_label.setAutoFillBackground(False)
        self.module3_label.setObjectName("module3_label")
        self.Module1Frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.Module1Frame_3.setGeometry(QtCore.QRect(11, 471, 691, 86))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.Module1Frame_3.setFont(font)
        self.Module1Frame_3.setAutoFillBackground(True)
        self.Module1Frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Module1Frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Module1Frame_3.setObjectName("Module1Frame_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.Module1Frame_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.n_module3_label = QtWidgets.QLabel(self.Module1Frame_3)
        self.n_module3_label.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.n_module3_label.setFont(font)
        self.n_module3_label.setObjectName("n_module3_label")
        self.gridLayout_3.addWidget(self.n_module3_label, 0, 2)
        self.obs_label = QtWidgets.QLabel(self.Module1Frame_3)
        self.obs_label.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.obs_label.setFont(font)
        self.obs_label.setObjectName("obs_label")
        # Fix Grid positioning for the label and the box
        self.gridLayout_3.addWidget(self.obs_label, 0, 0)
        self.Obs_button = QtWidgets.QToolButton(self.Module1Frame_3)
        self.Obs_button.setMinimumSize(QtCore.QSize(130, 25))
        self.Obs_button.setMaximumSize(QtCore.QSize(130, 25))
        self.Obs_button.setPopupMode(QtWidgets.QToolButton.DelayedPopup)
        self.Obs_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.Obs_button.setAutoRaise(True)
        self.Obs_button.setObjectName("Obs_button")
        self.gridLayout_3.addWidget(self.Obs_button, 0, 1)
        self.Obs_button.clicked.connect(self.openFileNameDialog)
        #----------------------------------------------
        # Setting Module 4's labels and boxes
        #----------------------------------------------
        self.module4_label = QtWidgets.QLabel(self.centralwidget)
        self.module4_label.setGeometry(QtCore.QRect(11, 573, 241, 16))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(16)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.module4_label.setFont(font)
        self.module4_label.setAutoFillBackground(False)
        self.module4_label.setObjectName("module4_label")
        self.Module1Frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.Module1Frame_4.setGeometry(QtCore.QRect(11, 605, 691, 156))
        self.Module1Frame_4.setMaximumSize(QtCore.QSize(700, 180))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.Module1Frame_4.setFont(font)
        self.Module1Frame_4.setAutoFillBackground(True)
        self.Module1Frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Module1Frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Module1Frame_4.setObjectName("Module1Frame_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.Module1Frame_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.x_module4_label = QtWidgets.QLabel(self.Module1Frame_4)
        self.x_module4_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.x_module4_label.setFont(font)
        self.x_module4_label.setObjectName("x_module4_label")
        self.gridLayout_4.addWidget(self.x_module4_label, 0, 0, 1, 1)
        self.x_select = QtWidgets.QComboBox(self.Module1Frame_4)
        self.x_select.setMaximumSize(QtCore.QSize(130, 30))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.x_select.setFont(font)
        self.x_select.setObjectName("x_select")
        self.x_select.addItem("")
        self.x_select.addItem("")
        self.x_select.addItem("")
        self.gridLayout_4.addWidget(self.x_select, 0, 1, 1, 1)
        self.lambda_x_label = QtWidgets.QLabel(self.Module1Frame_4)
        self.lambda_x_label.setMaximumSize(QtCore.QSize(80, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.lambda_x_label.setFont(font)
        self.lambda_x_label.setObjectName("lambda_x_label")
        self.gridLayout_4.addWidget(self.lambda_x_label, 0, 2, 1, 1)
        self.precision_label = QtWidgets.QTextEdit(self.Module1Frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.precision_label.sizePolicy().hasHeightForWidth())
        self.precision_label.setSizePolicy(sizePolicy)
        self.precision_label.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.precision_label.setFont(font)
        self.precision_label.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.precision_label.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.precision_label.setObjectName("precision_label")
        self.gridLayout_4.addWidget(self.precision_label, 0, 3, 1, 1)
        self.kernel_label = QtWidgets.QLabel(self.Module1Frame_4)
        self.kernel_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.kernel_label.setFont(font)
        self.kernel_label.setObjectName("kernel_label")
        self.gridLayout_4.addWidget(self.kernel_label, 0, 4, 1, 1)
        self.kernel_box = QtWidgets.QComboBox(self.Module1Frame_4)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.kernel_box.setFont(font)
        self.kernel_box.setObjectName("kernel_box")
        self.kernel_box.addItem("")
        self.kernel_box.addItem("")
        self.gridLayout_4.addWidget(self.kernel_box, 0, 5, 1, 2)

        self.n_pc_label = QtWidgets.QLabel(self.Module1Frame_4)
        self.n_pc_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.n_pc_label.setFont(font)
        self.n_pc_label.setObjectName("n_pc_label")
        self.gridLayout_4.addWidget(self.n_pc_label, 1, 0, 1, 1)
        self.n_pc_box = QtWidgets.QTextEdit(self.Module1Frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.n_pc_box.sizePolicy().hasHeightForWidth())
        self.n_pc_box.setSizePolicy(sizePolicy)
        self.n_pc_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.n_pc_box.setFont(font)
        self.n_pc_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.n_pc_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.n_pc_box.setObjectName("n_pc_box")
        self.gridLayout_4.addWidget(self.n_pc_box, 1, 1, 1, 1)

        self.matvec_label = QtWidgets.QLabel(self.Module1Frame_4)
        self.matvec_label.setMaximumSize(QtCore.QSize(80, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.matvec_label.setFont(font)
        self.matvec_label.setObjectName("matvec_label")
        self.gridLayout_4.addWidget(self.matvec_label, 1, 2, 1, 1)
        self.matvec_box = QtWidgets.QComboBox(self.Module1Frame_4)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.matvec_box.setFont(font)
        self.matvec_box.setObjectName("matvec_box")
        self.matvec_box.addItem("")
        self.matvec_box.addItem("")
        self.matvec_box.addItem("")
        self.matvec_box.addItem("")
        self.gridLayout_4.addWidget(self.matvec_box, 1, 3, 1, 1)
        self.prior_std_label = QtWidgets.QLabel(self.Module1Frame_4)
        self.prior_std_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.prior_std_label.setFont(font)
        self.prior_std_label.setObjectName("prior_std_label")
        self.gridLayout_4.addWidget(self.prior_std_label, 2, 0, 1, 1)
        self.r_label = QtWidgets.QTextEdit(self.Module1Frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.r_label.sizePolicy().hasHeightForWidth())
        self.r_label.setSizePolicy(sizePolicy)
        self.r_label.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.r_label.setFont(font)
        self.r_label.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.r_label.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.r_label.setObjectName("r_label")
        self.gridLayout_4.addWidget(self.r_label, 2, 1, 1, 1)
        self.maxiter_label = QtWidgets.QLabel(self.Module1Frame_4)
        self.maxiter_label.setMaximumSize(QtCore.QSize(80, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.maxiter_label.setFont(font)
        self.maxiter_label.setObjectName("maxiter_label")
        self.gridLayout_4.addWidget(self.maxiter_label, 2, 2, 1, 1)
        self.maxiter_box= QtWidgets.QTextEdit(self.Module1Frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.maxiter_box.sizePolicy().hasHeightForWidth())
        self.maxiter_box.setSizePolicy(sizePolicy)
        self.maxiter_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.maxiter_box.setFont(font)
        self.maxiter_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.maxiter_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.maxiter_box.setObjectName("maxiter_label")
        self.gridLayout_4.addWidget(self.maxiter_label, 2, 2, 1, 1)
        self.gridLayout_4.addWidget(self.maxiter_box, 2, 3, 1, 1)
        self.restol_label = QtWidgets.QLabel(self.Module1Frame_4)
        self.restol_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.restol_label.setFont(font)
        self.restol_label.setObjectName("restol_label")
        self.gridLayout_4.addWidget(self.restol_label, 2, 4, 1, 1)
        self.restol_box = QtWidgets.QTextEdit(self.Module1Frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.restol_label.sizePolicy().hasHeightForWidth())
        self.restol_box.setSizePolicy(sizePolicy)
        self.restol_box.setMaximumSize(QtCore.QSize(130, 25))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.restol_box.setFont(font)
        self.restol_box.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.restol_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.restol_box.setObjectName("restol_label")
        self.gridLayout_4.addWidget(self.restol_box, 2, 5, 1, 2)

        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.lm_label = QtWidgets.QLabel(self.Module1Frame_4)
        self.lm_label.setMaximumSize(QtCore.QSize(50, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.lm_label.setFont(font)
        self.lm_label.setObjectName("lm_label")
        self.gridLayout_4.addWidget(self.lm_label, 3, 2, 1, 1)
        self.lm_check = QtWidgets.QCheckBox(self.Module1Frame_4)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.lm_check.setFont(font)
        self.lm_check.setText("")
        self.lm_check.setObjectName("lm_check")
        self.gridLayout_4.addWidget(self.lm_check, 3, 3, 1, 1)
        self.linesearch_label = QtWidgets.QLabel(self.Module1Frame_4)
        self.linesearch_label.setMaximumSize(QtCore.QSize(150, 20))
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(15)
        self.linesearch_label.setFont(font)
        self.linesearch_label.setObjectName("linesearch_label")
        self.gridLayout_4.addWidget(self.linesearch_label, 3, 4, 1, 2)
        self.line_search = QtWidgets.QCheckBox(self.Module1Frame_4)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        self.line_search.setFont(font)
        self.line_search.setText("")
        self.line_search.setObjectName("line_search")
        self.gridLayout_4.addWidget(self.line_search, 3, 6, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1440, 22))
        self.menubar.setObjectName("menubar")
        self.menupyPCGA = QtWidgets.QMenu(self.menubar)
        self.menupyPCGA.setObjectName("menupyPCGA")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionDownload = QtWidgets.QAction(MainWindow)
        self.actionDownload.setObjectName("actionDownload")
        self.actionImport = QtWidgets.QAction(MainWindow)
        self.actionImport.setObjectName("actionImport")
        self.menupyPCGA.addAction(self.actionNew)
        self.menupyPCGA.addAction(self.actionSave)
        self.menupyPCGA.addAction(self.actionDownload)
        self.menupyPCGA.addAction(self.actionImport)
        self.menubar.addAction(self.menupyPCGA.menuAction())
        #----------------------------------------------
        # Setting the graphing frames
        #----------------------------------------------
        self.frame = QtWidgets.QWidget(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(760, 30, 651, 511))
        self.frame.setObjectName("frame")
        self.fig, self.axs = plt.subplots(2, constrained_layout=True)
        self.plotWidget = FigureCanvas(self.fig)
        self.plotWidget.setParent(self.frame)
        #----------------------------------------------
        # Calls retranslateUi
        #----------------------------------------------
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.lx_box, self.ly_box)
        MainWindow.setTabOrder(self.ly_box, self.lz_box)
        MainWindow.setTabOrder(self.lz_box, self.dxx_box)
        MainWindow.setTabOrder(self.dxx_box, self.dyy_box)
        MainWindow.setTabOrder(self.dyy_box, self.dz_box)
       # MainWindow.setTabOrder(self.dz_box, self.n_label)
      #  MainWindow.setTabOrder(self.n_label, self.nlocs_label)
        MainWindow.setTabOrder(self.x_select, self.precision_label)
        MainWindow.setTabOrder(self.precision_label, self.kernel_box)
        MainWindow.setTabOrder(self.kernel_box, self.n_pc_label)
        MainWindow.setTabOrder(self.n_pc_label, self.matvec_box)
        MainWindow.setTabOrder(self.matvec_box, self.r_label)
        MainWindow.setTabOrder(self.r_label, self.maxiter_label)
        MainWindow.setTabOrder(self.maxiter_label, self.restol_label)
        MainWindow.setTabOrder(self.restol_label, self.execute_button)
        MainWindow.setTabOrder(self.execute_button, self.export_settings)
        MainWindow.setTabOrder(self.export_settings, self.import_settings)
        self.MainWindow = MainWindow

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "pyPCGA"))
        self.export_settings.setText(_translate("MainWindow", "Export Settings"))
        self.import_settings.setText(_translate("MainWindow", "Import Settings"))
        self.execute_button.setText(_translate("MainWindow", "Execute"))
        self.restart_button.setText(_translate("MainWindow", "Restart"))
        self.check_button.setText(_translate("MainWindow", "Check"))
        #self.progress_bar_label.setText(_translate("MainWindow", "Progress Bar"))
        self.fname_label.setText(_translate("MainWindow", "File Name: Pumping History Identification"))
        self.dimension_label.setText(_translate("MainWindow", "Dimensions:"))
        self.module1_label.setText(_translate("MainWindow", "Module 1: Domain Parameters"))
        self.x0_label.setText(_translate("MainWindow", "x0:"))
        self.x0_box.setPlaceholderText(_translate("MainWindow", "0 ~ any real number"))
        self.y0_label.setText(_translate("MainWindow", "y0:"))
        self.y0_box.setPlaceholderText(_translate("MainWindow", "0 ~ any real number"))
        self.z0_label.setText(_translate("MainWindow", "z0:"))
        self.z0_box.setPlaceholderText(_translate("MainWindow", "0 ~ any real number"))
        self.lx_label.setText(_translate("MainWindow", "Lx:"))
        self.lx_box.setPlaceholderText(_translate("MainWindow", "0 ~ any real number"))
        self.ly_label.setText(_translate("MainWindow", "Ly:"))
        self.ly_box.setPlaceholderText(_translate("MainWindow", "0 ~ any real number"))
        self.lz_label.setText(_translate("MainWindow", "Lz:"))
        self.lz_box.setPlaceholderText(_translate("MainWindow", "0 ~ any real number"))
        self.dxx_label.setText(_translate("MainWindow", "dxx:"))
        self.dxx_box.setPlaceholderText(_translate("MainWindow", "0 - 200"))
        self.dyy_label.setText(_translate("MainWindow", "dyy:"))
        self.dyy_box.setPlaceholderText(_translate("MainWindow", "0 - 200"))
        self.dzz_label.setText(_translate("MainWindow", "dzz:"))
        self.dz_box.setPlaceholderText(_translate("MainWindow", "0 - 200"))
        self.s_true_label.setText(_translate("MainWindow", "s_true:"))
        self.s_true_button.setText(_translate("MainWindow", "Select File"))
        #self.n_module3_label.setText(_translate("MainWindow", "N:"))
        #self.m_output_label.setText(_translate("MainWindow", "M:"))
        self.s_init_box.setText(_translate("MainWindow", "Select File"))
        self.s_init_label.setText(_translate("MainWindow", "s_init type:"))
        self.module2_label.setText(_translate("MainWindow", "Module 2: Forward Model Parameters"))
        self.forward_model.setItemText(0, _translate("MainWindow", "MODFLOW"))
        self.forward_model.setItemText(1, _translate("MainWindow", "Matrix"))
        self.forward_model.setItemText(2, _translate("MainWindow", "Tough"))
        self.source_label.setText(_translate("MainWindow", "source:"))
        self.forward_model_label.setText(_translate("MainWindow", "forward_model:"))
        self.log_label.setText(_translate("MainWindow", "log:"))
        self.source_button.setText(_translate("MainWindow", "Select File"))
        self.module3_label.setText(_translate("MainWindow", "Module 3: Observations"))
        self.n_module3_label.setText(_translate("MainWindow", "n:"))
        self.obs_label.setText(_translate("MainWindow", "Obs:"))
        self.Obs_button.setText(_translate("MainWindow", "Select File"))
        self.module4_label.setText(_translate("MainWindow", "Module 4: Inversion Parameters"))
        self.x_module4_label.setText(_translate("MainWindow", "x:"))
        self.x_select.setItemText(0, _translate("MainWindow", "Unit"))
        self.x_select.setItemText(1, _translate("MainWindow", "Constant"))
        self.x_select.setItemText(2, _translate("MainWindow", "Linear"))
        self.lambda_x_label.setText(_translate("MainWindow", "λx:"))
        self.kernel_label.setText(_translate("MainWindow", "kernel:"))
        self.kernel_box.setItemText(0, _translate("MainWindow", "Gaussian"))
        self.kernel_box.setItemText(1, _translate("MainWindow", "Exponential"))
        self.n_pc_label.setText(_translate("MainWindow", "n pc:"))
        self.matvec_label.setText(_translate("MainWindow", "matvec:"))
        self.matvec_box.setItemText(0, _translate("MainWindow", "FFT"))
        self.matvec_box.setItemText(1, _translate("MainWindow", "Dense"))
        self.matvec_box.setItemText(2, _translate("MainWindow", "Hmatrix"))
        self.matvec_box.setItemText(3, _translate("MainWindow", "FMM"))
        self.prior_std_label.setText(_translate("MainWindow", "prior_std"))
        self.maxiter_label.setText(_translate("MainWindow", "maxiter:"))
        self.restol_label.setText(_translate("MainWindow", "restol:"))
        self.lm_label.setText(_translate("MainWindow", "LM:"))
        self.linesearch_label.setText(_translate("MainWindow", "Linesearch:"))
        self.menupyPCGA.setTitle(_translate("MainWindow", "File"))
        self.actionNew.setText(_translate("MainWindow", "New"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionDownload.setText(_translate("MainWindow", "Download"))
        self.actionImport.setText(_translate("MainWindow", "Import"))

    def switchFunction(self):
        passedVals = True
        x0 = int(self.x0_box.toPlainText())
        lx = int(self.lx_box.toPlainText())
        lambdax = int(self.precision_label.toPlainText())
        n_pc = int(self.n_pc_box.toPlainText())
        prior_std = float(self.r_box.toPlainText())
        maxiter = int(self.maxiter_box.toPlainText())
        restol = float(self.restol_box.toPlainText())
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
        object_name = self.MainWindow.sender().objectName()
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.s_true_button,"QFileDialog.getOpenFileName()", "","All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)
            setattr(self, object_name + '_file', fileName)

    def open_error_dialog(self, message):
        global window
        global app
        msg = QtWidgets.QMessageBox()

        msg.setIcon(QtWidgets.QMessageBox.Critical)

        msg.setText("Error!")
        msg.setInformativeText(message)
        msg.setWindowTitle("Error!")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()

    def validate_number(self, widget, min_value, max_value, datatype):
        message = widget.objectName() + 'value should be '
        value = widget.toPlainText()
        if min_value and max_value:
            message += 'between {0} and {1}'.format(min_value, max_value)
        try:
            value = datatype(value)
            if (min_value and value < min_value) or (max_value and value > max_value):
                self.open_error_dialog(message)
                self.is_valid = False

        except ValueError:
            self.is_valid = False
            self.open_error_dialog(message)
        return value


    def execute(self):
        self.is_valid = True
        value_2 = self.validate_number(self.x0_box, 0, 5000, int)
        value = self.validate_number(self.lx_box, 0, 5000, int)
        lambdax = int(self.precision_label.toPlainText())
        n_pc = int(self.n_pc_box.toPlainText())
        prior_std = float(self.r_label.toPlainText())
        maxiter = int(self.maxiter_box.toPlainText())
        restol = float(self.restol_box.toPlainText())
        y0 = self.validate_number(self.y0_box, 0, 5000, int)
        ly = self.validate_number(self.ly_box, 0, 5000, int)
        dyy = self.validate_number(self.dyy_box, 0, 5000, int)
        z0 = self.validate_number(self.z0_box, 0, 5000, int)
        lz = self.validate_number(self.lz_box, 0, 5000, int)
        dz = self.validate_number(self.dz_box, 0, 5000, int)
        # lm = bool(self.lm_check.isChecked())
        # linesearch= bool(self.line_search.isChecked())
        if not self.is_valid:
            return

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
        # change the scope of s_constant 
        # s_constant = 1 
        # M1: User selects constant checkbox from drop down, and inputs number in box
        if hasattr(self, 's_init_box_file') and self.s_init_box_file:
            s_init = np.loadtxt(self.s_init_box_file)
        elif self.s_init_text_box.toPlainText():
            try:
                values = self.s_init_text_box.toPlainText()
                # multiply the number that was entered (single value only) in the text box and push them into the list instead of splitting by spaces
                # push into s_constant = 1 
                values = int(values)
                print(type(values))
                print(values)
                s_init = values * np.ones((m, 1))
                print(s_init)
            except (TypeError, ValueError):
                self.open_error_dialog('s_init is not a proper value')
                return
        else:
            self.open_error_dialog('s_init value not present')
            return

        if len(s_init) != len(s_true):
            self.open_error_dialog('s_init and s_true should have the same number of values')
            return

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
             'parallel': False, 'linesearch': True,
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

    def set_default_dimensions_val(self, boxes):
        for index, box in enumerate(boxes):
            value = 1
            if index == 0:
                value = 0
            box.setText(str(value))

    def set_read_only_state(self, boxes, read_only=True):
        for box in boxes:
            box.setDisabled(read_only)

    def dimension_changed(self, value):
        y_boxes = [self.y0_box, self.ly_box, self.dyy_box]
        z_boxes = [self.z0_box, self.lz_box, self.dz_box]
        if value == '3D':
            self.set_read_only_state(y_boxes, False)
            self.set_read_only_state(z_boxes, False)
        elif value == '2D':
            self.set_read_only_state(y_boxes, False)
            self.set_read_only_state(z_boxes, True)
            self.set_default_dimensions_val(z_boxes)
        elif value == '1D':
            self.set_read_only_state(y_boxes)
            self.set_read_only_state(z_boxes)
            self.set_default_dimensions_val(y_boxes)
            self.set_default_dimensions_val(z_boxes)

    def s_init_type_changed(self, value):
        if value == 'file':
            self.s_init_box.setHidden(False)
            self.s_init_text_box.setHidden(True)
            self.gridLayout.addWidget(self.s_init_box, 4, 2, 1, 1)
        elif value == 'text':
            self.s_init_box.setHidden(True)
            self.s_init_text_box.setHidden(False)
            self.gridLayout.addWidget(self.s_init_text_box, 4, 2, 1, 1)




import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow

app = QApplication(sys.argv)
window = QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(window)

window.show()
sys.exit(app.exec_())
