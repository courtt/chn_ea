import numpy as np
import os

from nei import read_columns

class HydroData:
    names = ['layer','rad','rho','vel','u_int']

    def __init__(self,exp,amb,model_num):
        for name in self.names:
            self.__dict__[name] = None

        self.read_columns(exp,amb,model_num)
        

    def read_columns(self,exp,amb,model_num): 
        path = os.cwd()
        filename = path + exp + '/' + amb + '/snr_Ia_' + model_num +'.dat'
        dat = np.loadtxt(filename)
        for i, name in enumerate(self.names):
            self.__dict__[name] = dat[:,i]  
        # Kept for future readability since I'm new to classes
        # layer   = dat[:,0]
        # rad     = dat[:,1]
        # rho     = dat[:,2]
        # vel     = dat[:,3]
        # u_int   = dat[:,4]

        # self.layer  = layer
        # self.rad    = rad
        # self.rho    = rho
        # self.vel    = vel
        # self.u_int  = u_int

# using and calling class
# hd = HydroData(exp,amb,model_num)
# hd.rad 