import numpy as np
import os

from nei import read_columns

class HydroData:
    def __init__(self,exp,amb,model_num):
        self.layer  = None
        self.rad    = None
        self.rho    = None
        self.vel    = None
        self.u_int  = None
        self.read_columns(exp,amb,model_num)
        

    def read_columns(self,exp,amb,model_num): 
        path = os.cwd()
        filename = path + exp + '/' + amb + '/snr_Ia_' + model_num +'.dat'
        dat = np.loadtxt(filename)
        layer   = dat[:,0]
        rad     = dat[:,1]
        rho     = dat[:,2]
        vel     = dat[:,3]
        u_int   = dat[:,4]

        self.layer  = layer
        self.rad    = rad
        self.rho    = rho
        self.vel    = vel
        self.u_int  = u_int


# hd = HydroData(exp,amb,model_num)
