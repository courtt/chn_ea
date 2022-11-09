import numpy as np
import os

from nei import read_columns

class HydroData:
    def __init__(self,exp,amb,model_num):
        path = os.getcwd()
        filename = path + exp + '/' + amb + '/snr_Ia_' + model_num +'.dat'
        dat = np.loadtxt(filename)
        # Kept for future readability since I'm new to classes
        self.layer   = dat[:,0]
        self.rad     = dat[:,1]
        self.rho     = dat[:,2]
        self.vel     = dat[:,3]
        self.u_int   = dat[:,4]

# using and calling class
# hd = HydroData(exp,amb,model_num)
# hd.rad 