import numpy as np
import os
from glob import glob


class HydroData:
    def __init__(self, exp, amb, model_num):
        path = os.getcwd()
        filename = path + exp + "/" + amb + "/snr_Ia_" + model_num + ".dat"
        dat = np.loadtxt(filename)

        self.layer = dat[:, 0]
        self.rad = dat[:, 1]
        self.rho = dat[:, 2]
        self.vel = dat[:, 3]
        self.temp = dat[:, 4]


class HydroFullRun:
    def __init__(self, exp, amb, layer_num):
        """Reads values for a layer through each timestep 

        Args:
            exp (str): base of directory corresponding to explosion model
            amb (str): end of directory corresponding to ambient medium and any extras
            layer_num (int): layer of interest
        """
        infile = np.sort(glob('/Users/travis/pn_spectra/' +
                              exp+'_'+amb+'/output/snr_Ia_prof_a_1*.dat'))
        self.rad = np.zeros(100)
        self.rho = np.zeros(100)
        self.vel = np.zeros(100)
        self.temp = np.zeros(100)

        for i in range(len(infile)):
            dat = np.loadtxt(infile[i])

            self.layer = dat[:, 0]

            self.rad = dat[layer_num - 1, 1]
            self.rho = dat[layer_num - 1, 2]
            self.vel = dat[layer_num - 1, 3]
            self.temp = dat[layer_num - 1, 4]
# using and calling class
# hd = HydroData(exp,amb,model_num)
# hd.rad
