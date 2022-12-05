import numpy as np
from glob import glob


class HydroData:
    def __init__(self, exp, amb, model_num):
        filename = exp + "_" + amb + \
            "/output/snr_Ia_" + str(model_num) + ".dat"
        dat = np.loadtxt(filename)

        self.layer = dat[:, 0]
        self.rad = dat[:, 1]
        self.rho = dat[:, 2]
        self.vel = dat[:, 3]
        self.temp = dat[:, 4]


class HydroFullRun:
    def __init__(self, exp, amb, layer_num):
        """Reads values for a layer through each timestep, it's assumed a run has 100 
        profiles created. 

        Args:
            exp (str): base of directory corresponding to explosion model
            amb (str): end of directory corresponding to ambient medium and any extras
            layer_num (int): layer of interest
        """

        infile = np.sort(glob(exp+'_'+amb+'/output/snr_Ia_1*.dat'))
        self.rad = np.zeros(101)
        self.rho = np.zeros(101)
        self.vel = np.zeros(101)
        self.temp = np.zeros(101)

        for i in range(len(infile)):
            dat = np.loadtxt(infile[i])

            self.rad[i] = dat[layer_num - 1, 1]
            self.rho[i] = dat[layer_num - 1, 2]
            self.vel[i] = dat[layer_num - 1, 3]
            self.temp[i] = dat[layer_num - 1, 4]
# using and calling class
# hd = HydroData(exp,amb,model_num)
# hd.rad
#test commit message
