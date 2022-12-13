import numpy as np
import matplotlib.pyplot as plt
from glob import glob


class HydroData:

    cm_to_km = 1e5

    def __init__(self, exp, amb, model_num):
        """Reads the data from the hydro profile

        Args:
            exp (str): base of directory corresponding to explosion model
            amb (str): end of directory corresponding to ambient medium and any extras
            model_num (int): Model number (1000-1100) of interest
        """
        filename = exp + "_" + amb + \
            "/output/snr_Ia_" + str(model_num) + ".dat"
        dat = np.loadtxt(filename)
        self.exp = exp
        self.amb = amb
        self.model_num = model_num
        self.layer = dat[:, 0]
        self.rad = dat[:, 1]
        self.rho = dat[:, 2]
        self.vel = dat[:, 3]
        self.temp = dat[:, 4]

    def rho_vel_T(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
        ax1.loglog(self.rad, self.rho)
        ax2.semilogy(self.rad, self.vel/self.cm_to_km)
        ax3.loglog(self.rad, self.temp)

        ax1.set_ylabel(r'\rho [g/cm$^3$]')
        ax2.set_ylabel('Velocity [km/s]')
        ax3.set_ylabel('Temperature [K]')
        ax3.set_xlabel('Radius [cm]')
        fig.savefig(f'{self.exp}_{self.amb}_{str(self.model_num)}_rvt.png')


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
