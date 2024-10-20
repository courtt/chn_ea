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

    def rho_vel_T(self,fig=None):
        """Plots hydro values as function of radius

        Args:
            exp (str): base of directory corresponding to explosion model
            amb (str): end of directory corresponding to ambient medium and any extras
            model_num (int): Model number (1000-1100) of interest
        """

        if fig is None:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
        else:
            ax1, ax2, ax3 = fig.axes
            
        ax1.loglog(self.rad, self.rho)
        ax2.semilogx(self.rad, self.vel/self.cm_to_km)
        ax3.loglog(self.rad, self.temp)

        ax1.set_ylabel(r'\rho [g/cm$^3$]')
        ax2.set_ylabel('Velocity [km/s]')
        ax3.set_ylabel('Temperature [K]')
        ax3.set_xlabel('Radius [cm]')
        # fig.savefig(f'{self.exp}_{self.amb}_{str(self.model_num)}_rvt.png')
        return fig


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

class HydroVh1Data: 
    def __init__(self, mdot, vwind, model_num) -> None:
        """Reads the data from the VH1 simulations

        Args:
            mdot (str): mass loss rate for wind model
            vwind (str): velocity of wind in km/s
            model_num (int): Model number (1000-1100) of interest
        """
        filename = "mdot_" + mdot + "_vwind_" + vwind + "_cool_" + str(model_num) + ".dat"
        dat = np.loadtxt(filename)
        self.mdot = mdot
        self.vwind = vwind
        self.model_num = model_num
        self.rad = dat[:, 0]
        self.rho = dat[:, 1]
        self.temp = dat[:, 2]
        self.vel = dat[:, 3]


    def rho_vel_T(self,fig=None):
        """Plots hydro values as function of radius

        Args:
            mdot (str): mass loss rate for wind model
            vwind (str): velocity of wind in km/s
            model_num (int): Model number (1000-1100) of interest
        """

        if fig is None:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
        else:
            ax1, ax2, ax3 = fig.axes
            
        ax1.loglog(self.rad, self.rho)
        ax2.semilogy(self.rad, self.vel)
        ax3.loglog(self.rad, self.temp)

        ax1.set_ylabel(r'$\rho$ [g/cm$^3$]')
        ax2.set_ylabel('Velocity [km/s]')
        ax3.set_ylabel('Pressure [dyne/cm^3]')
        ax3.set_xlabel('Radius [cm]')
        ax3.set_ylim(1e-15,1e-10)
        # fig.savefig(f'{self.exp}_{self.amb}_{str(self.model_num)}_rvt.png')
        return fig


class ShockData: 
    def __init__(self, exp, amb):
        """Reads the shock data from ChN simulations

        Args:
            exp (str): base of directory corresponding to explosion model
            amb (str): end of directory corresponding to ambient medium and any extras
        """
        filename = f'{exp}_{amb}/output/snr_Ia_shock.dat'
        dat = np.loadtxt(filename)

        self.age_yr = dat[:,0]
        self.rad_FS = dat[:,1]
        self.rad_RS = dat[:,2]
        self.rad_CD = dat[:,3]
        self.vel_FS = dat[:,4]
        self.vel_RS = dat[:,5]

    def shock_plot(self,fig=None):
        if fig is None:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10))
        else:
            ax1, ax2 = fig.axes


        fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
        ax1.plot(self.age_yr, self.rad_FS, '-',label='FS')
        ax1.plot(self.age_yr, self.rad_RS, '--',label='RS')
        ax1.plot(self.age_yr, self.rad_CD, ':',label='CD')
        ax1.set_xlabel('Age [yr]')
        ax1.set_ylabel('Radius [cm]')

        ax2.plot(self.age_yr, self.vel_FS, '-',label='FS')
        ax2.plot(self.age_yr, self.vel_RS, '--',label='RS')
        ax2.set_xlabel('Age [yr]')
        ax2.set_ylabel('Velocity [km/s]')
        ax1.legend()
        ax2.legend()
        return fig
