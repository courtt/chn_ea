import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from scipy import integrate


class NEIData:
    # names = ['igrid'....]
    def __init__(self, exp, amb, model_num):
        """Reads the data from the NEI profile

        Args:
            exp (str): base of directory corresponding to explosion model
            amb (str): end of directory corresponding to ambient medium and any extras
            model_num (int): Model number (1000-1100) of interest
        """
        path = os.getcwd()
        infile = (
            path + '/' + exp + "_" + amb + "/output/snr_Ia_prof_a_" +
            str(model_num) + ".dat"
        )
        dat = np.loadtxt(infile)

        if len(dat.shape) == 1:
            dat = dat.reshape([1, len(dat)])
        self.exp = exp
        self.amb = amb
        self.model_num = model_num
        self.igrid = dat[:, 0]
        self.age = dat[:, 1]
        self.rad = dat[:, 2]  # cm
        self.mcoord = dat[:, 3]  # Lagrangian mass coordinate
        self.mgrid = dat[:, 4]  # mass of grid
        self.vol = dat[:, 5]
        self.rho = dat[:, 6]
        self.vel = dat[:, 8]
        self.Te = dat[:, 9]
        self.ne = dat[:, 11]
        self.nH = dat[:, 12]
        self.ni = dat[:, 13]
        self.ab = 10 ** (dat[:, 14:33] - 12.0)
        self.jmax = np.argmax(self.ab)
        self.xion = dat[:, 33:330]
        self.Ti = dat[:, 330:349]
        self.ztau = dat[:, 349]

    def plot_diagnostic(self, ion, fig=None):
        """Reads the data from the NEI profile and plots electron temp (Te), density (rho),
        ionizaiton timescale (tau), electron temperature / mean ion temperature, and 
        ionization state (ionization) as a function of Lagrangian mass coordinate

        Args:
            exp (str): base of directory corresponding to explosion model
            amb (str): end of directory corresponding to ambient medium and any extras
            model_num (int): Model number (1000-1100) of interest
            ion (str): Ion of interest for ionization state
        """
        msun = 1.989e33
        if fig is None:
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
                5, 1, sharex=True, figsize=(8, 10))
        else:
            ax1, ax2, ax3, ax4, ax5 = fig.axes
        # This will plot on the same figure after setting it up. 
        n_spec = 19
        ion_name = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S',
                    'Ar', 'Ca', 'Fe', 'Ni', 'Na', 'Al', 'P', 'Ti', 'Cr', 'Mn']
        # index of element, e.g., 4 = 'O', 11 = 'Fe'
        i_spec = ion_name.index(ion)
        nj = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 18,
                       20, 26, 28, 11, 13, 15, 22, 24, 25])

        ind = np.zeros([n_spec])
        ix = 0
        for no in range(n_spec):
            ixn = ix + nj[no]
            ind[no] = ix
            ix = ixn+1
        ind = np.array(ind, dtype='int')

        frac = self.xion[:, ind[i_spec]:ind[i_spec]+nj[i_spec]+1]
        charges = np.arange(nj[i_spec]+1)
        ionization = np.sum(frac*charges, axis=1)

        ax1.loglog(self.mcoord/msun, self.Te,label=f'{self.amb}')
        ax2.loglog(self.mcoord/msun, self.rho)
        ax3.loglog(self.mcoord/msun, self.ztau)
        ax4.loglog(self.mcoord/msun, self.Te/np.mean(self.Ti, axis=1))
        ax5.semilogx(self.mcoord/msun, ionization)

        ax1.set_xscale('log')
        ax1.set_ylabel(r'$\mathrm{T_e [K]}$')
        ax1.set_ylim(1e6, 1e9)
        ax1.set_title(self.exp + '_' + self.amb +
                      '_' + str(self.model_num) + '_' + ion +'\n' + str(self.age[0]) + ' yr')

        ax1.legend(ncol=2, title='rho_amb')
        ax2.set_ylabel(r'$\mathrm{\rho [g/cm^3]}$')

        ax3.set_ylabel(r'$\mathrm{\tau [cm^{-3}s]}$')

        ax4.set_ylabel(r'$\mathrm{T_e/<T_i>}$')

        ax5.set_ylabel(r'$\mathrm{<Z_>}$')
        ax5.set_xlabel(r'$\mathrm{M[M_{\odot}]}$')

        # fig.savefig(self.exp + '_' + self.amb + '_' +
        #             str(self.model_num) + ion + '_diag.pdf')
        return fig

                


    def spectrum_XRISM(self, i_CD):
        """Outputs the spectrum for a given model convolved with XRISM

        Args:
            i_CD (int): Layer at which the CD is placed (N_part_ejecta in src files)

        Returns:
            Ec (float): Centered energy bins
            lum_ej (float): Luminosity of ejecta in ph/s/keV
            lum_amb (float): Luminosity of ambient medium in ph/s/keV
            lum (float): Total luminosity in ph/s/keV

        """
        import sys
        import os
        import numpy as np
        import pyatomdb.atomdb
        import pyatomdb.spectrum as spec
        import astropy.io.fits as fits
        keV 	= 1e3*1.602e-12

        # XRISM
        myrmf = os.path.expandvars('$ATOMDB/resolve_h7ev_2019a.rmf')
        myarf = os.path.expandvars(
            '$ATOMDB/resolve_pnt_spec_noGV_20190611.arf')

        hdul = fits.open(myrmf)

        outfile = self.exp + '_' + self.amb + \
            '/output/sedXRISM_snr_Ia_' + str(self.model_num) + '.dat'
        n_spec = 19
        dE = 0.0005
        Emax    = np.max(hdul[2].data['E_MAX'])          #* spectral bin max [keV]
        Emin    = np.min(hdul[2].data['E_MIN'])           #* spectral bin min [keV]
        NE = 50001
        E = np.linspace(Emin,Emax,NE)

        if not os.path.exists(outfile):
#            replot = 1
#        if not replot:

            n_part = len(self.igrid)
            print('Model:', self.exp+'_'+self.amb, self.model_num)
            print('data dimension:', n_part)

            zatom 	= np.array([1,2,6,7,8,10,12,14,16,18,20,26,28,11,13,15,22,24,25])
            nj		= np.array([1,2,6,7,8,10,12,14,16,18,20,26,28,11,13,15,22,24,25])

            ind = np.zeros([n_spec])
            ix  = 0
            for no in range(n_spec):
                    ixn = ix + nj[no]
                    ind[no] = ix
                    ix = ixn+1
            ind = np.array(ind,dtype='int') 

            nei = spec.NEISession()
            nei.set_response(myrmf,arf=myarf)
            nei.elements = list(zatom) 
            
            #AG89 values
            ab_init   = pyatomdb.atomdb.get_abundance()
            ab_rel    = dict(zip(np.arange(1,31),np.zeros(30)))
            popul     = dict(zip(np.arange(1,31),np.zeros(30)))


            for i in range(n_part):
                xion = self.xion[i]
                ne   = self.ne[i]
                nH   = self.nH[i]
                Te   = self.Te[i]
                vol  = self.vol[i]
                ab   = self.ab[i]
                jmax = np.argmax(ab)
                igrid = self.igrid[i]

                lum_ej = np.zeros(len(E)-1)+1e-50
                lum_am = np.zeros(len(E)-1)+1e-50

                if (Te < 0.000862*keV/1.381e-16):
                    return lum_ej, lum_am


                for j in range(n_spec):
                        ab_rel[zatom[j]] = (ab[j]/ab[jmax])/ab_init[zatom[j]]
                        popul[zatom[j]]  = xion[ind[j]:ind[j]+nj[j]+1]
                nei.abundsetvector = ab_rel


                if igrid < i_CD:	#you can output the DEM profile separately by storing ne*ni*vol against rad
                    lum_ej += ne*nH*ab[jmax]*nei.return_spectrum(Te,1e11,init_pop=popul,freeze_ion_pop=True,teunit='K')*vol #ph/s/bin
                else:
                    lum_am += ne*nH*ab[jmax]*nei.return_spectrum(Te,1e11,init_pop=popul,freeze_ion_pop=True,teunit='K')*vol #ph/s/bin


                sys.stdout.write("\rProgress:%i/%i (particle:%i) (peak Z:%i)"%(i+1,n_part,igrid,zatom[jmax]))
                sys.stdout.flush()


            Ec = 0.5*(E[1:]+E[:-1])
            #lum_ej *= (Ec**2/dE)*1e3*1.602e-12
            #lum_am *= (Ec**2/dE)*1e3*1.602e-12
            lum_ej /= dE  # ph/s/keV
            lum_am /= dE  # ph/s/keV
            lum = lum_ej+lum_am

            np.savetxt(outfile, np.array([Ec, lum_ej, lum_am, lum]).T)
        else:
            print('Outfile already created')
            dat = np.loadtxt(outfile)
            Ec = dat[:, 0]
            lum_ej = dat[:, 1]
            lum_am = dat[:, 2]
            lum = dat[:, 3]

        return Ec, lum_ej, lum_am, lum 

class NEIFullRun:
    def __init__(self, exp, amb, N_part):
        """Reads the files for each outfile in a run

        Args:
            exp (str): base of directory corresponding to explosion model
            amb (str): end of directory corresponding to ambient medium and any extras
            N_part (int): number of Lagrangian mass particles used for NEI
        """
        yr_to_s = 365*24*3600

        infile = np.sort(glob('/Users/travis/pn_spectra/' +
                         exp+'_'+amb+'/output/snr_Ia_prof_a_1*.dat'))
        layer_list = np.arange(1, N_part + 1)
        self.igrid_list = np.zeros((N_part, 101))
        self.age_list = np.zeros((N_part, 101))
        self.rad_list = np.zeros((N_part, 101))   # cm
        # Lagrangian mass coordinate
        self.mcoord_list = np.zeros((N_part, 101))
        self.mgrid_list = np.zeros((N_part, 101))   # mass of grid
        self.vol_list = np.zeros((N_part, 101))
        self.vel_list = np.zeros((N_part, 101))
        self.Te_list = np.zeros((N_part, 101))
        self.ne_list = np.zeros((N_part, 101))
        self.nH_list = np.zeros((N_part, 101))
        self.ni_list = np.zeros((N_part, 101))
        # self.ab_list
        # self.xion_list
        # self.Ti_list
        self.ztau_list = np.zeros((N_part, 101))
        age_arr = np.zeros(101)
        # for i in range(len(infile[:target_ind + 1])):
        for i in range(len(infile)):

            dat = np.loadtxt(infile[i])

            if len(dat.shape) == 1:
                dat = dat.reshape([1, len(dat)])

            shocked_layers = dat[:, 0]

            for j, layer in enumerate(layer_list):
                if layer in shocked_layers:
                    self.igrid_list = np.zeros((N_part, 101))
                    self.age_list = np.zeros((N_part, 101))
                    self.rad_list = np.zeros((N_part, 101))   # cm
                    # Lagrangian mass coordinate
                    self.mcorod_list = np.zeros((N_part, 101))
                    self.mgrid_list = np.zeros((N_part, 101))   # mass of grid
                    self.vol_list = np.zeros((N_part, 101))
                    self.vel_list = np.zeros((N_part, 101))
                    self.Te_list = np.zeros((N_part, 101))
                    self.ne_list = np.zeros((N_part, 101))
                    self.nH_list = np.zeros((N_part, 101))
                    self.ni_list = np.zeros((N_part, 101))
                else:
                    self.igrid_list[j, i] = 0
                    self.age_list[j, i] = 0
                    self.rad_list[j, i] = 0
                    self.mcorod_list[j, i] = 0
                    self.mgrid_list[j, i] = 0
                    self.vol_list[j, i] = 0
                    self.vel_list[j, i] = 0
                    self.Te_list[j, i] = 0
                    self.ne_list[j, i] = 0
                    self.nH_list[j, i] = 0
                    self.ni_list[j, i] = 0
            self.igrid = dat[:, 0]
            self.age = dat[:, 1]
            self.rad = dat[:, 2]  # cm
            self.mcoord = dat[:, 3]  # Lagrangian mass coordinate
            self.mgrid = dat[:, 4]  # mass of grid
            self.vol = dat[:, 5]
            self.vel = dat[:, 8]
            self.Te = dat[:, 9]
            self.ne = dat[:, 11]
            self.nH = dat[:, 12]
            self.ni = dat[:, 13]
            self.ab = 10 ** (dat[:, 14:33] - 12.0)
            self.jmax = np.argmax(self.ab)
            self.xion = dat[:, 33:330]
            self.Ti = dat[:, 330:349]
            self.ztau = dat[:, 349]

            age_arr[i] = dat[0, 1] * yr_to_s

    def ne_t_calc(self, exp, amb, CD_layer):
        """
        Calculates tau for each layer from the RS to CD for all ages

        Parameters
        ----------
        exp : str, base of directory corresponding to explosion model

        amb : str, end of directory corresponding to ambient medium and any extras

        CD_layer : int, layer number of CD in output
        """

        infile = np.sort(glob('/Users/travis/pn_spectra/' +
                         exp+'_'+amb+'/output/snr_Ia_prof_a_1*.dat'))
        yr_to_s = 365*24*3600
        layer_list = np.arange(1, CD_layer+1)
        ne_t_arr = np.zeros((CD_layer, 101))
        ne_arr = np.zeros((CD_layer, 101))
        age_arr = np.zeros(101)
        # for i in range(len(infile[:target_ind + 1])):
        for i in range(len(infile)):

            dat = np.loadtxt(infile[i])

            if len(dat.shape) == 1:
                dat = dat.reshape([1, len(dat)])

            shocked_layers = dat[:, 0]
            ne = dat[:, 11]

            age_arr[i] = dat[0, 1] * yr_to_s

            for j, layer in enumerate(layer_list):
                if layer in shocked_layers:
                    ne_arr[j, i] = ne[layer == shocked_layers]
                else:
                    ne_arr[j, i] = 0

        ne_t_arr = integrate.cumtrapz(ne_arr, age_arr)
        return ne_t_arr

    def emission_measure_avg(self, ion):
        """
        Calculates tau for each layer from the RS to CD for all ages

        Parameters
        ----------
        exp : str, base of directory corresponding to explosion model

        amb : str, end of directory corresponding to ambient medium and any extras

        CD_layer : int, layer number of CD in output

        model_num: number of output file to which you're integrating 
        e.g. snr_Ia_prof_a_1065.dat would be 1065

        ion      : str, ion symbol for which you want to calculate
        """
        n_spec = 19
        ion_name = ['H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S',
                    'Ar', 'Ca', 'Fe', 'Ni', 'Na', 'Al', 'P', 'Ti', 'Cr', 'Mn']
        # index of element, e.g., 4 = 'O', 11 = 'Fe'
        i_spec = ion_name.index(ion)
        nj = np.array([1, 2, 6, 7, 8, 10, 12, 14, 16, 18,
                      20, 26, 28, 11, 13, 15, 22, 24, 25])

        ind = np.zeros([n_spec])
        ix = 0
        for no in range(n_spec):
            ixn = ix + nj[no]
            ind[no] = ix
            ix = ixn + 1
        ind = np.array(ind, dtype="int")

        EM_x = self.ne * self.nH * self.ab[:, i_spec] * self.vol
        frac = self.xion[
            :, ind[i_spec]: ind[i_spec] + nj[i_spec] + 1
        ]  # gives array of ion fractions
        charges = np.arange(nj[i_spec] + 1)
        ionization = np.sum(charges * frac, axis=1)
        ionization_frac = np.sum(charges * frac, axis=1) / np.sum(frac, axis=1)
        ion_s_EM_avg = np.sum(ionization_frac * EM_x) / np.sum(EM_x)
        tau_EM_avg = np.sum(self.ztau * EM_x) / np.sum(EM_x)
        vel_EM_avg = np.sum(self.vel * EM_x) / np.sum(EM_x)

        return ionization, ion_s_EM_avg, tau_EM_avg, vel_EM_avg


# this is how i'll do this later when calling the class
# hd = HydroData(exp,amb,model_num)
# hd.rad
