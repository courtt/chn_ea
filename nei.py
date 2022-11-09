import numpy as np
import os
from glob import glob
from scipy import integrate
class NEIData:
    # names = ['igrid'....]
    def __init__(self,exp,amb,model_num):
        # docstring for the class 
        # what is the purpose, some of the methods, input variables
        # save intermediate data files
        # for name in self.names:
        #   self.__dict__[name] = None
        path = os.getcwd()
        infile = path + exp + '_' + amb + '/output/snr_Ia_prof_a_' + str(model_num) +'.dat'
        dat = np.loadtxt(infile)
        # names = ['igrid'..'tau']
        # for name,indeces in zip(names,indeces):
        # self.__dict__[name] = dat:
        self.igrid= dat[:,0]
        self.age  = dat[:,1]
        self.rad  = dat[:,2] #cm
        self.mcorod=dat[:,3] #Lagrangian mass coordinate
        self.mgrid= dat[:,4] #mass of grid 
        self.vol  = dat[:,5]
        self.vel  = dat[:,8]
        self.Te   = dat[:,9]
        self.ne   = dat[:,11]
        self.nH   = dat[:,12]
        self.ni   = dat[:,13]
        self.ab   = 10**(dat[:,14:33]-12.0)
        self.jmax = np.argmax(self.ab)
        self.xion = dat[:,33:330]
        self.Ti   = dat[:,330:349]
        self.ztau = dat[:,349]


    def ne_t_calc(self,exp,amb,CD_layer):
        '''
        Calculates tau for each layer from the RS to CD for all ages

        Parameters
        ----------
        exp : str, base of directory corresponding to explosion model

        amb : str, end of directory corresponding to ambient medium and any extras

        CD_layer : int, layer number of CD in output
        '''

        infile = np.sort(glob('/Users/travis/pn_spectra/'+exp+'_'+amb+'/output/snr_Ia_prof_a_1*.dat'))
        yr_to_s = 365*24*3600
        layer_list  = np.arange(1,CD_layer+1)
        ne_t_arr    = np.zeros((CD_layer,101))
        ne_arr      = np.zeros((CD_layer,101))
        age_arr     = np.zeros(101)
        # for i in range(len(infile[:target_ind + 1])):
        for i in range(len(infile)):

            dat = np.loadtxt(infile[i])

            if len(dat.shape) == 1:
                dat = dat.reshape([1,len(dat)])
                    
            shocked_layers = dat[:,0]
            ne = dat[:,11]

            age_arr[i] = dat[0,1]*yr_to_s

            for j,layer in enumerate(layer_list):
                if layer in shocked_layers:
                    ne_arr[j,i] = ne[layer==shocked_layers]
                else:
                    ne_arr[j,i] = 0

        ne_t_arr = integrate.cumtrapz(ne_arr,age_arr)
        return ne_t_arr


    def emission_measure_avg(self,exp,amb,CD_layer,model_num,ion):
        '''
        Calculates tau for each layer from the RS to CD for all ages

        Parameters
        ----------
        exp : str, base of directory corresponding to explosion model

        amb : str, end of directory corresponding to ambient medium and any extras

        CD_layer : int, layer number of CD in output

        model_num: number of output file to which you're integrating 
        e.g. snr_Ia_prof_a_1065.dat would be 1065

        ion      : str, ion symbol for which you want to calculate
        '''
        infile = '/Users/travis/pn_spectra/'+exp+'_'+amb+'/output/snr_Ia_prof_a_'+ str(model_num) + '.dat'
        n_spec  = 19
        ion_name = ['H','He','C','N','O','Ne','Mg','Si','S','Ar','Ca','Fe','Ni','Na','Al','P','Ti','Cr','Mn']
        i_spec 	= ion_name.index(ion)  #index of element, e.g., 4 = 'O', 11 = 'Fe'
        nj  = np.array([1,2,6,7,8,10,12,14,16,18,20,26,28,11,13,15,22,24,25])
        km_to_cm = 1e5

        ind = np.zeros([n_spec])
        ix  = 0
        for no in range(n_spec):
            ixn = ix + nj[no]
            ind[no] = ix
            ix = ixn+1
        ind = np.array(ind,dtype='int') 

        dat = np.loadtxt(infile)
        if len(dat.shape) == 1:
            dat = dat.reshape([1,len(dat)])

        layer_list  = np.arange(1,CD_layer+1)
        i_CD = np.where(dat[:,0] == CD_layer)[0][0]

        shocked_layers = dat[:i_CD+1,0]
        xion = dat[:i_CD+1,33:]
        ne   = dat[:i_CD+1,11]
        nH   = dat[:i_CD+1,12]
        vol  = dat[:i_CD+1,5]
        ab   = 10**(dat[:i_CD+1,14:33]-12.0)
        vel  = dat[:i_CD+1,8]*km_to_cm


        EM_x = ne*nH*ab[:,i_spec]*vol     
        frac = xion[:i_CD+1,ind[i_spec]:ind[i_spec]+nj[i_spec]+1] #gives array of ion fractions
        charges = [np.arange(nj[i_spec]+1) for i in range(i_CD+1)]
        ionization_state = np.sum(charges*frac,axis=1)/np.sum(frac,axis=1)
        ne_t_arr = self.ne_t_calc(exp,amb,CD_layer)
        tau = ne_t_arr[:,model_num - 1000 - 1][np.isin( layer_list,shocked_layers)] # the -1 is because there is no tau for the first output bc no time elapsed
        ion_s_EM_avg = np.sum(ionization_state*EM_x) / np.sum(EM_x)
        tau_EM_avg = np.sum(tau*EM_x) / np.sum(EM_x)
        vel_EM_avg = np.sum(vel*EM_x) / np.sum(EM_x)

        return ion_s_EM_avg, tau_EM_avg, vel_EM_avg


# this is how i'll do this later when calling the class
# hd = HydroData(exp,amb,model_num)
# hd.rad

