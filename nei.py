import numpy as np
import os
from glob import glob
class NEIData:
    # names = ['igrid'....]
    def __init__(self,exp,amb,model_num):
        # docstring for the class 
        # what is the purpose, some of the methods, input variables
        # save intermediate data files
        # for name in self.names:
        #   self.__dict__[name] = None
        self.igrid  = None
        self.age    = None
        self.rad    = None
        self.vol    = None
        self.vel    = None
        self.Te     = None
        self.ne     = None
        self.nH     = None
        self.ni     = None
        self.ab     = None
        self.jmax   = None
        self.xion   = None
        self.Ti     = None
        self.tau    = None
        self.read_columns(exp,amb,model_num)


    def read_columns(self,exp,amb,model_num): 
        path = os.cwd()
        infile = path + exp + '_' + amb + '/output/snr_Ia_prof_a_' + str(model_num) +'.dat'
        dat = np.loadtxt(infile)
        # names = ['igrid'..'tau']
        # for name,indeces in zip(names,indeces):
        # self.__dict__[name] = dat:
        igrid= dat[:,0]
        age  = dat[:,1]
        rad  = dat[:,2] #cm
        vol  = dat[:,5]
        vel	 = dat[:,8]
        Te   = dat[:,9]
        ne   = dat[:,11]
        nH   = dat[:,12]
        ni   = dat[:,13]
        ab   = 10**(dat[:,14:33]-12.0)
        jmax = np.argmax(ab)
        xion = dat[:,33:330]
        Ti   = dat[:,330:349]
        tau  = dat[:,349]

        self.igrid = igrid
        self.age   = age  
        self.rad   = rad  
        self.vol   = vol  
        self.vel   = vel	 
        self.Te    = Te   
        self.ne    = ne   
        self.nH    = nH       
        self.ni    = ni   
        self.ab    = ab   
        self.jmax  = jmax 
        self.xion  = xion 
        self.Ti    = Ti   
        self.tau   = tau  



    def ne_t_calc(exp,amb,model_num,CD_layer):
        '''
        Calculates tau for each layer from the RS to CD for all ages
        up to specified age 

        Parameters
        ----------
        exp : str, base of directory corresponding to explosion model

        amb : str, end of directory corresponding to ambient medium and any extras

        model_num: number of output file to which you're integrating 
        e.g. snr_Ia_prof_a_1065.dat would be 1065

        CD_layer : int, layer number of CD in output
        '''
        infile = np.sort(glob('./'+ exp + '_' + amb + '/output/snr_Ia_prof_a_1*.dat'))
        target_file = './'+ exp + '_' + amb + '/output/snr_Ia_prof_a_' + str(model_num) +'.dat'
        target_ind = np.where(infile==target_file)[0][0]

        layer_list= [[] for x in range(CD_layer)]
        ne_list  = [[] for x in range(CD_layer)]
        age_list = [[] for x in range(CD_layer)]
        tau      = []
        for i in range(len(infile[:target_ind + 1])):
            dat = np.loadtxt(infile[i])

            if len(dat.shape) == 1:
                dat = dat.reshape([1,len(dat)])
                    
            shocked_layers = dat[:,0]
        
            ne = dat[:,11]
            age = dat[:,1]
            for layer in range(CD_layer):
                if any(dat[:,0] == layer):
                    layer_ind = np.where(shocked_layers == layer)[0][0]
                    layer_list[layer].append(layer_ind)
                    ne_list[layer].append(ne[layer_ind])
                    age_list[layer].append(age[layer_ind]*3.154e+7)
# create a big matrix 562x100 ne and tau
# age list 1-d array
# layer list                 
            
        for layer in range(CD_layer):
            if len(age_list[layer]) > 0:
                if len(age_list[layer]) == 1: 
                    tau.append(0)
                else:
                    tau.append(np.integrate.trapz(ne_list[layer],age_list[layer]))

        return tau # self.tau=tau

# this is how i'll do this later when calling the class
# hd = HydroData(exp,amb,model_num)
# hd.rad
'''
infile = np.sort(glob('/Users/travis/pn_spectra/ddt40_2p0_uniform/output/snr_Ia_prof_a_1*.dat'))
target_file = '/Users/travis/pn_spectra/ddt40_2p0_uniform/output/snr_Ia_prof_a_1100.dat'
target_ind = np.where(infile==target_file)[0][0]
CD_layer=400 

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

    age_arr[i] = dat[0,1]

    for j,layer in enumerate(layer_list):
        if layer in shocked_layers:
            ne_arr[j,i] = ne[layer==shocked_layers]
        else:
            ne_arr[j,i] = 0

print(ne_arr[0])
'''