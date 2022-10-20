import numpy as np
import os
from glob import glob
class NEIData:
    def __init__(self,exp,amb,model_num):
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

            
        for layer in range(CD_layer):
            if len(age_list[layer]) > 0:
                if len(age_list[layer]) == 1: 
                    tau.append(0)
                else:
                    tau.append(np.integrate.trapz(ne_list[layer],age_list[layer]))

        return tau