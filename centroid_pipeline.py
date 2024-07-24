# ----------------------------------------------------------
# X-ray spectral synthesis script (Herman Lee 09/2021)
# Prerequisites: ATOMDB, pyatomdb, numpy, scipy, matplotlib
# ----------------------------------------------------------

import pyatomdb.atomdb
import pyatomdb.spectrum as spec 
from numpy import *
import pylab as pl
import scipy.ndimage as ndimage
import sys,os
from pathos.multiprocessing import ProcessingPool
import pathos as pa
import numpy as np

print(sys.argv)	
n_spec 	= 19  		#* no. of elements to include
i_CD   	= 500 		#* particle number at CD
dE 		= 1e-3 		#* spectral bin size [keV]
Emax 	= 10.0	 	#* spectral bin max [keV]
Emin 	= 1.0 		#* spectral bin min [keV]
NE 		= int((Emax-Emin)/dE + 1)
E 		= linspace(Emin,Emax,NE)
keV 	= 1e3*1.602e-12
norm    = 4*pi*(3.086e+22)**2
msun 	= 1.989e33
color 	= ['k','r','b','g','m','c','y',(1,0.5,0.3)]
nj     	= array([1,2,6,7,8,10,12,14,16,18,20,26,28,11,13,15,22,24,25])

ind = zeros([n_spec])
ix  = 0
for no in range(n_spec):
	ixn = ix + nj[no]
	ind[no] = ix
	ix = ixn+1
ind = array(ind,dtype='int') 

#infile  = "output/%s_prof_a_%s.dat"%(sys.argv[1],sys.argv[2])
#outfile = "sed_%s_%s.dat"%(sys.argv[1],sys.argv[2])
# This change is to read in the explosion_wind model as the first argument
infile  = "../../data/%s_%s/output/snr_Ia_prof_a_%s.dat"%(sys.argv[1],sys.argv[2],sys.argv[3])
outfile = "../../data/%s_%s/output/sed_raw_%s.dat"%(sys.argv[1],sys.argv[2],sys.argv[3])

dat = loadtxt(infile)
age_yr = dat[0,1]
if not os.path.exists(outfile):
        n_part = dat.shape[0]
        n_ion  = dat.shape[1]-33	

        print("\nModel: ",sys.argv[1],sys.argv[2],sys.argv[3],age_yr)
        print("Data dimension: ", n_part, n_ion)
        
        nei = spec.NEISession()
        nei.set_response(E,raw=True)
        nei.elements = list(nj) 
        
        #AG89 values
        ab_init   = pyatomdb.atomdb.get_abundance()
        ab_rel    = dict(zip(arange(1,31),zeros(30)))
        popul     = dict(zip(arange(1,31),zeros(30)))

        def spec_run(i):
                xion = dat[i,33:]
                ne   = dat[i,11]
                nH   = dat[i,12]
                ni   = dat[i,13]
                Te   = dat[i,9]
                vol  = dat[i,5]
                rad  = dat[i,2] #cm
                ab   = 10**(dat[i,14:33]-12.0)
                jmax = argmax(ab)
                igrid = int(dat[i,0])
                
                lum_ej  = zeros(len(E)-1)+1e-50 
                lum_csm = zeros(len(E)-1)+1e-50

                if (Te < 0.000862*keV/1.381e-16): 
                        return lum_ej, lum_csm

                for j in range(n_spec):
                        ab_rel[nj[j]] = (ab[j]/ab[jmax])/ab_init[nj[j]]
                        popul[nj[j]]  = xion[ind[j]:ind[j]+nj[j]+1]
                nei.abundsetvector = ab_rel
                if igrid==245:
                        brems_flux = pyatomdb.apec.calc_ee_brems(E,Te,ne)

                try:
                        if igrid < i_CD:	#you can output the DEM profile separately by storing ne*ni*vol against rad
                                lum_ej = ne*nH*ab[jmax]*nei.return_spectrum(Te,1e11,init_pop=popul,freeze_ion_pop=True,teunit='K')*vol #ph/s/bin

                        else:
                                lum_csm = ne*nH*ab[jmax]*nei.return_spectrum(Te,1e11,init_pop=popul,freeze_ion_pop=True,teunit='K')*vol #ph/s/bin
                except:
                        print("warning...", igrid, ne, Te, vol)
	
                sys.stdout.write("\rProgress:%i/%i (particle:%i) (peak Z:%i)"%(i+1,n_part,igrid,nj[jmax]))
                sys.stdout.flush()

                return lum_ej, lum_csm

        pool = ProcessingPool(pa.helpers.cpu_count()-1)
        result = array(pool.map(spec_run, range(n_part)),dtype=float)

        lum_ej = sum(result[:,0],0) #ph/s/bin
        lum_csm = sum(result[:,1],0)

        Ec = 0.5*(E[1:]+E[:-1])
        lum_ej /= dE
        lum_csm /= dE
        lum = lum_ej+lum_csm
        savetxt(outfile,array([Ec,lum_ej,lum_csm,lum]).T)

dats = loadtxt(outfile)
Ec  = dats[:,0]
lum_ej = dats[:,1]
lum_csm = dats[:,2]
lum = dats[:,3]

flux_ej = lum_ej / norm
lum_ej = np.log10(lum_ej)
flux_ej = np.log10(flux_ej)
diff_min = 10  # derivative threshold for continuum

duf = (flux_ej[1:] - flux_ej[:-1])/(Ec[1:] - Ec[:-1])  # forward derivative
dub = (flux_ej[:-1] - flux_ej[1:]) / \
      (Ec[:-1] - Ec[1:])  # backward derivative

cont = np.polyfit(Ec[:-1][abs(duf) < diff_min],
                  flux_ej[:-1][abs(duf) < diff_min], 3)
continuum = np.poly1d(cont)
sub_ej = (10**flux_ej[(Ec > 6.3) & (Ec < 6.9)]) - \
         10**continuum(Ec[(Ec > 6.3) & (Ec < 6.9)])

centroid = np.sum(Ec[(Ec > 6.3) & (Ec < 6.9)] * sub_ej *
                  dE) / np.sum(sub_ej * dE)
feka_lum = sum(sub_ej * dE)*norm/1e40  # ph/s
eq_width = np.sum((1 - 10**flux_ej[(Ec > 6.3) & (Ec < 6.9)]/10**continuum(Ec)[(Ec > 6.3) & (Ec < 6.9)]) *
                      dE)
with open("centroids_ddt24_beta_gamma_high.csv", "a",newline='') as myfile:
        myfile.write("\n %s, %s, %f, %f, %e, %f" % (sys.argv[1],sys.argv[2],age_yr, centroid, feka_lum, eq_width))

	
