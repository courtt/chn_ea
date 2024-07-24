# ----------------------------------------------------------
# X-ray spectral synthesis script (Herman Lee 09/2021)
# Prerequisites: ATOMDB, pyatomdb, numpy, scipy, matplotlib
# ----------------------------------------------------------
import sys,os
import numpy as np

print(sys.argv)	
 
dE 	= 1e-3 		#* spectral bin size [keV]
norm    = 4*np.pi*(3.086e+22)**2 # Normalization to distacne 10 kpc
csvfile = 'newtest.csv'
exps    = ['ddt24']
ambs    = ['0p04','0p1','0p2','0p3','1p0','2p0','5p0',
           '0p04_hiB','0p1_hiB','0p2_hiB','0p3_hiB','1p0_hiB','2p0_hiB','5p0_hiB']
outputs = [1002,1004,1006,1009,1010,1011,1012,1017,1020,1029,1040,1048,1100] #sequence number(s) of output data
with open(csvfile, "a",newline='') as myfile:
        myfile.write('expModel,windModel,age,RS,FS,FeKaCentroid_ej,FeKaLum_ej,EqvWidth_ej,FeKaCentroid_csm,FeKaLum_csm,EqvWidth_csm,FeKaCentroid_tot,FeKaLum_tot,EqvWidth_tot')


def calc_feka_vals(Ec,luminosity,diff_min): 
        flux = luminosity / norm
        luminosity = np.log10(luminosity)
        flux = np.log10(flux)

        du_forward = (flux[1:] - flux[:-1])/(Ec[1:] - Ec[:-1])  # forward derivative
        # dub = (flux [:-1] - flux[1:]) / (Ec[:-1] - Ec[1:])  # backward derivative

        cont = np.polyfit(Ec[:-1][abs(du_forward) < diff_min],
                        flux[:-1][abs(du_forward) < diff_min], 3)
        continuum = np.poly1d(cont)
        subtracted_flux = (10**flux[(Ec > 6.3) & (Ec < 6.9)]) - 10**continuum(Ec[(Ec > 6.3) & (Ec < 6.9)])

        centroid = np.sum(Ec[(Ec > 6.3) & (Ec < 6.9)] * subtracted_flux * dE) / np.sum(subtracted_flux * dE)
        feka_lum = np.sum(subtracted_flux * dE)*norm/1e40  # ph/s
        eq_width = np.sum((1 - 10**flux[(Ec > 6.3) & (Ec < 6.9)]/10**continuum(Ec)[(Ec > 6.3) & (Ec < 6.9)]) *
                        dE)
        return centroid, feka_lum, eq_width

for exp in exps:
        for amb in ambs:
                for output in outputs:
                        print(exp,amb,output)

                        # This change is to read in the explosion_wind model as the first argument
                        infile  = f'../../data/{exp}_{amb}/output/snr_Ia_prof_a_{output}.dat'
                        outfile = f'../../data/{exp}_{amb}/output/sed_raw_{output}.dat'
                        shockfile=f'../../data/{exp}_{amb}/output/snr_Ia_shock.dat'


                        dat_prof = np.loadtxt(infile)
                        age_yr = dat_prof[0,1]
                        if not os.path.exists(outfile):
                                print(outfile,'does not exist')

                        dat_shock = np.loadtxt(shockfile)
                        age_dat = dat_shock[:, 0]  # years
                        fs_dat  = dat_shock[:, 1]  # cm
                        rs_dat  = dat_shock[:, 2]  # cm
                        cd_dat  = dat_shock[:, 3]  # cm
                        fs_vel_dat = dat_shock[:, 4]  # km/s
                        rs_vel_dat = dat_shock[:, 5]  # km/s

                        idx = (np.abs(age_dat - age_yr)).argmin()
                        
                        rs = rs_dat[idx]
                        fs = fs_dat[idx]

                        dat_sed = np.loadtxt(outfile)
                        Ec      = dat_sed[:,0]
                        lum_ej  = dat_sed[:,1]
                        lum_csm = dat_sed[:,2]
                        lum_tot = dat_sed[:,3]

                        centroid_ej,  feka_lum_ej,  eqw_ej  = calc_feka_vals(Ec,lum_ej,10)
                        centroid_csm, feka_lum_csm, eqw_csm = calc_feka_vals(Ec,lum_csm,10)
                        centroid_tot, feka_lum_tot, eqw_tot = calc_feka_vals(Ec,lum_tot,10)
                        
                        newline = '\n'

                        #with open("function_test.csv", "a",newline='') as myfile:
                        #        myfile.write("\n %s, %s, %f, %f, %f, %e, %f, %f, %e, %f, %f,%e, %f" % (exp,amb,age_yr, rs, fs,centroid_ej,  feka_lum_ej,  eqw_ej, centroid_csm,  feka_lum_csm,  eqw_csm, centroid_tot, feka_lum_tot, eqw_tot ))
                        with open(csvfile, "a",newline='') as myfile:
                                myfile.write(f"\n{exp},{amb},{age_yr},{rs},{fs},{centroid_ej:.6f},{feka_lum_ej:.6e},{eqw_ej:.6f},{centroid_csm:.6f},{feka_lum_csm:.6e},{eqw_csm:.6f},{centroid_tot:.6f},{feka_lum_tot:.6e},{eqw_tot:.6f}")
	

