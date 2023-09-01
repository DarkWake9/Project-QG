import numpy as np
import os
import matplotlib.pyplot as plt
import multiprocessing as mul
from scipy import stats
import pickle
from scipy.integrate import quad
from corner import corner
import pandas as pd
from scipy.stats import gaussian_kde
import dynesty as dyn


grbparam = pd.read_csv('GRBPARAM.csv', index_col=0)
grbname = 'GRB190114C.txt'
grb = 'GRB190114C'


#Properties of GRB
E0 = grbparam[grbname.replace('.txt','')].E0
E0rest = E0
z_com = grbparam[grbname.replace('.txt','')].redshift #redshift
H0=67.36 #Hubble constant km -1 Mpc s -1
omega_m = 0.315
omega_l = 1 - omega_m




newe = pd.read_csv('2010.16029.txt', sep='\s+', header=None)
ncpu=12
E_newe = (newe[1].values + newe[0].values)/2.0
y_newe = newe[2].values
yerr_newe = newe[3].values


#MODELS

#NULL model
def nullhp(E, alpha, tau):
    return (1 + z_com)*(tau * ((E ** alpha) - (E0 ** alpha)))
    


def int_z(z_prime, n):
    integ_fn = lambda z: (1+z)**n / np.sqrt(omega_m * (1+z)**3 + omega_l)
    return quad( integ_fn, a=0, b=z_prime)[0]

int_z1 = np.asarray(int_z(z_com, 1))
int_z2 = np.asarray(int_z(z_com, 2))

#LINEAR model
def linearhp(E, logEqg, alpha, tau):
    
    e0qg = (E - E0) / (10 ** logEqg)
    
    return - (e0qg * int_z1)/H0 + nullhp(E, alpha, tau)

#QUADRATIC model
def quadhp(E, logEqg, alpha, tau):
    e0qg = (E**2 - E0 **2) / ((10 ** logEqg)**2)
    
    return -1.5 * (e0qg * int_z2)/H0 + nullhp(E, alpha, tau)



# Properties of GRB
E0 = grbparam[grbname.replace('.txt','')].E0
E0rest = E0

z_com = grbparam[grbname.replace('.txt','')].redshift #redshift

#PLOTTING FITS

nplot = 1000

E_n = np.linspace(min(E_newe), max(E_newe), nplot)


null2 = [nullhp(E_n[i], -0.84, -10.61) for i in range(nplot)]
lin2 = [linearhp(E_n[i], 14.49, -0.84, -10.61) for i in range(nplot)]
quad2 = [quadhp(E_n[i], 14.49, -0.84, -10.61) for i in range(nplot)]





plt.figure()
# plt.errorbar(Erest, y, yerr, fmt='o', color='black', label='data')
plt.errorbar(E_newe, y_newe, yerr_newe, fmt='o', color='red', label='2010.16029')
# plt.plot(E, np.round(null_fit, 12), label='Null fit')
# plt.plot(E, np.round(liv_lin_fit, 12),label='Linear fit')
# plt.plot(E, np.round(liv_quad_fit, 12), label='Quadratic fit')
plt.plot(E_n, np.round(null2, 12), label='Null fit 2010', ls='-.')
plt.plot(E_n, np.round(lin2, 12),label='Linear fit 2010', ls='-.')
plt.plot(E_n, np.round(quad2, 12), label='Quadratic fit 2010', ls='-.')
# plt.plot(E_n, np.round(null_fit_newe, 12), label='Null fit 2010', ls='--')
# plt.plot(E_n, np.round(lin_fit_newe, 12),label='Linear fit 2010', ls='--')
# plt.plot(E_n, np.round(quad_fit_newe, 12), label='Quadratic fit 2010', ls='--')
plt.xscale('log')
# plt.yscale('log')
# plt.ylim(min(y) - max(abs(yerr)), max(y) + max(abs(yerr)))
# plt.ylim(0.25, 0.9)
# plt.ylim(-200, 20)
plt.legend()
plt.xlabel('E (keV)')
plt.ylabel('lag (s)')
plt.title(grb)
plt.show()


plt.figure()
plt.errorbar(E_newe, y_newe, yerr_newe, fmt='o', label='2010.16029')
plt.plot(E_n, np.round(null_fit_newe, 12), label='Null fit', lw=5)
plt.plot(E_n, np.round(lin_fit_newe, 12),label='Linear fit', lw=3)
plt.plot(E_n, np.round(quad_fit_newe, 12), label='Quadratic fit')
plt.xscale('log')
# plt.yscale('log')
plt.ylim(min(y_newe) - max(abs(yerr_newe)), max(y_newe) + max(abs(yerr_newe)))
# plt.ylim(-200, 20)
plt.legend()
plt.xlabel('E (keV)')
plt.ylabel('lag (s)')
plt.title(grb)
plt.show()


def chi2_gof(x, y, yerr, fit_func, *fit_func_args):
        
    return np.sum(((y - fit_func(x, *fit_func_args))/yerr)**2)/(len(y) - len(fit_func_args))


gof_null = chi2_gof(Erest, y, yerr, nullhp, samples0[0], samples0[1])
gof_lin = chi2_gof(Erest, y, yerr, linearhp, samples1[0], samples1[1], samples1[2])
gof_quad = chi2_gof(Erest, y, yerr, quadhp, samples2[0], samples2[1], samples2[2])
gof_null_newe = chi2_gof(E_newe, y_newe, yerr_newe, nullhp, samples0[0], samples0[1])
gof_lin_newe = chi2_gof(E_newe, y_newe, yerr_newe, linearhp, samples1[0], samples1[1], samples1[2])
gof_quad_newe = chi2_gof(E_newe, y_newe, yerr_newe, quadhp, samples2[0], samples2[1], samples2[2])


print('fit\t\t\t LIU GOF\t 2010.16029 GOF')
print('Null fit:\t', gof_null, '\t', gof_null_newe)
print('Lin fit:\t', gof_lin, '\t', gof_lin_newe)
print('Quadr fit:\t', gof_quad, '\t', gof_quad_newe)





