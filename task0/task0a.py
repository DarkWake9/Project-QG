import numpy as np
import os
import matplotlib.pyplot as plt
import dynesty as dyn
from scipy.special import ndtri
import scipy.stats as stats
import multiprocessing as mul
import pickle
arr = np.loadtxt(os.getcwd() + '/asciidataof_fig1/32lag/GRB081221.txt')
# # plt.plot(arr[:,0], arr[:,1], '*')
# plt.errorbar(arr[:,0], arr[:,1], yerr=arr[:,2], fmt='*')
# plt.show()
# fig, axs = plt.subplots(8, 4, figsize=(16, 16))
# pltr = -1
# pltc = 0
# for file in os.listdir(os.getcwd() + '/asciidataof_fig1/32lag/'):
#     arr = np.loadtxt(os.getcwd() + '/asciidataof_fig1/32lag/' + file)
#     axs[pltc // 4, pltc%4].errorbar(arr[:,0], arr[:,1], yerr=arr[:,2], fmt='*')
#     axs[pltc // 4, pltc%4].set_title(file.replace('.txt', ''))
#     axs[pltc // 4, pltc%4].set_xlabel('E_obs')
#     axs[pltc // 4, pltc%4].set_xscale('log')
#     axs[pltc // 4, pltc%4].set_ylabel('lag (s)')
#     pltc += 1

    
    
# plt.tight_layout()    
# plt.show()
## $\Delta t_{int} = \zeta \left(\dfrac{E - E_0}{E_b}\right)^{\alpha_1} \{\dfrac{1}{2}\left[1 + \left(\dfrac{E - E_0}{E_b}\right)^{(\alpha_2 - \alpha_1)\mu}\right]\}$
# y = $\Delta t_{int}$

# x = $E$

def MODEL_delta_t_intrinstic(E, Eb, alpha1, alpha2, mu, zeta, E0):
    '''
    Broken power law model for lag-energy relation
    
    Parameters
    ----------
    E : float
        Energy in keV
    Eb : float
        Break energy in keV
    alpha1 : float
        Slope below break/transition energy
    alpha2 : float
        Slope above break/transition energy
    mu : float
        Transition smoothness parameter
    zeta : float
        Normalization constant
    E0 : float
        Energy shift parameter
        
    Returns
    -------
    delta_t : float
        Time lag in seconds
    '''
    E0b = (E - E0)/Eb
    return zeta * (E0b ** alpha1) * (1 + E0b ** ((alpha2 - alpha1)*mu))/2
data = [arr[:,0], arr[:,1], arr[:,2]]
M = len(data[0])
LN2PI = np.log(2. * np.pi)
x = arr[:,0]
y = arr[:,1]
yerr = arr[:,2]
norm = -0.5 * M * LN2PI - np.sum(np.log(yerr))

def loglklhood(theta, *args):
    # if E0 in args:
    E0 = args[0]
    # x, y, yerr = data
    Eb, alpha1, alpha2, mu, zeta = theta
    model = MODEL_delta_t_intrinstic(x, Eb, alpha1, alpha2, mu, zeta, E0)
    
    # chisq = np.sum(((y-MODEL_delta_t(x, *theta))/yerr)**2)
    # return norm - chisq/2.0
    return sum(stats.norm.logpdf(*args) for args in zip(y,model,yerr))



def prior_transform(theta):
    Eb, alpha1, alpha2, mu, zeta = theta
    # if E0 in args:
    
   
    return [5000*Eb, -3+13*alpha1, -10+13*alpha2, 3*mu, 4*zeta]
    
    
    
with dyn.pool.Pool(12, loglklhood, prior_transform) as Pool:
    sampler = dyn.NestedSampler(loglklhood, prior_transform, ndim=5, bound='multi', sample='rwalk', pool=Pool, nlive=1024, update_interval=100000, logl_args=[10])
    sampler.run_nested( dlogz=0.1)
    results = sampler.results
    
print(results.summary())
print('\n')
sampler = dyn.NestedSampler(loglklhood, prior_transform, ndim=5, bound='multi', sample='rwalk', nlive=1024, update_interval=100000, logl_args=[10])
sampler.run_nested( dlogz=0.1)
results = sampler.results
print(results.summary())
with open('./pickle/dynesty_results.pkl', 'wb') as f:
    pickle.dump(results, f)
    

