import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import dynesty as dyn
from scipy.special import ndtri
import multiprocessing as mul
from scipy import stats
import pickle
from scipy.integrate import quad
from corner import corner

grbname = 'GRB081221.txt'




H0=70 #km/s/Mpc Taken from Sir's Code

parser = argparse.ArgumentParser(description='GRBNAME')
parser.add_argument('grbname', type=str, help='Name of GRB')
argss = parser.parse_args()
grbname = argss.grbname



arr = np.loadtxt(os.getcwd() + '/asciidataof_fig1/32lag/'+grbname)
#Properties of GRB081221
E0 = 10
E0rest = E0
Erest = arr[:,0]    #in keV
z_com = 2.26 #redshift
def MODEL_delta_t_intrinsic(E, Eb, alpha1, alpha2, mu, zeta, E0):
    '''
    Broken power law model for lag-energy relation (NULL hypothesis)
    
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


x = arr[:,0]
y = arr[:,1]
yerr = arr[:,2]


data = [arr[:,0], arr[:,1], arr[:,2]]


M = len(data[0])


x = arr[:,0]
y = arr[:,1]
yerr = arr[:,2]


def loglklhood_null_HP(theta, *args):
    # if E0 in args:
    E0 = args[0]
    # x, y, yerr = data
    Eb, alpha1, alpha2, mu, zeta = theta
    model = MODEL_delta_t_intrinsic(x, Eb, alpha1, alpha2, mu, zeta, E0)
    
    # chisq = np.sum(((y-MODEL_delta_t(x, *theta))/yerr)**2)
    # return norm - chisq/2.0
    return sum(stats.norm.logpdf(*args) for args in zip(y,model,yerr))




def prior_transform(theta):
    Eb, alpha1, alpha2, mu, zeta = theta
    # if E0 in args:
    
   
    return [5000*Eb, -3+13*alpha1, -10+13*alpha2, 3*mu, 4*zeta]


def int_over_red_shift(z, n):
    '''
    Integrate over redshift
    
    Parameters
    ----------
    z : float
        Redshift
    n : float
        LIV Polynomial power        
    '''
    
    
    #    f = lambda x: ((1+x)**n)/h_gp(x)
    f = lambda x: ((1+x)**n)/np.sqrt(0.3*(1+x)**3 + 0.7)
    return quad(f, 0, z)[0]


K_z1 = int_over_red_shift(z_com,1)
K_z2 = int_over_red_shift(z_com,2)

# def null_hypo(z, tau, alpha):
# #    Erest=E*(1+z)
# #    E0rest=E_0*(1+z)
#     return (1+z)*tau*(Erest**alpha - E0rest**alpha)  ---------------> already covered by MODEL_delta_t_intrinsic!!??
	
def linear(E, z, E_qg, Eb, alpha1, alpha2, mu, zeta):
    K_z= np.asarray(K_z1)
    # return (1+z)*tau*(Erest - E0rest) + (-(10**14)/(H0*3.24))*((Erest - E0rest)*K_z/(E_qg*(1+z)))
    return MODEL_delta_t_intrinsic(E, Eb, alpha1, alpha2, mu, zeta, E0) + (-(10**14)/(H0*3.24))*((Erest - E0rest)*K_z/(E_qg*(1+z)))

def quadratic(E, z, E_qg, Eb, alpha1, alpha2, mu, zeta):
    E_0=E0rest/(1+z)
    E=Erest/(1+z)
    K_z = np.asarray(K_z2)
    # return (1+z)*tau*(Erest**2 - E0rest**2) + (-1.5*(10**8)/(H0*3.24))*((E**2 - E_0**2)*K_z/E_qg**2)
    return MODEL_delta_t_intrinsic(E, Eb, alpha1, alpha2, mu, zeta, E0) + (-1.5*(10**8)/(H0*3.24))*((E**2 - E_0**2)*K_z/E_qg**2)



def loglklhood_LIV_lin(theta1):
    
    E_qg, Eb, alpha1, alpha2, mu, zeta = theta1
    
    model = linear(x, z_com, E_qg, Eb, alpha1, alpha2, mu, zeta)
    
    return sum(stats.norm.logpdf(*rgs) for rgs in zip(y,model,yerr))


def loglklhood_LIV_quad(theta2):
    
    E_qg, Eb, alpha1, alpha2, mu, zeta = theta2
    
    model = linear(x, z_com, E_qg, Eb, alpha1, alpha2, mu, zeta)
    
    return sum(stats.norm.logpdf(*rgs) for rgs in zip(y,model,yerr))
    
    
    
def prior_transform_LIV_lin(theta1):
    E_qg, Eb, alpha1, alpha2, mu, zeta = theta1
    return [1e20*E_qg, 5000*Eb, -3+13*alpha1, -10+13*alpha2, 3*mu, 4*zeta]

def prior_transform_LIV_quad(theta2):
    E_qg, Eb, alpha1, alpha2, mu, zeta = theta2
    return [1e15*E_qg, 5000*Eb, -3+13*alpha1, -10+13*alpha2, 3*mu, 4*zeta]



print('#'*100)
print('NULL HYPOTHESIS')



nlive = 1024
ndim_NULL = 5
ndim_LIV = 6



# NULL hypothesis
with dyn.pool.Pool(12, loglklhood_null_HP, prior_transform) as Pool:
    sampler0 = dyn.NestedSampler(loglklhood_null_HP, prior_transform, ndim=ndim_NULL, bound='multi', sample='rwalk', pool=Pool, nlive=nlive,  logl_args=[10])
    sampler0.run_nested( dlogz=0.1)
    results0 = sampler0.results
    
Pool.close()
print(results0.summary())

with open('./pickle/dynesty_results_null.pkl', 'wb') as f:
    pickle.dump(results0, f)

with open('./pickle/dynesty_sampler_null.pkl', 'wb') as f:
    pickle.dump(sampler0, f)


print('#'*100)
print('LINEAR LIV')

# Linear LIV
with dyn.pool.Pool(12, loglklhood_LIV_lin, prior_transform_LIV_lin) as Pool:
    sampler1 = dyn.NestedSampler(loglklhood_LIV_lin, prior_transform_LIV_lin, ndim=ndim_LIV, bound='multi', sample='rwalk', pool=Pool, nlive=nlive)
    sampler1.run_nested( dlogz=0.1)
    results1 = sampler1.results

Pool.close()
print(results1.summary())


with open('./pickle/dynesty_results_LIV_lin.pkl', 'wb') as f:
    pickle.dump(results1, f)

with open('./pickle/dynesty_sampler_LIV_lin.pkl', 'wb') as f:
    pickle.dump(sampler1, f)


print('#'*100)
print('QUADRATIC LIV')

# Quadratic LIV
with dyn.pool.Pool(12, loglklhood_LIV_quad, prior_transform_LIV_quad) as Pool:
    sampler2 = dyn.NestedSampler(loglklhood_LIV_quad, prior_transform_LIV_quad, ndim=ndim_LIV, bound='multi', sample='rwalk', pool=Pool, nlive=nlive)
    sampler2.run_nested( dlogz=0.1)
    results2 = sampler2.results
    
Pool.close()
print(results2.summary())

with open('./pickle/dynesty_results_LIV_quad.pkl', 'wb') as f:
    pickle.dump(results2, f)
    

with open('./pickle/dynesty_sampler_LIV_quad.pkl', 'wb') as f:
    pickle.dump(sampler2, f)


def plotsamples(results, figname='corner'):
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    
    corner(samples, weights=weights, labels=["E_qg", "Eb", "alpha1", "alpha2", "mu", "zeta"], levels=[0.68, 0.9], show_titles=True, title_kwargs={"fontsize": 12})
    plt.savefig(figname + '.png', facecolor='white')
    plt.show()


plotsamples(results0, figname='null_HP' + grbname.replace('.txt',''))


plotsamples(results1, figname='LIV_lin' + grbname.replace('.txt',''))


plotsamples(results2, figname='LIV_quad' + grbname.replace('.txt',''))


print('#'*100)
print(grbname.replace('.txt','') + ' DONE')
print('#'*100)


