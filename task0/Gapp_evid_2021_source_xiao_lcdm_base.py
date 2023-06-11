import numpy as np
from gapp import dgp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy import optimize , stats
from scipy.special import ndtri
import matplotlib.pyplot as plt
import dynesty
from dynesty import NestedSampler
from dynesty import DynamicNestedSampler
from dynesty.utils import resample_equal
from dynesty import plotting as dyplot
import corner
E0rest=42.5
sE0rest=27.5
Erest=370.0/2
sErest=65.0
print (Erest)
H0=70
data = np.genfromtxt('liv2022short.dat')


#data = data[:72,:]
z_com=data[:,0]
E0_comb=E0rest/(1+z_com)
delta_E_comb = Erest/(1+z_com)
delta_t_comb = data[:,5]
sigma_t = data[:,6]
#z_com = data[:,4]
#sigma_E = data[:,5]
sigma_E=np.sqrt(42.5**2+185**2)

def K(z, n):
#    f = lambda x: ((1+x)**n)/h_gp(x)
    f = lambda x: ((1+x)**n)/np.sqrt(0.3*(1+x)**3 + 0.7)
    return [quad(f, 0, zarray)[0] for zarray in z]
K_z1 = K(z_com,1)
K_z2 = K(z_com,2)

init_vals_1 = [1e16,1e-5,1.0]
init_vals_2 = [-1e7,1e-6,2.0]


def null_hypo(z, tau, alpha):
#    Erest=E*(1+z)
#    E0rest=E_0*(1+z)
    return (1+z)*tau*(Erest**alpha - E0rest**alpha)
	
def linear(z, E_qg, tau, alpha):
    K_z= np.asarray(K_z1)
    return (1+z)*tau*(Erest**alpha - E0rest**alpha) + (-(10**14)/(H0*3.24))*((Erest - E0rest)*K_z/(E_qg*(1+z)))

def quadratic(z, E_qg, tau, alpha):
    E_0=E0rest/(1+z)
    E=Erest/(1+z)
    K_z = np.asarray(K_z2)
    return (1+z)*tau*(Erest**alpha - E0rest**alpha) + (-1.5*(10**8)/(H0*3.24))*((E**2 - E_0**2)*K_z/E_qg**2)
    
#------------------------------Using sigma_E for best-fit params----------------------------------------------------

init_guess_null = [1e-6, 1.0]
bound_null = ((0,5),(0,10))
def chi2_null(params_null, E0, delta_t, delta_E, z_com, sigma_t, sigma_E):
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'): #To supress overflow in multiply and divide runtime warn
        sigma_tot = np.sqrt(sigma_t**2 + (((1+z_com)*(1+z_com)*params_null[0]*(params_null[1]*Erest**(params_null[1]-1)))**2)*sigma_E**2)
        loss = (delta_t - null_hypo(z_com, params_null[0], params_null[1]))/sigma_tot
    return np.sum(loss**2)


init_guess_lin = [1e14, 1.0, 1.0]
bound_lin = ((1e14,4e14),(0,50),(0,2))
def chi2_linear(params1, E0, delta_t, delta_E, z_com, sigma_t, sigma_E):
    K_z=K_z1
    K_z = np.array(K_z)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        delta_E=Erest/(1+z_com)
        sigma_tot = np.sqrt(sigma_t**2 + (((1+z_com)*(1+z_com)*params1[1]*(params1[2]*delta_E**(params1[2]-1)) + (-(10**14)/(H0*3.24))*K_z/params1[0])**2)*sigma_E**2)
        loss = (delta_t - linear(z_com, params1[0], params1[1], params1[2]))/sigma_tot
    return np.sum(loss**2)

init_guess_quad = [1e6, 1.0, 1.0]
bound_quad = ((1e6,2e6),(0,100),(0,2))
def chi2_quad(params2, E0, delta_t, delta_E, z_com, sigma_t, sigma_E):
    K_z=K_z2
    K_z = np.array(K_z)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        delta_E=Erest/(1+z_com)
        sigma_tot = np.sqrt(sigma_t**2 + (((1+z_com)*(1+z_com)*params2[1]*(params2[2]*Erest**(params2[2]-1)) + (-1.5*(10**8)/(H0*3.24))*2*K_z/(params2[0]**2))**2)*sigma_E**2)
        loss = (delta_t - quadratic((E0,delta_E,z_com), params2[0], params2[1], params2[2]))/sigma_tot	
    return np.sum(loss**2)
#res_quad = minimize(chi2_quad, init_guess_quad, args=(E0_comb,delta_t_comb,delta_E_comb,z_com,sigma_t,sigma_E), bounds=bound_quad, method='SLSQP')
#print(res_quad)

#--------------------------------------------------------------------------------------------------------------------------------
#Best-fit parameters with only sigma_t, to be ignored.    
#params_null_comb,_ = curve_fit(null_hypo, (E0_comb,delta_E_comb ,z_com), delta_t_comb, sigma=sigma_t)
#diff0 = (null_hypo((E0_comb,delta_E_comb ,z_com),params_null_comb[0],params_null_comb[1]) - delta_t_comb)/sigma_t
#print('Reduced chi2 for null hypo:', np.sum(diff0**2)/(len(diff0)-2))
#params1_comb,_ = curve_fit(linear, (E0_comb,delta_E_comb ,z_com), delta_t_comb, sigma=sigma_t,p0=init_vals_1)
#diff1=(linear((E0_comb,delta_E_comb ,z_com),params1_comb[0],params1_comb[1],params1_comb[2])-delta_t_comb)/sigma_t
#print('Reduced chi2 for n=1:', np.sum(diff1**2)/(len(diff1)-3))
#params2_comb,covar = curve_fit(quadratic, (E0_comb,delta_E_comb ,z_com), delta_t_comb, sigma=sigma_t,p0=init_vals_2,maxfev=1500)
#diff2=(quadratic((E0_comb,delta_E_comb ,z_com),params2_comb[0],params2_comb[1],params2_comb[2])-delta_t_comb)/sigma_t
#print('Reduced chi2 for n=2',np.sum(diff2**2)/(len(diff2)-3))

#print('\nParameters for null hypothesis:')
#print('tau:',params_null_comb[0])
#print('alpha:',params_null_comb[1])
#print('\nParameters for n=1 with combined data:')
#print('E_QG:',params1_comb[0])
#print('tau:',params1_comb[1])
#print('alpha:', params1_comb[2])
#print('\nParameters for n=2 with combined data:')
#print('E_QG:',params2_comb[0])
#print('tau:',params2_comb[1])
#print('alpha:', params2_comb[2])


nlive=1024
ndim=3
bound='multi'
sample='unif'
tol=0.1

def plotpostsnull(samples,figfile):
    labels=[r'$\tau$',r'$\alpha$']
    fig = corner.corner(samples, labels=labels,levels=(0.68,0.90),color = 'b',alpha=0.1,fill_contours = 1,show_titles=True,title_fmt='.3f',max_n_ticks = 3, title_kwargs={"fontsize": 14},label_kwargs={"fontsize": 16},contour_kwargs={"color": 'r'})
    fig.savefig(figfile)
    
#null hypothesis
def log_likelihood_null(theta0):
    tau0, alpha0 = theta0
    null_model = null_hypo(z_com, tau0, alpha0)
    sigma_tot = np.sqrt(sigma_t**2 + (((1+z_com)*tau0*(alpha0*Erest**(alpha0-1)))**2)*sErest**2 + (((1+z_com)*tau0*(alpha0*E0rest**(alpha0-1)))**2)*sE0rest**2)
    return sum(stats.norm.logpdf(*args) for args in zip(delta_t_comb,null_model,sigma_tot))
    #return sum(stats.norm.logpdf(*args) for args in zip(delta_t_comb,null_model,sigma_t))
    #return -0.5*np.sum((delta_t_comb-null_model)**2/sigma_t**2)

def prior_transform_null(theta0):
    #using same as n=1 as told.
    tau0, alpha0 = theta0
    #tau0 = tau0*10
    #alpha0 = alpha0*1
    return np.array([10*tau0 - 5, alpha0 - 0.5])

#with  Pool() as pool:
null_sampler = NestedSampler(log_likelihood_null, prior_transform_null, 2, bound=bound, sample=sample, nlive=nlive)
null_sampler.run_nested(dlogz=tol, print_progress=False)
res_null = null_sampler.results
print(res_null.summary())
print('Log Bayesian evidence for null hypo:',res_null.logz[-1])
#weights = np.exp(res_null['logwt'] - res_null['logz'][-1])
#samples_null = resample_equal(res_null.samples, weights)
#plotpostsnull(samples_null,'nullhypothesis.pdf')
#exit(0)
#n=1
#theta1 = (params1_comb[0],params1_comb[1],params1_comb[2])

def plotposts(samples,figfile):
    """
    Function to plot posteriors using corner.py and scipy's gaussian KDE function.
    """
    labels=[r'$\log (E_{QG}) $',r'$\tau$',r'$\alpha$']
    fig = corner.corner(samples, labels=labels,levels=(0.68,0.90),color = 'b',alpha=0.1,fill_contours = 1,show_titles=True,title_fmt='.3f',max_n_ticks = 3, title_kwargs={"fontsize": 14},label_kwargs={"fontsize": 16},contour_kwargs={"color": 'r'})
    fig.savefig(figfile)

def log_likelihood(theta1):
    log10E_QG1, tau1, alpha1 = theta1
    E_QG1=10**log10E_QG1
    model1 = linear(z_com, E_QG1, tau1, alpha1)
    sigma_tot = np.sqrt(sigma_t**2 + (((1+z_com)*tau1*(alpha1*Erest**(alpha1-1)))**2)*sErest**2 + (((1+z_com)*tau1*(alpha1*E0rest**(alpha1-1)))**2)*sE0rest**2 + (sErest**2+sE0rest**2)*(((10**14)/(H0*3.24))*(np.asarray(K_z1)/(E_QG1*(1+z_com))))**2)
    return sum(stats.norm.logpdf(*args) for args in zip(delta_t_comb,model1,sigma_tot))
    #return sum(stats.norm.logpdf(*args) for args in zip(delta_t_comb,model1,sigma_t))
    #return -0.5*np.sum((delta_t_comb-model1)**2/sigma_t**2)

def prior_transform(theta1):
    log10E_QG1, tau1, alpha1 = theta1
#    E_QG1 = E_QG1*9*10**15 + 1*10**15 #[1e15,1e16]
    #tau1 = tau1*10 #[0,1]
    #alpha1 = alpha1*1  #[1,2]
    return np.array([13*log10E_QG1+6, 10*tau1 - 5, alpha1 - 0.5])
sampler1 = NestedSampler(log_likelihood, prior_transform, ndim, bound=bound, sample=sample, nlive=nlive)
sampler1.run_nested(dlogz=tol, print_progress=False)
res1 = sampler1.results
print(res1.summary())
print('Log Bayesian evidence for n=1 model:',res1.logz[-1])
#print('n=1 vs n=1 (full prior range):', np.exp(res1.logz[-1] - 66.53))
#exit(0)

weights = np.exp(res1['logwt'] - res1['logz'][-1])
samples_dynesty1 = resample_equal(res1.samples, weights)

#plotposts(samples_dynesty1,'LIV_linear.pdf')

#exit(0)

#n=2
#theta2 = (params2_comb[0],params2_comb[1],params2_comb[2])
def log_likelihood_quad(theta2):
    log10E_QG2, tau2, alpha2 = theta2
    E_QG2=10**log10E_QG2
    model2 = quadratic(z_com, E_QG2, tau2, alpha2)
#    sigma_tot = np.sqrt(sigma_t**2 + (((1+z_com)*tau2*(alpha2*delta_E_comb**(alpha2-1)) + (-1.5*(10**8)/(H0*3.24))*2*np.asarray(delta_E_comb)*np.asarray(K_z2)/(E_QG2**2))**2)*sigma_E**2)
    sigma_tot = np.sqrt(sigma_t**2 + (((1+z_com)*tau2*(alpha2*Erest**(alpha2-1)))**2)*sErest**2 + (((1+z_com)*tau2*(alpha2*E0rest**(alpha2-1)))**2)*sE0rest**2 + (((1.5*(10**8)/(H0*3.24))*2*np.asarray(K_z2)/((1+z_com)*E_QG2)**2)**2)*((Erest*sErest)**2+(E0rest*sE0rest)**2))
    return sum(stats.norm.logpdf(*args) for args in zip(delta_t_comb,model2,sigma_tot))
    #return sum(stats.norm.logpdf(*args) for args in zip(delta_t_comb,model2,sigma_t))
    #return -0.5*np.sum((delta_t_comb-model2)**2/sigma_t**2)

def prior_transform_quad(theta2):
    log10E_QG2, tau2, alpha2 = theta2
#    E_QG2 = E_QG2*(1*10**9 - 1*10**6) + 1*10**6 #[1e6,1e9]
    #tau2 = tau2*10 #[0,10]
    #alpha2 = alpha2*1 #[1,2]
    return np.array([13*log10E_QG2+6, 10.0*tau2 - 5, alpha2 - 0.5])



sampler2 = NestedSampler(log_likelihood_quad, prior_transform_quad, ndim, bound=bound, sample=sample, nlive=nlive)
sampler2.run_nested(dlogz=tol, print_progress=False)
res2 = sampler2.results
print(res2.summary())
print('Log Bayesian evidence for n=2 model:',res2.logz[-1])


weights = np.exp(res2['logwt'] - res2['logz'][-1])
samples_dynesty2 = resample_equal(res2.samples, weights)

#Results of Bayesian Model Comparison:
print('n=1 vs n=1 (full prior range):', np.exp(res1.logz[-1] - res_null.logz[-1]))
print('n=2 vs n=2 (full prior range):', np.exp(res2.logz[-1] - res_null.logz[-1]))








