# ### USE EQNS FROM THE CHINESE PAPER ONLY
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


ncpu = int(mul.cpu_count())
grbparam = pd.read_csv(os.getcwd() + '/../data/GRBPARAM.csv', index_col=0)

param_ranges_NULL = [(1e-20, 5000), (-3, 10), (-10, 3), (0, 3), (0, 4)]
param_range_lin = [(1e-20, 1e20), (1e-20, 5000), (-3, 10), (-10, 3), (0, 3), (0, 4)]
param_range_quad = [(1e-20, 1e15), (1e-20, 5000), (-3, 10), (-10, 3), (0, 3), (0, 4)]


# GRBs = ['GRB210619B', 'GRB210610B', 'GRB210204A', 'GRB201216C', 'GRB200829A', 'GRB200613A', 'GRB190114C', 'GRB180720B', 'GRB180703A', 'GRB171010A', 'GRB160625B', 'GRB160509A']
GRBs = [ 'GRB150821A', 'GRB150514A', 'GRB150403A', 'GRB150314A', 'GRB141028A', 'GRB140508A', 'GRB140206A', 'GRB131231A', 'GRB131108A', 'GRB130925A', 'GRB130518A']
# [ 'GRB130427A', 'GRB120119A', 'GRB100728A', 'GRB091003A', 'GRB090926A', 'GRB090618', 'GRB090328', 'GRB081221', 'GRB080916C']


for grb in GRBs:
    print('#'*100)
    print(grb, 'started')
    print('#'*100)
    grbname = grb + '.txt'
    grbname_wtht_ext = grbname.replace('.txt','')


    arr = np.loadtxt(os.getcwd() + '/asciidataof_fig1/32lag/'+grbname)
    data = [arr[:,0], arr[:,1], arr[:,2]]
    x = arr[:,0]
    y = arr[:,1]
    yerr = arr[:,2]

    #Properties of GRB
    E0 = grbparam[grbname.replace('.txt','')].E0
    E0rest = E0
    Erest = arr[:,0]    #in keV
    z_com = grbparam[grbname.replace('.txt','')].redshift #redshift
    H0=67.36 #Hubble constant km -1 Mpc s -1
    omega_m = 0.315
    omega_l = 1 - omega_m





    #MODELS

    #NULL model
    def nullhp(E, Eb, alpha1, alpha2, mu, zeta):
        
        eob = (E - E0) / (Eb)
        
        return zeta * (eob**alpha1) * ((0.5 * ((1 + eob)**(1/mu))) ** ((alpha2 - alpha1) * mu))


    def int_z(z_prime, n):
        integ_fn = lambda z: (1+z)**n / np.sqrt(omega_m * (1+z)**3 + omega_l)
        return quad( integ_fn, a=0, b=z_prime)[0]

    int_z1 = np.asarray(int_z(z_com, 1))
    int_z2 = np.asarray(int_z(z_com, 2))

    #LINEAR model
    def linearhp(E, logEqg, Eb, alpha1, alpha2, mu, zeta):
        
        e0qg = (E - E0) / (10 ** logEqg)
        
        return - (e0qg * int_z1)/H0 + nullhp(E, Eb, alpha1, alpha2, mu, zeta)

    #QUADRATIC model
    def quadhp(E, logEqg, Eb, alpha1, alpha2, mu, zeta):
        e0qg = (E**2 - E0 **2) / ((10 ** logEqg)**2)
        
        return -1.5 * (e0qg * int_z2)/H0 + nullhp(E, Eb, alpha1, alpha2, mu, zeta)


    #LOG-LIKELIHOODS
    def loglike_null(theta):
        Eb, alpha1, alpha2, mu, zeta = theta
        
        if alpha1 >= alpha2:
            model = nullhp(x,  Eb, alpha1, alpha2, mu, zeta)
            
            return sum(stats.norm.logpdf(*args) for args in zip(y,model,yerr))
        
        return -np.inf

    def loglike_linear(theta):
        logEqg, Eb, alpha1, alpha2, mu, zeta = theta
        
        if alpha1 >= alpha2:
            model = linearhp(x, logEqg, Eb, alpha1, alpha2, mu, zeta)
            
            return sum(stats.norm.logpdf(*args) for args in zip(y,model,yerr))
        
        return -np.inf

    def loglike_quad(theta):
        logEqg, Eb, alpha1, alpha2, mu, zeta = theta
        
        if alpha1 >= alpha2:
            model = quadhp(x, logEqg, Eb, alpha1, alpha2, mu, zeta)
            
            return sum(stats.norm.logpdf(*args) for args in zip(y,model,yerr))
        
        return -np.inf    


    #PRIORS
    Ebmax = 5000 #keV
    Ebmin = 0
    alpha1min = -3
    alpha1max = 10  
    alpha2min = -10
    alpha2max = 3
    mumin = 0
    mumax = 3
    zetamin = 0
    zetamax = 4
    eqglinmin = 1e-20
    eqglinmax = 1e20
    eqquadmin = 1e-20
    eqquadmax = 1e15
    logeq1min = 0
    logeq1max = 20
    logeq2min = 0
    logeq2max = 15


    #PRIOR DISTRIBUTIONS

    def prior_transform_null(theta):
        Eb, alpha1, alpha2, mu, zeta = theta
        return [Ebmax * Eb + Ebmin, (alpha1max - alpha1min) * alpha1 + alpha1min, (alpha2max - alpha2min) * alpha2 + alpha2min, (mumax - mumin) * mu + mumin, (zetamax - zetamin) * zeta + zetamin]

    def prior_transform_linear(theta):
        logEqg, Eb, alpha1, alpha2, mu, zeta = theta
        return [(logeq1max - logeq1min) * logEqg + logeq1min, Ebmax * Eb + Ebmin, (alpha1max - alpha1min) * alpha1 + alpha1min, (alpha2max - alpha2min) * alpha2 + alpha2min, mumax * mu + mumin, zetamax * zeta + zetamin]

    def prior_transform_quadratic(theta):
        logEqg, Eb, alpha1, alpha2, mu, zeta = theta
        return [(logeq2max - logeq2min) * logEqg + logeq2min, Ebmax * Eb + Ebmin, (alpha1max - alpha1min) * alpha1 + alpha1min, (alpha2max - alpha2min) * alpha2 + alpha2min, mumax * mu + mumin, zetamax * zeta + zetamin]


    #SAMPLING
    nlive = 1024
    
    with dyn.pool.Pool(ncpu, loglike_null, prior_transform_null) as pool0:
        sampler0 = dyn.NestedSampler(loglike_null, prior_transform_null, ndim=5, nlive = nlive, sample='rwalk', bound='multi', pool=pool0)
        sampler0.run_nested(dlogz=0.001)
        
    pool0.close()
    dyn.utils.save_sampler(sampler0, os.getcwd() + '/outputs/sampler_saves/' + grb + '_null_sampler.pkl')

    with dyn.pool.Pool(ncpu, loglike_linear, prior_transform_linear) as pool1:
        sampler1 = dyn.NestedSampler(loglike_linear, prior_transform_linear, ndim=6, nlive = nlive, sample='rwalk', bound='multi', pool=pool1)
        sampler1.run_nested(dlogz=0.001)
        
    pool1.close()
    dyn.utils.save_sampler(sampler1, os.getcwd() + '/outputs/sampler_saves/' + grb + '_linear_sampler.pkl')
    
    with dyn.pool.Pool(ncpu, loglike_quad, prior_transform_quadratic) as pool2:
        sampler2 = dyn.NestedSampler(loglike_quad, prior_transform_quadratic, ndim=6, nlive = nlive, sample='rwalk', bound='multi', pool=pool2)
        sampler2.run_nested(dlogz=0.001)
        
    pool2.close()
    dyn.utils.save_sampler(sampler2, os.getcwd() + '/outputs/sampler_saves/' + grb + '_quadratic_sampler.pkl')
    

    results0 = sampler0.results
    results1 = sampler1.results
    results2 = sampler2.results


    def smooth_plot(results, figname, labels=["logE_qg", "Eb(keV)", "alpha1", "alpha2", "mu", "zeta"]):
        plt.figure()
        weights = np.exp(results.logwt - results.logz[-1])
        samples = dyn.utils.resample_equal(  results.samples, weights)
        
        fig = corner(samples, weights=weights, labels=labels, levels=[0.68, 0.9], show_titles=True, title_kwargs={"fontsize": 12}, hist_kwargs={'density': True})
        ndim =samples.shape[1]
        for axidx, samps in zip([i*(ndim+1) for i in range(ndim)],samples.T):
            kde = gaussian_kde(samps)
            xvals = fig.axes[axidx].get_xlim()
            xvals = np.linspace(xvals[0], xvals[1], 100)
            fig.axes[axidx].plot(xvals, kde(xvals), color='firebrick')
            
        plt.suptitle(str(grb))
        plt.savefig(os.getcwd() + '/outputs/contours/' + grb + '_' + figname + '.png')
        # plt.show()


    smooth_plot(results0, 'nullhp', labels=["Eb(keV)", "alpha1", "alpha2", "mu", "zeta"])


    smooth_plot(results1, 'linearhp', labels=["logE_qg", "Eb(keV)", "alpha1", "alpha2", "mu", "zeta"])


    smooth_plot(results2, 'quadhp', labels=["logE_qg", "Eb(keV)", "alpha1", "alpha2", "mu", "zeta"])


    #PLOTTING FITS

    nplot = 1000
    E = np.linspace(min(Erest), max(Erest), nplot)
    samples0 = dyn.utils.resample_equal( results0.samples, np.exp(results0.logwt - results0.logz[-1]))
    # samples0 = np.median(samples0, axis=0)
    samples0 = samples0[np.argmax(results0.logl)]

    samples1 = dyn.utils.resample_equal( results1.samples, np.exp(results1.logwt - results1.logz[-1]))
    # samples1 = np.median(samples1, axis=0)
    samples1 = samples1[np.argmax(results1.logl)]

    samples2 = dyn.utils.resample_equal( results2.samples, np.exp(results2.logwt - results2.logz[-1]))
    # samples2 = np.median(samples2, axis=0)
    samples2 = samples2[np.argmax(results2.logl)]
    null_fit = [nullhp(E[i], samples0[0], samples0[1], samples0[2], samples0[3], samples0[4]) for i in range(nplot)]
    liv_lin_fit = [linearhp(E[i], samples1[0], samples1[1], samples1[2], samples1[3], samples1[4], samples1[5]) for i in range(nplot)]
    liv_quad_fit = [quadhp(E[i], samples1[0], samples1[1], samples1[2], samples1[3], samples1[4], samples1[5]) for i in range(nplot)]

    plt.figure()
    plt.errorbar(Erest, y, yerr, fmt='o', color='black', label='data')
    plt.plot(E, null_fit, label='Null fit')
    plt.plot(E, liv_lin_fit,label='Linear fit')
    plt.plot(E, liv_quad_fit, label='Quadratic fit')
    plt.xscale('log')
    # plt.yscale('log')
    plt.ylim(min(y) - max(abs(yerr)), max(y) + max(abs(yerr)))
    # plt.ylim(-200, 20)
    plt.legend()
    plt.xlabel('E (keV)')
    plt.ylabel('lag (s)')
    plt.title(grbname_wtht_ext)
    plt.show()
    plt.savefig(os.getcwd() + '/outputs/fits/' + grbname_wtht_ext + '_fit_logE.png', facecolor='white')
    # bayes_factor_lin = np.exp(results1.logz[-1] - results0.logz[-1])
    # bayes_factor_quad = np.exp(results2.logz[-1] - results0.logz[-1])

    print('Bayes factor for null model: ', results0.logz[-1], '+/-', results0.logzerr[-1])
    print('Bayes factor for linear LIV model: ', results1.logz[-1], '+/-', results1.logzerr[-1])
    print('Bayes factor for quadratic LIV model: ', results2.logz[-1], '+/-', results2.logzerr[-1])

    with open('outputs/BF/' + grb + '_results.txt', 'w') as f:
        f.write(str(results0.logz[-1]) + ',' + str(results0.logzerr[-1]) + '\n')
        f.write(str(results1.logz[-1]) + ',' + str(results1.logzerr[-1]) + '\n')
        f.write(str(results2.logz[-1]) + ',' + str(results2.logzerr[-1]) + '\n')

    f.close()
    f = []

    #SAVING GOODNESS OF FIT

    def chi2_gof(x, y, yerr, fit_func, *fit_func_args):
        '''
            Calculates the chi2/DOF for the function
        '''    
        return np.sum(((y - fit_func(x, *fit_func_args))/yerr)**2)/(len(y) - len(fit_func_args))
    
    
    
    gof_null = chi2_gof(Erest, y, yerr, nullhp, samples0[0], samples0[1], samples0[2], samples0[3], samples0[4])
    gof_lin = chi2_gof(Erest, y, yerr, linearhp, samples1[0], samples1[1], samples1[2], samples1[3], samples1[4], samples1[5])
    gof_quad = chi2_gof(Erest, y, yerr, quadhp, samples2[0], samples2[1], samples2[2], samples2[3], samples2[4], samples2[5])


    with open('outputs/GOF/' + grb + '_GOF.txt', 'w') as f:
        f.write(str(gof_null) + '\n')
        f.write(str(gof_lin) + '\n')
        f.write(str(gof_quad) + '\n')
        
    f.close()
    f = []


    
    print('#'*150)
    print('Done with ' + grb + '!')
    print('#'*150)