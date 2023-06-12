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
import pandas as pd
from scipy.stats import gaussian_kde

grbparam = pd.read_csv('GRBPARAM.csv', index_col=0)

param_ranges_NULL = [(1e-20, 5000), (-3, 10), (-10, 3), (0, 3), (0, 4)]
param_range_lin = [(1e-20, 1e20), (1e-20, 5000), (-3, 10), (-10, 3), (0, 3), (0, 4)]
param_range_quad = [(1e-20, 1e15), (1e-20, 5000), (-3, 10), (-10, 3), (0, 3), (0, 4)]

ncpu = mul.cpu_count()
GRBs = ['GRB210619B']#, 'GRB210610B', 'GRB210204A', 'GRB201216C', 'GRB200829A', 'GRB200613A', 'GRB190114C', 'GRB180720B', 'GRB180703A', 'GRB171010A', 'GRB160625B', 'GRB160509A', 'GRB150821A', 'GRB150514A', 'GRB150403A', 'GRB150314A'] 
# GRBs = [ 'GRB141028A', 'GRB140508A', 'GRB140206A', 'GRB131231A', 'GRB131108A', 'GRB130925A', 'GRB130518A', 'GRB130427A', 'GRB120119A', 'GRB100728A', 'GRB091003A', 'GRB090926A', 'GRB090618', 'GRB090328', 'GRB081221', 'GRB080916C']


for grb in GRBs:
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
    nburnin = 1000 #burn-in
    H0=70 #km/s/Mpc Taken from Sir's Code



    # plt.plot(arr[:,0], arr[:,1], '*')
    plt.errorbar(arr[:,0], arr[:,1], yerr=arr[:,2], fmt='*')
    plt.xlabel('E (keV)')
    plt.xscale('log')
    plt.ylim(-2, 3)
    plt.show()

    def MODEL_delta_t_intrinsic(E, Eb, alpha1, alpha2, mu, zeta):
        
        E0b = (E - E0)/Eb
        return zeta * (E0b ** alpha1) * (1 + E0b ** ((alpha2 - alpha1)*mu))/2


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
        
    def linear(E, z, logE_qg, Eb, alpha1, alpha2, mu, zeta):
        K_z= np.asarray(K_z1)
        # return (1+z)*tau*(Erest - E0rest) + (-(10**14)/(H0*3.24))*((Erest - E0rest)*K_z/(E_qg*(1+z)))
        return MODEL_delta_t_intrinsic(E, Eb, alpha1, alpha2, mu, zeta) + (-(10**14)/(H0*3.24))*((E - E0)*K_z/((10**logE_qg)*(1+z)))

    def quadratic(E, z, logE_qg, Eb, alpha1, alpha2, mu, zeta):
        E_0=E0rest/(1+z)
        Eres=E/(1+z)
        K_z = np.asarray(K_z2)
        # return (1+z)*tau*(Erest**2 - E0rest**2) + (-1.5*(10**8)/(H0*3.24))*((E**2 - E_0**2)*K_z/E_qg**2)
        return MODEL_delta_t_intrinsic(E, Eb, alpha1, alpha2, mu, zeta) + (-1.5*(10**8)/(H0*3.24))*((Eres**2 - E_0**2)*K_z/(10**logE_qg)**2)

    def loglklhood_null_HP(theta):
        Eb, alpha1, alpha2, mu, zeta = theta
        
        if alpha1 >= alpha2:
            model = MODEL_delta_t_intrinsic(x, Eb, alpha1, alpha2, mu, zeta)
            return sum(stats.norm.logpdf(*args) for args in zip(y,model,yerr))
        return -np.inf

    def loglklhood_LIV_lin(theta1):
        
        logE_qg, Eb, alpha1, alpha2, mu, zeta = theta1
        
        if alpha1 >= alpha2:
            model = linear(x, z_com, logE_qg, Eb, alpha1, alpha2, mu, zeta)    
            return sum(stats.norm.logpdf(*rgs) for rgs in zip(y,model,yerr))
        return -np.inf

    def loglklhood_LIV_quad(theta2):
        
        logE_qg, Eb, alpha1, alpha2, mu, zeta = theta2
        if alpha1 >= alpha2:
            model = linear(x, z_com, logE_qg, Eb, alpha1, alpha2, mu, zeta)    
            return sum(stats.norm.logpdf(*rgs) for rgs in zip(y,model,yerr))
        return -np.inf

    def prior_transform(theta):
        Eb, alpha1, alpha2, mu, zeta = theta

        # log_Eb_max = np.log10(5000) + 3
        Eb_max = 5000 #* 1e3
        alpha1_min, alpha1_max = -3.0, 10.0
        alpha2_min, alpha2_max = -10.0, 3.0
        mu_max = 3.0
        zeta_max = 4.0

        # log_Eb = log_Eb_max * log_Eb
        Eb = Eb_max * Eb
        alpha1 = alpha1_min + (alpha1_max - alpha1_min) * alpha1
        alpha2 = alpha2_min + (alpha2_max - alpha2_min) * alpha2
        mu = mu_max * mu
        zeta = zeta_max * zeta
        
        return [Eb, alpha1, alpha2, mu, zeta]


    def prior_transform_LIV_lin(theta1):
        log_E_qg, Eb, alpha1, alpha2, mu, zeta = theta1

        Eb_max = 5000 #* 1e3
        log_E_qg_max = np.log10(1e20) + 9
        # log_Eb_max = np.log10(5000)  + 3
        alpha1_min, alpha1_max = -3.0, 10.0
        alpha2_min, alpha2_max = -10.0, 3.0
        mu_max = 3.0
        zeta_max = 4.0

        
        log_E_qg = log_E_qg_max * log_E_qg
        # log_Eb = log_Eb_max * log_Eb
        Eb = Eb_max * Eb
        alpha1 = alpha1_min + (alpha1_max - alpha1_min) * alpha1
        alpha2 = alpha2_min + (alpha2_max - alpha2_min) * alpha2
        mu = mu_max * mu
        zeta = zeta_max * zeta

        return [log_E_qg, Eb, alpha1, alpha2, mu, zeta]


    def prior_transform_LIV_quad(theta2):
        log_E_qg, Eb, alpha1, alpha2, mu, zeta = theta2

        log_E_qg_max = np.log10(1e15) + 9
        Eb_max = 5000 #* 1e3
        # log_Eb_max = np.log10(5000) + 3  
        alpha1_min, alpha1_max = -3.0, 10.0
        alpha2_min, alpha2_max = -10.0, 3.0
        mu_max = 3.0
        zeta_max = 4.0

        
        log_E_qg = log_E_qg_max * log_E_qg
        # log_Eb = log_Eb_max * log_Eb
        Eb = Eb_max * Eb
        alpha1 = alpha1_min + (alpha1_max - alpha1_min) * alpha1
        alpha2 = alpha2_min + (alpha2_max - alpha2_min) * alpha2
        mu = mu_max * mu
        zeta = zeta_max * zeta

        return [log_E_qg, Eb, alpha1, alpha2, mu, zeta]



    nlive = 1024
    ndim_NULL = 5
    ndim_LIV = 6

    # NULL hypothesis

    with dyn.pool.Pool(ncpu, loglklhood_null_HP, prior_transform) as Pool:
        sampler0 = dyn.NestedSampler(loglklhood_null_HP, prior_transform, ndim=ndim_NULL, bound='multi', sample='rwalk', pool=Pool, nlive=nlive)
        sampler0.run_nested( dlogz=0.1)
    results0 = sampler0.results
        
    
    print(results0.summary())



    with open(os.getcwd() + f'/pickle/{grbname_wtht_ext}_results_null_logE_qg_linEb.pkl', 'wb') as f:
        pickle.dump(results0, f)

    with open(os.getcwd() + f'/pickle/{grbname_wtht_ext}_sampler_null_logE_qg_linEb.pkl', 'wb') as f:
        pickle.dump(sampler0, f)

    # Linear LIV
    with dyn.pool.Pool(ncpu, loglklhood_LIV_lin, prior_transform_LIV_lin) as Pool:
        sampler1 = dyn.NestedSampler(loglklhood_LIV_lin, prior_transform_LIV_lin, ndim=ndim_LIV, bound='multi', sample='rwalk', pool=Pool, nlive=nlive)
        sampler1.run_nested( dlogz=0.1)
    results1 = sampler1.results

    
    print(results1.summary())


    with open(os.getcwd() + f'/pickle/{grbname_wtht_ext}_results_LIV_lin_logE_qg_linEb.pkl', 'wb') as f:
        pickle.dump(results1, f)

    with open(os.getcwd() + f'/pickle/{grbname_wtht_ext}_sampler_LIV_lin_logE_qg_linEb.pkl', 'wb') as f:
        pickle.dump(sampler1, f)

    # Quadratic LIV
    with dyn.pool.Pool(ncpu, loglklhood_LIV_quad, prior_transform_LIV_quad) as Pool:
        sampler2 = dyn.NestedSampler(loglklhood_LIV_quad, prior_transform_LIV_quad, ndim=ndim_LIV, bound='multi', sample='rwalk', pool=Pool, nlive=nlive)
        sampler2.run_nested( dlogz=0.1)
    results2 = sampler2.results
    
    
    print(results2.summary())

    with open(os.getcwd() + f'/pickle/{grbname_wtht_ext}_results_LIV_quad_logE_qg_linEb.pkl', 'wb') as f:
        pickle.dump(results2, f)
        

    with open(os.getcwd() + f'/pickle/{grbname_wtht_ext}_sampler_LIV_quad_logE_qg_linEb.pkl', 'wb') as f:
        pickle.dump(sampler2, f)

    def smooth_plot(results, figname, labels=["logE_qg", "Eb(keV)", "alpha1", "alpha2", "mu", "zeta"]):
        weights = np.exp(results.logwt - results.logz[-1])
        samples = dyn.utils.resample_equal(  results.samples, weights)
        
        fig = corner(samples, weights=weights, labels=labels, levels=[0.68, 0.9], show_titles=True, title_kwargs={"fontsize": 12}, hist_kwargs={'density': True})
        ndim =samples.shape[1]
        for axidx, samps in zip([i*(ndim+1) for i in range(ndim)],samples.T):
            kde = gaussian_kde(samps)
            xvals = fig.axes[axidx].get_xlim()
            xvals = np.linspace(xvals[0], xvals[1], 100)
            fig.axes[axidx].plot(xvals, kde(xvals), color='firebrick')
        plt.savefig(os.getcwd() + '/outputs/' + figname + '.png')
        plt.show()

    smooth_plot(results0, figname=grbname_wtht_ext + 'null_HPlogE_qg_linEb', labels=["Eb(keV)", "alpha1", "alpha2", "mu", "zeta"])

    smooth_plot(results1, figname=grbname_wtht_ext + 'LIV_lin' + '_smoothlogE_qg_linEb')

    smooth_plot(results2, figname=grbname_wtht_ext + 'LIV_quad' + '_smoothlogE_qg_linEb')

    # Eb, alpha1, alpha2, mu, zeta = theta2
    nplot = 100
    E = np.linspace(min(Erest), max(Erest), nplot)
    samples0 = dyn.utils.resample_equal( results0.samples, np.exp(results0.logwt - results0.logz[-1]))
    samples0 = np.median(samples0, axis=0)

    samples1 = dyn.utils.resample_equal( results1.samples, np.exp(results1.logwt - results1.logz[-1]))
    samples1 = np.median(samples1, axis=0)

    samples2 = dyn.utils.resample_equal( results2.samples, np.exp(results2.logwt - results2.logz[-1]))
    samples2 = np.median(samples2, axis=0)

    null_fit = [MODEL_delta_t_intrinsic(E[i], samples0[0], samples0[1], samples0[2], samples0[3], samples0[4]) for i in range(nplot)]
    liv_lin_fit = [linear(E[i], z_com, samples1[0], samples1[1], samples1[2], samples1[3], samples1[4], samples1[5]) for i in range(nplot)]
    liv_quad_fit = [quadratic(E[i], z_com, samples1[0], samples1[1], samples1[2], samples1[3], samples1[4], samples1[5]) for i in range(nplot)]

    plt.errorbar(Erest, y, yerr, fmt='o', color='black', label='data')
    plt.plot(E, null_fit, label='Null fit')
    plt.plot(E, liv_lin_fit,label='Linear fit')
    plt.plot(E, liv_quad_fit, label='Quadratic fit')
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.xlabel('E (keV)')
    plt.ylabel('lag (s)')
    plt.title(grbname_wtht_ext)
    plt.savefig(os.getcwd() + '/outputs/' + grbname_wtht_ext + '_fit_logE.png', facecolor='white')
    plt.show()

    bayes_factor_lin = np.exp(results1.logz[-1] - results0.logz[-1])
    bayes_factor_quad = np.exp(results2.logz[-1] - results0.logz[-1])

    print('Bayes factor for linear LIV model: ', bayes_factor_lin)
    print('Bayes factor for quadratic LIV model: ', bayes_factor_quad)

    # with open(os.getcwd() + f'/outputs/BF/{grbname_wtht_ext}_BF_logE_qg_linEb.txt', 'w') as f:
    #         # f.write(f'Bayes factor for linear LIV model: {bayes_factor_lin}')
    #         # f.write(f'Bayes factor for quadratic LIV model: {bayes_factor_quad}')
    #         f.write(str([bayes_factor_lin, bayes_factor_quad]))