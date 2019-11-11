import sncosmo
import numpy as np
import json
import pickle
from astropy.table import vstack

def texp_from_tmax(tmax, model_name, z):
    with open('sncosmo_texp.json', 'rb') as fin:
        texp_model = json.load(fin)
    texp = {}
    for name in model_name:
        texp[name] = texp_model[name] * (1+np.array(z)) + np.array(tmax)
    return texp


def get_tdif_from_fit(fitres, model_name, zfix=True):
    modelind = fitres['model'] == model_name
    zind = fitres['zfix'] == zfix
    t0_fit = np.array(fitres['t0'][modelind & zind])
    t0_fit_e = np.array(fitres['t0_e'][modelind & zind])
    t0_true = np.array(fitres['t0_true'][modelind & zind])
    ID = np.array(fitres['ID'][modelind & zind])
    return (t0_fit - t0_true, t0_fit_e, ID)


def get_tables(filename_base, ind , appendix):
    if appendix == '.pkl':
        chi_list = []
        mcmc_list = []
        for i in ind:
            with open(filename_base+str(i)+appendix, 'rb') as f:
                dat = pickle.load(f)
            chi_list += [dat[0]]
            mcmc_list += [dat[1]]
        chi_tab = vstack(chi_list)
        mcmc_tab = vstack(mcmc_list)
        return (chi_tab, mcmc_tab)
    else: raise NotImplementedError('data type not implemented')