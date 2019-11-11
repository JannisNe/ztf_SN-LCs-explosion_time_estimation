import sncosmo
import numpy as np
import pickle
from astropy.table import Table
import sncosmo_register_ztf_bands
import random
import argparse
import json
from tqdm import tqdm

# set up parser
parser = argparse.ArgumentParser(description='Fit simulated lightcurves')
parser.add_argument('lc', type=int, help='Index of the lightcurve to be fitted')
parser.add_argument('infile', type=str, help='path to the input file')
parser.add_argument('outfile', type=str, help='name of the output file')
args = parser.parse_args(); lc = args.lc-1; infile=args.infile; outname=args.outfile

# set seed
random.seed(7281)

# model names of models to be used
with open('/lustre/fs23/group/icecube/necker/software/sncosmo_model_names.json') as fin:
    model_names = json.load(fin)

# name of the pickle input file
# datafile_name = inpath+'/simsurvey-paper-scripts/lcs/lcs_Ibc_nugent_000000.pkl'
datafile_name = infile

#parameters to vary in fit and bounds
vparam_names = ['z', 't0', 'amplitude']; nparam = len(vparam_names)
t0ind = np.array(vparam_names) == 't0'
bounds={'z':(0, 1)}

#load data
with open(datafile_name, 'rb') as fin:
    data = pickle.load(fin, encoding='latin1')
    lcs = data['lcs'][lc]
    t0_true = data['meta']['t0'][lc]
    ID = data['meta']['idx_orig'][lc]
    z_true = data['meta']['z'][lc]


# set up result arrays
nfit = len(model_names) *2
vparam_namese = [x+'_e' for x in vparam_names]
arr_names = vparam_names + vparam_namese + ['t0_true', 'model', 'ID', 'ztrue','zfix','red_chi2','nobs','error']
formats = ['<f8'] * (2*nparam+1) + ['<U15', '<i8', '<f8','?','<f8','<f8','?']
methods = [sncosmo.fit_lc, sncosmo.mcmc_lc]
arrarr = [[]]*len(methods)
for i in range(len(methods)): arrarr[i] = np.zeros(nfit, dtype={'names':arr_names, 'formats':formats})

# loop over models and lightcurves and fixing z or not to execute the fits
for modeln, model_name in enumerate(tqdm(model_names, desc='fit models')):
    model = sncosmo.Model(source = model_name)
    for zfix in [1]:
        ind = 2*modeln + zfix
        if zfix==1:
            vparam_names_used = ['t0', 'amplitude']
            bounds_used = {}
            model.set(z=z_true)
        else: vparam_names_used = vparam_names; bounds_used=bounds
        nparam_used = len(vparam_names_used)
        for nm, method in enumerate(methods):
            outarr = arrarr[nm]
            # if method == sncosmo.nest_lc: bounds_used['amplitude'] = [0, 1]
            try:
                res, fitted_model   = method(lcs, model, vparam_names_used, bounds=bounds_used)
                outarr[vparam_names][ind] = tuple(res.parameters)
                outarr[['ztrue','zfix']][ind] = tuple([z_true, zfix==1])
                outarr['t0_true'][ind] = t0_true
                outarr['model'][ind] = model_name
                outarr['ID'][ind] = ID
                outarr['nobs'][ind] = len(lcs)
                for j in range(nparam_used):
                    outarr[vparam_namese[j+zfix]][ind] = res.errors[vparam_names_used[j]]
                tmp_chisq = sncosmo.chisq(lcs, fitted_model)/res.ndof
                outarr['red_chi2'][ind] = tmp_chisq
            except (ValueError, RuntimeError, KeyError, RuntimeWarning, sncosmo.fitting.DataQualityError) as err:
                print(err, ' for model ', model_name, ' and lightcurve ', str(ID), ' zfix is ', str(zfix))
                outarr['error'][ind] = True

# write to pickle file
out = arrarr
with open(outname,'wb') as fout:
    pickle.dump(out, fout)
