import sncosmo
import numpy as np
import pickle
import random
import argparse
# import estimate_explosion_time.sncosmo_register_ztf_bands
from estimate_explosion_time.core.fit_data.sncosmo import sncosmo_model_names as model_names_dict
from estimate_explosion_time.core.analyse_fits_from_simulation.get_source_explosion_time.find_explosion_time \
    import get_explosion_time



# set up parser
parser = argparse.ArgumentParser(description='Fits lightcurves using SNCosmo')
parser.add_argument('lc', type=int, help='Index of the lightcurve to be fitted')
parser.add_argument('infile', type=str, help='path to the input file')
parser.add_argument('outfile', type=str, help='name of the output file')
parser.add_argument('--method', type=str,
                    help='name of fitting routine to be used',
                    choices=['chi2', 'mcmc', 'nester'], default='chi2')
parser.add_argument('--sn_type', type=str,
                    help='SN type',
                    choices=['Ia', 'Ibc', 'IIP', 'IIn', 'IIpec', 'all'], default='Ibc')
parser.add_argument('--zfix', type=bool,
                    help='fix redshift in fit to known true value',
                    default=True)
args = parser.parse_args()

lc = args.lc-1
infile = args.infile
outname = args.outfile
method_name = args.method
sn_type = args.sn_type
zfix = args.zfix

# select model names corresponding to the used sn type
if sn_type is not 'all':
    model_names = model_names_dict[sn_type]

# if all, use all models
else:
    model_names = []
    for ls in model_names_dict.values(): model_names += ls

# keyword arguments to be passed to fit routine
kwargs = {}

# get the fit routine from sncosmo
if 'chi2' in method_name:
    fit_routine = sncosmo.fit_lc
elif 'mcmc' in method_name:
    fit_routine = sncosmo.mcmc_lc
elif 'nester' in method_name:
    fit_routine = sncosmo.nest_lc
    kwargs['guess_amplitude_bound'] = True
else:
    raise ValueError(f'Method {method_name} not known in SNCosmo!')

# set seed
random.seed(7281)

# parameters to vary in fit and bounds
vparam_names = ['z', 't0', 'amplitude']
nparam = len(vparam_names)
t0ind = np.array(vparam_names) == 't0'
zbound = {'z': (0, 1)}

# load data
with open(infile, 'rb') as fin:
    data = pickle.load(fin, encoding='latin1')
    lcs = data['lcs'][lc]
    t0_true = data['meta']['t0'][lc]
    ID = data['meta']['idx_orig'][lc]

    if zfix:
        z_true = data['meta']['z'][lc]
    else:
        z_true = None

# set up result arrays
nfit = len(model_names)
vparam_namese = [x+'_e' for x in vparam_names]

if zfix:
    vparam_names_used = ['t0', 'amplitude']
    bounds = {}
else:
    vparam_names_used = vparam_names
    bounds = zbound

arr_names = vparam_names + vparam_namese + \
            ['t_exp_fit',
             't_exp_true',
             't_exp_dif',
             'model',
             'ID',
             'ztrue',
             'zfix',
             'red_chi2',
             'nobs',
             'error']

formats = ['<f8'] * (2*nparam+3) + ['<U20', '<i8', '<f8', '?', '<f8', '<f8', '?']

outarr = np.zeros(nfit, dtype={'names': arr_names, 'formats': formats})

# loop over models and lightcurves and fixing z or not to execute the fits
for ind, model_name in enumerate(model_names):
    model = sncosmo.Model(source=model_name)

    if zfix:
        model.set(z=z_true)

    nparam_used = len(vparam_names_used)

    try:
        res, fitted_model = fit_routine(lcs, model, vparam_names_used, bounds=bounds, **kwargs)

        outarr[vparam_names][ind] = tuple(res.parameters)
        outarr['ztrue'][ind] = z_true
        outarr['zfix'][ind] = zfix
        outarr['t_exp_true'][ind] = t0_true
        outarr['model'][ind] = model_name
        outarr['ID'][ind] = ID
        outarr['nobs'][ind] = len(lcs)

        for j in vparam_names_used:
            outarr[j+'_e'][ind] = res.errors[j]

        tmp_chisq = sncosmo.chisq(lcs, fitted_model)/res.ndof
        outarr['red_chi2'][ind] = tmp_chisq

        t_exp_fitted = get_explosion_time(model_name) * \
                       (1 +
                        res.parameters[np.array(model.param_names) == 'z'] ) \
                       + res.parameters[np.array(model.param_names) == 't0']
        outarr['t_exp_fit'][ind] = t_exp_fitted
        outarr['t_exp_dif'][ind] = t0_true - t_exp_fitted

    except (ValueError, RuntimeError, KeyError, RuntimeWarning, sncosmo.fitting.DataQualityError) as err:
        print(err, ' for model ', model_name, ' and lightcurve ', str(ID), ' zfix is ', str(zfix))
        outarr['error'][ind] = True

# write to pickle file
with open(outname, 'wb') as fout:
    pickle.dump(outarr, fout)
