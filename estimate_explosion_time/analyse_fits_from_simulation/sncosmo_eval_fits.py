import sncosmo
import numpy as np
import pickle
import sncosmo_xtract
import eval_tools

# read file containing results of the fit
with open('sncosmo_fit_out/sncosmo_fit_res_10.pkl','rb') as fres:
    fitres_list = pickle.load(fres)

chi_tab, mcmc_tab = sncosmo_xtract.get_tables('sncosmo_fit_out/sncosmo_fit_res_', [1, 10], '.pkl')

# read the fitted simulations
with open('../simsurvey-paper-scripts/lcs/lcs_Ibc_nugent_000000.pkl', 'rb') as fin:
    sim = pickle.load(fin, encoding='latin1')

model_name = 'nugent-sn1a'
tdif, tdif_e, ID = sncosmo_xtract.get_tdif_from_fit(chi_tab, model_name)
eval_tools.eval_dist(tdif, filename=model_name+'_texp_pred.pdf', xlabel='$\Delta t_{exp}$', title=model_name)
