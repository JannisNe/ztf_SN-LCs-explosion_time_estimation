from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods, pickle_dir
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.shared import simulation_dir, es_scratch_dir
from estimate_explosion_time.core.data_prep.data import DataHandler
from estimate_explosion_time.cluster import n_tasks
from estimate_explosion_time.analyses.dummy import DummyDH
from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import Fitter
import pickle
import numpy as np


band = 'ztfg'
logger.debug(f'create dummy data with peak band {band}')
DummyDH.create_dummy_data(peak_band=band)

dh = DummyDH.get_dhandler()
dh.get_explosion_times_from_template(ncpu=10, band=band)

# check whether explosion ime estimation works
with open(dh.get_data('sncosmo'), 'rb') as f:
    data = pickle.load(f, encoding='latin1')

meta = data['meta']
t0_true_true = np.array(meta['t0_true'])
t0_true_estimated = np.array(meta['t0'])

med_dif, ic_upper, ic_lower = np.quantile(t0_true_true - t0_true_estimated, [0.5, 0.05, 0.95])
input(f'median(tdif) = {med_dif:.2f} (- {med_dif-ic_lower:.2f} + {ic_upper-med_dif:.2f}). Continue? ')

# fit the lightcurves with the desired method (only 'mosfit' is good!)
method = 'mosfit'
fitter = Fitter.get_fitter(method)

logger.debug(
    f'fitter method {fitter.method_name} \n'
    f'job-id {fitter.job_id}'
)

# missing_indice_file = f'{pickle_dir}/{dh.name}/{fitter.method_name}/missing_indices.txt'
fitter.fit_lcs(dh, tasks_in_group=1,
               # missing_indice_file=missing_indice_file  # to be used when repeating the fit
               )

# make a selection of lightcurves based on the available photometry
dh.select_and_adjust_selection_string()

# get the results
dh.results('mosfit')
