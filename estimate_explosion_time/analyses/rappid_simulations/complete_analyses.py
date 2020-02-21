from estimate_explosion_time.shared import get_custom_logger, main_logger_name, pickle_dir
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.INFO)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.analyses.rappid_simulations import rappidDH
from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import Fitter
from estimate_explosion_time.analyses.rappid_simulations.convert_to_pickle_files import \
    rappid_pkl_name, write_model_to_pickle, rappid_original_data
import os


# take the original simulated data and convert it into pickles in the right format
# the path to the original data is to be specified in convert_to_pickle_files.py
for model_number in [3, 13]:
    if not os.path.isfile(rappid_pkl_name(model_number)):
        write_model_to_pickle(model_number)

# sepcify where to look for the SED files that were used in the simulation.
# That's necesarry for getting the explosion time from the template
sed_directory = rappid_original_data + '/SEDs'

# get the lightcurves either generated using MOSFiT type 'mosfit'
# or using specral templates type 'templates'
generated_with = 'mosfit'

# get the DataHandler object who takes care of all the book keeping
thisDH = rappidDH.get_dhandler(generated_with, sed_directory=sed_directory)

# get the explosion times for the simulations
thisDH.get_explosion_times_from_template(ncpu=25)

# fit the lightcurves with the desired method (only 'mosfit' is good!)
method = 'mosfit'
fitter = Fitter.get_fitter(method)

logger.debug(
    f'fitter method {fitter.method_name} \n'
    f'job-id {fitter.job_id}'
)

missing_indice_file = f'{pickle_dir}/{thisDH.name}/{fitter.method_name}/missing_indices.txt'
fitter.fit_lcs(rappidDH, tasks_in_group=100,
               # missing_indice_file=missing_indice_file  # to be used when repeating the fit
               )

# make a selection of lightcurves based on the available photometry
thisDH.select_and_adjust_selection_string()

# get the results
thisDH.results('mosfit')