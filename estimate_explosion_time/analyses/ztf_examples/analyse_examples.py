from estimate_explosion_time.shared import get_custom_logger, main_logger_name, pickle_dir
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.analyses.ztf_examples import convert_to_pickles, ExampleDH, pickle_filename
from estimate_explosion_time.core.data_prep.data import DataHandler
from estimate_explosion_time.analyses.rappid_simulations import rappidDH
from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import Fitter
from estimate_explosion_time.analyses.rappid_simulations.convert_to_pickle_files import \
    rappid_pkl_name, write_model_to_pickle, rappid_original_data
import os


# take the original data and convert it into pickles in the right format
# if not os.path.exists(pickle_filename):
#     convert_to_pickles()

convert_to_pickles()
# get the DataHandler object who takes care of all the book keeping
thisDH = ExampleDH.get_dhandler()
fitter = Fitter.get_fitter('mosfit')
thisDH.pickle_dir = fitter.get_output_directory(thisDH)
thisDH.use_method('mosfit', None)
thisDH.save_me()

# # fit the lightcurves with the desired method (only 'mosfit' is good!)
# method = 'mosfit'
# fitter = Fitter.get_fitter(method)
#
# logger.debug(
#     f'fitter method {fitter.method_name} \n'
#     f'job-id {fitter.job_id}'
# )
#
# missing_indice_file = f'{pickle_dir}/{thisDH.name}/{fitter.method_name}/missing_indices.txt'
# fitter.fit_lcs(rappidDH,
#                tasks_in_group=100,
#                # missing_indice_file=missing_indice_file  # to be used when repeating the fit
#                )
#
# # make a selection of lightcurves based on the available photometry
# thisDH.select_and_adjust_selection_string()
#
# # get the results
# thisDH.results('mosfit')
