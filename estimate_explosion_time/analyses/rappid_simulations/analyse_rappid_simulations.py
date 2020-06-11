from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.shared import simulation_dir
from estimate_explosion_time.core.data_prep.data import DataHandler
from estimate_explosion_time.analyses.rappid_simulations import rappidDH
from estimate_explosion_time.cluster import n_tasks
from estimate_explosion_time.analyses.rappid_simulations.convert_to_pickle_files import \
    rappid_pkl_name, write_model_to_pickle, rappid_original_data
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--force_results', default=False, type=bool)
parser.add_argument('-e', '--force_explosion_time', default=False, type=bool)
args = parser.parse_args()

# for model_number in [3, 13]:
#     if not os.path.isfile(rappid_pkl_name(model_number)):
#         write_model_to_pickle(model_number)

band = 'ztfr'
sed_directory = rappid_original_data + '/SEDs'
generated_with = 'mosfit'
thisDH = rappidDH.get_dhandler(generated_with, sed_directory=sed_directory, peak_mag=19)

thisDH.get_explosion_times_from_template(ncpu=46,
                                         band=band,
                                         force_all=args.force_explosion_time,
                                         method='threshold',
                                         # flux_threshold=0.5
                                         )

dt = 10

thisDH.select_and_adjust_selection_string(
    # req_prepeak=4,
    # req_peak_mag=19,
    # req_std=None,
    # check_band='any',
    # req_texp_dif=['mosfit', dt]
)

thisDH.results('mosfit', force=args.force_results, n=20, dt=dt,
               # band=band
)
