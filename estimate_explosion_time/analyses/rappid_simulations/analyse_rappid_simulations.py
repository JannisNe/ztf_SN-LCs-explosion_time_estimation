from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.INFO)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.shared import simulation_dir
from estimate_explosion_time.core.data_prep.data import DataHandler
from estimate_explosion_time.analyses.rappid_simulations import rappidDH
from estimate_explosion_time.cluster import n_tasks
from estimate_explosion_time.analyses.rappid_simulations.convert_to_pickle_files import \
    rappid_pkl_name, write_model_to_pickle, rappid_original_data
import os


for model_number in [3, 13]:
    if not os.path.isfile(rappid_pkl_name(model_number)):
        write_model_to_pickle(model_number)

sed_directory = rappid_original_data + '/SEDs'
methods = all_methods[:-1]
generated_with = 'mosfit'
thisDH = rappidDH.get_dhandler(generated_with, sed_directory=sed_directory)
# thisDH.get_explosion_times_from_template(ncpu=25)
thisDH.select_and_adjust_selection_string()
thisDH.results('mosfit')