from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import Fitter
from estimate_explosion_time.shared import simulation_dir, pickle_dir
from estimate_explosion_time.core.data_prep.data import DataHandler
from estimate_explosion_time.analyses.rappid_simulations import rappidDH
from estimate_explosion_time.cluster import n_tasks
from estimate_explosion_time.analyses.rappid_simulations.convert_to_pickle_files import \
    rappid_pkl_name, write_model_to_pickle, rappid_original_data
import os


sed_directory = rappid_original_data + '/SEDs'
methods = all_methods[:-1]
generated_with = 'mosfit'
rappidDH = rappidDH.get_dhandler(generated_with, sed_directory=sed_directory)

logger.debug(
    f'Name: {rappidDH.name} \n'
    f'ResultHandler: {rappidDH.rhandlers.keys()} \n'
    f'rhandlers: \n '
)

for rhandler in rappidDH.rhandlers.values():
    logger.debug(
        f'method {rhandler.method} \n'
        f'job ID {rhandler.job_id} \n'
        f'tasks left {n_tasks(rhandler.job_id)}'
    )

input('continue? ')

fitter = Fitter.get_fitter('mosfit')

logger.debug(
    f'fitter method {fitter.method_name} \n'
    f'job-id {fitter.job_id}'
)

missing_indice_file = f'{pickle_dir}/{rappidDH.name}/{fitter.method_name}/missing_indices.txt'
fitter.fit_lcs(rappidDH, tasks_in_group=100)
