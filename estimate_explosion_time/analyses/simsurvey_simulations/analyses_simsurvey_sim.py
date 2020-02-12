from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

import argparse
from estimate_explosion_time.shared import simulation_dir
from estimate_explosion_time.core.data_prep.data import DataHandler
from estimate_explosion_time.cluster import n_tasks


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--force', type=bool)
args=parser.parse_args()

methods = ['mosfit']
simulation_name = 'simsurvey_simulation'

simsurvey_path = f'{simulation_dir}/{simulation_name}'

simsurveyDH = DataHandler.get_dhandler(simulation_name, simsurvey_path)

logger.debug(
    f'Name: {simsurveyDH.name} \n'
    f'ResultHandler: {simsurveyDH.rhandlers.keys()} \n'
    f'rhandlers: \n '
)

for rhandler in simsurveyDH.rhandlers.values():
    logger.debug(
        f'method {rhandler.method} \n'
        f'job ID {rhandler.job_id} \n'
        f'tasks left {n_tasks(rhandler.job_id)}'
    )

input('continue? ')

simsurveyDH.select_and_adjust_selection_string()

for method in methods:

    simsurveyDH.results(method, force=args.force)
