from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.shared import simulation_dir, all_methods, es_scratch_dir
from estimate_explosion_time.core.data_prep.data import DataHandler
from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import Fitter
from estimate_explosion_time.cluster import n_tasks
from estimate_explosion_time.analyses.dummy import DummyDH


methods = ['mosfit']
# simulation_name = 'dummy'
#
# simsurvey_path = f'{es_scratch_dir}/../dummy.pkl'

simsurveyDH = DummyDH.get_dhandler()

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

for method in methods:

    fitter = Fitter.get_fitter(method)

    logger.debug(
        f'fitter method {fitter.method_name} \n'
        f'job-id {fitter.job_id}'
    )

    fitter.fit_lcs(simsurveyDH, tasks_in_group=1)
