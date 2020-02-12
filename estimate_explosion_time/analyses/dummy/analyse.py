from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.shared import simulation_dir, es_scratch_dir
from estimate_explosion_time.core.data_prep.data import DataHandler
from estimate_explosion_time.cluster import n_tasks


methods = ['mosfit']
simulation_name = 'dummy'

simsurvey_path = f'{es_scratch_dir}/../dummy.pkl'

dh = DataHandler.get_dhandler(simulation_name, simsurvey_path)
dh.select_and_adjust_selection_string(
    req_prepeak=2,
    req_std=None,
    check_band='any',
    req_texp_dif=['mosfit', 10],
    print_selected_indices=True
)

logger.debug(
    f'Name: {dh.name} \n'
    f'ResultHandler: {dh.rhandlers.keys()} \n'
    f'rhandlers: \n '
)

for rhandler in dh.rhandlers.values():
    logger.debug(
        f'method {rhandler.method} \n'
        f'job ID {rhandler.job_id} \n'
        f'tasks left {n_tasks(rhandler.job_id)}'
    )

logger.debug('selection: ' + dh.selection_string)

input('continue? ')

for method in methods:

    dh.results(method, dt=1)
