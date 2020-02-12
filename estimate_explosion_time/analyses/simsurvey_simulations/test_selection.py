from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.shared import simulation_dir
from estimate_explosion_time.core.data_prep.data import DataHandler


methods = ['mosfit']
logger.debug(f'using methods{methods}')
simulation_name = 'simsurvey_simulation'

simsurvey_path = f'{simulation_dir}/{simulation_name}'

simsurveyDH = DataHandler.get_dhandler(simulation_name, simsurvey_path)

dt = 10
simsurveyDH.select_and_adjust_selection_string(
    req_prepeak=2,
    req_std=None,
    check_band='any',
    req_texp_dif=['mosfit', dt]
)



for method in methods:
    simsurveyDH.results(method, dt=dt, n=20)
