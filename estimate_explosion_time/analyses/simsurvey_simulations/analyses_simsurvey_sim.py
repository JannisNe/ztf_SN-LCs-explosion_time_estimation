from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.shared import simulation_dir, plots_dir
from estimate_explosion_time.core.data_prep.data import DataHandler, load_dh
from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import Fitter


methods = all_methods
simulation_name = 'simsurvey_simulation'

simsurvey_path = f'{simulation_dir}/{simulation_name}'

simsurveyDH = DataHandler(simsurvey_path, simulation_name)

for method in all_methods:

    fitter = Fitter(method)

    fitter.fit_lcs(simsurveyDH)

# simsurveyDH = load_dh(simulation_name)

for method in all_methods:

    simsurveyDH.results(method)

# simsurveyDH.rhandlers[method].sub_collect_results()
# simsurveyDH.results('mosfit')
