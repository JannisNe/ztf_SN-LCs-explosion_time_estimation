from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.shared import simulation_dir
from estimate_explosion_time.core.data_prep.data import DataHandler
import pickle


methods = ['sncosmo_mcmc']
logger.debug(f'using methods{methods}')
simulation_name = 'simsurvey_simulation'

simsurvey_path = f'{simulation_dir}/{simulation_name}'

simsurveyDH = DataHandler.get_dhandler(simulation_name, simsurvey_path)

# simsurveyDH.select_and_adjust_selection_string(
#     req_prepeak=3,
#     req_postpeak=1,
#     req_max_timedif=50,
#     req_std=0.5,
#     check_band='all'
# )

input('continue? ')

for method in methods:

    logger.info(f'getting results for {simsurveyDH.name} analyzed by {method}')
    rhandler = simsurveyDH.rhandlers[method]

    try:
        rhandler.sub_collect_results()
        rhandler.get_t_exp_dif_distribution()

    finally:
        simsurveyDH.save_me()

    with open(simsurveyDH._sncosmo_data_, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    Ids_res = [res['fit_output']['ID'][0] for res in rhandler.collected_data[:10]]

    logger.info(f'IDs in data: \n'
                f'{data["meta"]["idx_orig"][:10]}')
    logger.info(f'IDs in results: \n'
                f'{Ids_res}')
