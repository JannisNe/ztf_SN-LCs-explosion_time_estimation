from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())

from estimate_explosion_time.core.data_prep.data import DataHandler, DataImportError
from estimate_explosion_time.shared import simulation_dir
import os


simsurvey_name = 'simsurvey_simulation'

simsurvey_path = f'{simulation_dir}/{simsurvey_name}'


class SimsurveyDH(DataHandler):

    name = simsurvey_name

    def __init__(self):
        input('I\'m about to create a new DataHandler for the Simsurvey simulations. Continue? ')
        DataHandler.__init__(self, path=simsurvey_path, name=SimsurveyDH.name, simulation=False)

    def get_spectral_time_evolution_file(self, ind):
        return 'nugent-sn1bc'

    @staticmethod
    def get_dhandler(**kwargs):

        if os.path.isfile(DataHandler.dh_path(SimsurveyDH.name)):
            logger.info(f'DataHandler for {SimsurveyDH.name} already exists, loading it ...')
            return DataHandler.load_dh(SimsurveyDH.name)

        elif os.path.exists(simsurvey_path):
            logger.info(f'creating DataHandler for {SimsurveyDH.name}')
            return SimsurveyDH()

        else:
            raise DataImportError(f'ResultHandler for {SimsurveyDH.name} doesn\'t exist. '
                                  f'Please specify path to the data!')
