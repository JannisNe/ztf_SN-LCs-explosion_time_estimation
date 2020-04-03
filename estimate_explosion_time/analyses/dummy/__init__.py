from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods, es_scratch_dir
import logging

logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())

from estimate_explosion_time.core.data_prep.data import DataHandler, DataImportError
from estimate_explosion_time.shared import simulation_dir
from estimate_explosion_time.analyses.simsurvey_simulations import SimsurveyDH
from estimate_explosion_time.core.analyse_fits_from_simulation.get_source_explosion_time.find_explosion_time \
    import bolometric_bandpass
import os
import numpy as np
import pickle
import sncosmo


# simsurvey_name = 'simsurvey_simulation'
#
# simsurvey_path = f'{simulation_dir}/{simsurvey_name}'

simulation_name = 'dummy'

dummy_path = f'{es_scratch_dir}/../dummy.pkl'


class DummyDH(DataHandler):

    name = simulation_name

    def __init__(self):
        input('I\'m about to create a new DataHandler for the Dummy simulations. Continue? ')
        DataHandler.__init__(self, path=dummy_path, name=DummyDH.name, simulation=True)

    def get_spectral_time_evolution_file(self, ind):
        return 'nugent-sn1bc'

    @staticmethod
    def get_dhandler(**kwargs):

        if os.path.isfile(DataHandler.dh_path(DummyDH.name)):
            logger.info(f'DataHandler for {DummyDH.name} already exists, loading it ...')
            return DataHandler.load_dh(DummyDH.name)

        elif os.path.exists(dummy_path):
            logger.info(f'creating DataHandler for {DummyDH.name}')
            return DummyDH()

        else:
            raise DataImportError(f'ResultHandler for {DummyDH.name} doesn\'t exist. '
                                  f'Please specify path to the data!')

    @staticmethod
    def create_dummy_data(seed=123, peak_band='ztfg'):

        logger.debug('create dummy data')
        simsurvey_dh = SimsurveyDH.get_dhandler()
        simsurvey_filename = simsurvey_dh.get_data('sncosmo')
        with open(simsurvey_filename, 'rb') as f:
            simsurvey_data = pickle.load(f, encoding='latin1')
        lcs = simsurvey_data['lcs']
        meta = simsurvey_data['meta']

        np.random.seed(seed)
        indices = list()
        while len(indices) < 10:
            ind = np.random.randint(0, simsurvey_dh.nlcs)
            if ind not in indices:
                indices.append(ind)
        logger.debug(f'only getting {len(indices)} LCs from simsurvey simulation')

        lcs_dummy = [lcs[i] for i in indices]
        meta_dummy = {key: [meta[key][i] for i in indices] for key in meta}
        t_peak = list()

        logger.debug('get the true peak for the lightcurves')

        model = sncosmo.Model('nugent-sn1bc')

        for i, lc in enumerate(lcs_dummy):

            model.set(z=meta_dummy['z'][i], t0=meta_dummy['t0'][i])
            time = np.linspace(model.mintime(), model.maxtime(), 5000)
            bandflux = model.bandflux(peak_band if peak_band else bolometric_bandpass(model), time)
            t_peak.append(time[np.argmax(bandflux)])

        meta_dummy['t_peak'] = np.array(t_peak)

        logger.debug('pretend explosion time was not know, e.g. change it to t0_true')
        meta_dummy['t0_true'] = meta_dummy['t0']
        meta_dummy['t0'] = [None] * len(lcs_dummy)

        dummy_dict = {'lcs': lcs_dummy, 'meta': meta_dummy}

        logger.debug(f'saving dummy data under {dummy_path}')
        with open(dummy_path, 'wb') as f:
            pickle.dump(dummy_dict, f)
