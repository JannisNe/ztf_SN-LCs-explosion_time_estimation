import logging
import os
import pickle
import numpy as np
from shutil import copy2
from tqdm import tqdm
from estimate_explosion_time.core.data_prep.data import DataHandler, DataImportError
from estimate_explosion_time.shared import get_custom_logger, main_logger_name, TqdmToLogger, es_scratch_dir
from estimate_explosion_time.analyses.rappid_simulations.convert_to_pickle_files import rappid_pkl_name


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
# tqdm_deb = TqdmToLogger(logger, level=logging.DEBUG)
# tqdm_info = TqdmToLogger(logger, level=logging.INFO)


class rappidDH(DataHandler):

    def __init__(self, generated_with, sed_directory):

        if 'mosfit' in generated_with:
            self.number = '13'
        elif 'template' in generated_with:
            self.number = '03'
        else:
            self.number = None
            DataImportError(f'Input {generated_with} for generated_with not understood')

        dh_name = 'rappid_sim_model' + self.number
        path = rappid_pkl_name(self.number)

        DataHandler.__init__(self, path, dh_name)

        self.sed_directory = self.dir + '/SEDs'

        if not os.path.isdir(self.sed_directory):
            logger.debug('making directory ' + self.sed_directory)
            os.mkdir(self.sed_directory)

        sed_directory += '/MODEL' + self.number

        itr = os.listdir(sed_directory)
        for file in itr if logger.getEffectiveLevel() > logging.INFO else tqdm(itr, desc='copying SED files'):

            if file.startswith('.'):
                continue

            copy2(f'{sed_directory}/{file}', self.sed_directory + '/')

        self.save_me()

    def get_spectral_time_evolution_file(self, indice):

        sed_directory = self.sed_directory if os.path.exists(self.sed_directory) \
            else es_scratch_dir + '/input' + self.sed_directory.split('input')[1]

        with open(self.get_data('sncosmo'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        model = data['meta']['model'][indice]

        if self.number == '13':
            return sed_directory + '/ibc_' + str(model) + '.dat.gz'

        if self.number == '03':
            sed_map = np.loadtxt(sed_directory + '/NON1A.LIST',
                                 dtype={'names': ['s', 'model_number', 'type', 'file'],
                                        'formats': ['<U15', '<f8', '<U15', '<U15']})
            file = sed_map['file'][sed_map['model_number'] == model][0]
            return sed_directory + '/' + str(file) + '.gz'


    @staticmethod
    def get_dhandler(generated_with, **kwargs):

        if 'mosfit' in generated_with:
            number = '13'
        elif 'template' in generated_with:
            number = '03'
        else:
            number = None
            DataImportError(f'Input {generated_with} for generated_with not understood')

        dh_name = 'rappid_sim_model' + number
        path = f'/afs/ifh.de/user/n/neckerja/scratch/ZTF_20190512/ZTF_MSIP_MODEL{number}.pkl'

        if os.path.isfile(DataHandler.dh_path(dh_name)):
            logger.info(f'DataHandler for {dh_name} already exists, loading it ...')
            return DataHandler.load_dh(dh_name)

        elif os.path.exists(path):
            logger.info(f'creating rappidDataHandler for {dh_name}')
            return rappidDH(generated_with, **kwargs)

        else:
            raise DataImportError(f'ResultHandler for {dh_name} doesn\'t exist. '
                                  f'Please sepcify path to the data!')
