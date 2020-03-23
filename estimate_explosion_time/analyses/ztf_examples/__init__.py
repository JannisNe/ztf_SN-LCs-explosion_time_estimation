import logging
from estimate_explosion_time.shared import get_custom_logger, main_logger_name, es_scratch_dir


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())


import os
import numpy as np
from tqdm import tqdm
import pickle
from estimate_explosion_time.core.data_prep.data import DataHandler, DataImportError


raw_data_dir = es_scratch_dir + '/../ZTF_examples'
lightcurve_dir = raw_data_dir + '/csv'
meta_file = raw_data_dir + '/meta.csv'
pickle_filename = raw_data_dir + '/ZTF_examples.pkl'


def convert_to_pickles():

    data = {
        'lcs': [],
        'meta':{
            'z': [],
            'idx_orig': [],
        }
    }

    meta_raw = np.genfromtxt(
        meta_file,
        delimiter=',', skip_header=1, dtype={'names': ['id', 'z'], 'formats': ['<U15', '<f8']}
    )
    logger.debug(f'photometry located in {lightcurve_dir}')

    listed_lightcurve_dir =[_ for _ in os.listdir(lightcurve_dir) if _.endswith('.csv') and not _.startswith('.')]

    # logger.debug(f'loading {listed_lightcurve_dir}')

    for i, file in enumerate(tqdm(listed_lightcurve_dir, desc='loading photometry') \
            if logger.getEffectiveLevel() <= logging.INFO else listed_lightcurve_dir):

        lc = np.genfromtxt(lightcurve_dir + '/' + file, skip_header=1, delimiter=',',
                           dtype={
                               'names': ['jd', 'band', 'mag', 'emag', 'limmag', 'instrument', 'isdiffpos'],
                               'formats': ['<f8', '<U15', '<f8', '<f8', '<f8', '<U15', '?']
                           },
                           usecols=(1, 2, 4, 5, 6, 7, 12)
                           )

        lc = lc[lc['isdiffpos']]
        logger.debug(f'shape of lightcurve is {np.shape(lc)}')
        logger.debug(f'original lc data:\n {lc}')

        out_columns = ['jd', 'band', 'mag', 'emag', 'upper_limit', 'instrument']
        mag_comb_lc = np.empty(len(lc),
                               dtype = {
                                   'names': out_columns,
                                   'formats': ['<f8', '<U15', '<f8', '<f8', '?', '<U15']
                               })

        logger.debug(f'empty array for new lc:\n {mag_comb_lc}')

        for col in out_columns:
            try:
                mag_comb_lc[col] = lc[col]
            except ValueError:
                mag_comb_lc['upper_limit'] = [False] * len(lc)

        logger.debug(f'after inserting values: \n{mag_comb_lc}')

        comb_mask = lc['mag'] == 99
        mag_comb_lc['mag'][comb_mask] = lc[comb_mask]['limmag']
        mag_comb_lc['emag'][comb_mask] = [1] * len(np.where(comb_mask)[0])
        mag_comb_lc['upper_limit'][comb_mask] = [True] * len(np.where(comb_mask)[0])

        logger.debug(f'after replacing error mag values: \n{mag_comb_lc}')
        input('continue? ')

        data['lcs'].append(mag_comb_lc)

        id = file.split('.')[0]
        z = float(meta_raw['z'][np.array(meta_raw['id']) == id])

        data['meta']['idx_orig'].append(id)
        data['meta']['z'].append(z)

    with open(pickle_filename, 'wb') as f:
        pickle.dump(data, f)


class ExampleDH(DataHandler):

    name = 'ztf_example'

    def __init__(self):
        DataHandler.__init__(self, path=pickle_filename, name=ExampleDH.name, simulation=False)

    @staticmethod
    def get_dhandler(**kwargs):

        if os.path.isfile(DataHandler.dh_path(ExampleDH.name)):
            logger.info(f'DataHandler for {ExampleDH.name} already exists, loading it ...')
            return DataHandler.load_dh(ExampleDH.name)

        elif os.path.exists(pickle_filename):
            logger.info(f'creating rappidDataHandler for {ExampleDH.name}')
            return ExampleDH()

        else:
            raise DataImportError(f'ResultHandler for {ExampleDH.name} doesn\'t exist. '
                                  f'Please sepcify path to the data!')
