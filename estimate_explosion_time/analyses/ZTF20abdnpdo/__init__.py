import logging
from estimate_explosion_time.shared import get_custom_logger, main_logger_name, es_scratch_dir


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())


import os
import numpy as np
from tqdm import tqdm
import pickle
from astropy.table import Table
from astropy.time import Time
import pandas as pd
from estimate_explosion_time.core.data_prep.data import DataHandler, DataImportError

name = 'ZTF20abdnpdo'


raw_data_dir = es_scratch_dir + f'/../{name}'
lightcurve_dir = raw_data_dir + '/csv'
meta_file = raw_data_dir + '/meta.csv'
pickle_filename = raw_data_dir + f'/{name}.pkl'

photometry_file = raw_data_dir + f'/{name}_phot.csv'
forced_photometry_file = raw_data_dir + f'/{name}_forced_phot.csv'
previous_upper_limits_file = raw_data_dir + f'/{name}_ul.csv'


def get_lightcurve(include_previous_limits=5):

    photometry = pd.read_csv(photometry_file)
    forced_photometry = pd.read_csv(forced_photometry_file)
    forced_photometry = forced_photometry.sort_values(by='obsmjd')
    mask = np.array([False] * len(forced_photometry))
    ind = np.where(forced_photometry['mag'] != 99.)[0][0]
    mask[ind] = True

    logger.debug(ind)
    logger.debug(f'dropping \n{forced_photometry[mask]}')

    forced_photometry = forced_photometry[~mask]

    earliest_photometry_date = min(forced_photometry['obsmjd']) - include_previous_limits

    new = pd.DataFrame(columns=['mjd', 'band', 'mag', 'emag', 'upper_limit', 'instrument', 'forced_phot'])

    for i, row in photometry.iterrows():

        mjd = Time(row['jdobs'], format='jd').mjd
        band = row['filter']
        mag = row['magpsf'] if row['magpsf'] != 99. else row['limmag']
        mag_err = row['sigmamagpsf'] if row['magpsf'] != 99. else 1
        upper_limit = True if row['magpsf'] == 99. else False
        instrument = row['instrument']

        app = pd.DataFrame([(mjd, band, mag, mag_err, upper_limit, instrument, False)], columns=new.columns)
        new = new.append(app, ignore_index=True)

    for i, row in forced_photometry.iterrows():

        mjd = row['obsmjd']
        band = row['filter'].split('ZTF_')[1]
        mag = row['mag'] if row['mag'] != 99. else row['upper_limit']
        mag_err = row['mag_err'] if row['mag_err'] != 99. else 1
        upper_limit = True if row['mag'] == 99. else False

        app = pd.DataFrame([(mjd, band, mag, mag_err, upper_limit, 'ZTF', True)], columns=new.columns)
        new = new.append(app, ignore_index=True)

    new = new.sort_values(by='mjd')

    arr = new.to_numpy()
    new_arr = np.array([tuple(row) for row in arr],
                       dtype={'names': list(new.columns), 'formats': ['<f8', '<U15', '<f8', '<f8', '?', '<U15', '?']})
    return new_arr


class ZTF20abdnpdoDataHandler(DataHandler):

    my_name = name

    def __init__(self):
        self.make_pickle_file()
        DataHandler.__init__(self, name=ZTF20abdnpdoDataHandler.my_name, path=pickle_filename, simulation=False)

    @staticmethod
    def make_pickle_file():

        data = {
            'lcs': [],
            'meta': {
                'z': [],
                'idx_orig': [],
                't0': [],
                'neutrino_time': []
            }
        }

        meta_data = pd.read_csv(meta_file, delimiter=', ')

        meta_data_ind = meta_data['ID'] == ZTF20abdnpdoDataHandler.my_name
        data['meta']['z'].append(float(meta_data['z'][meta_data_ind].values))
        data['meta']['idx_orig'].append(str(meta_data['ID'][meta_data_ind].values[0]))
        data['meta']['t0'].append(np.nan)
        ti = meta_data['neutrino_time'][meta_data_ind].values[0]
        logger.debug(type(ti))
        data['meta']['neutrino_time'].append(Time(ti).mjd)

        data['lcs'].append(get_lightcurve())

        logger.debug(f'meta data is {data["meta"]}')
        logger.debug(f'saving under {pickle_filename}')
        with open(pickle_filename, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def get_dhandler(**kwargs):
        dhpath = DataHandler.dh_path(ZTF20abdnpdoDataHandler.my_name)
        if kwargs.get('force_new', False):
            if os.path.isfile(dhpath):
                os.remove(dhpath)
        if os.path.isfile(dhpath):
            return DataHandler.load_dh(ZTF20abdnpdoDataHandler.my_name)
        else:
            return ZTF20abdnpdoDataHandler()
