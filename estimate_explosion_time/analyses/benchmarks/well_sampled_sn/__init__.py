import logging
from estimate_explosion_time.shared import get_custom_logger, main_logger_name, es_scratch_dir


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())


import os
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
from estimate_explosion_time.core.data_prep.data import DataHandler, DataImportError


raw_data_dir = es_scratch_dir + '/../benchmark_lcs'
lightcurve_dir = raw_data_dir + '/lcs'
meta_file = raw_data_dir + '/meta.csv'
pickle_filename = raw_data_dir + '/benchmark_lcs.pkl'


# def get_lc_


def get_lc_sn2008d():

    fn = f'{lightcurve_dir}/SN2008D.txt'

    # load the data into a pandas DataFrame
    dat = pd.read_csv(fn, delimiter='\t')

    # set up a new black DataFrame with the desired columns
    new = pd.DataFrame(columns=['mjd', 'filter', 'mag', 'emag', 'upper_limit', 'instrument'])

    # get the str that indicates that no data is present
    nan_proxy = dat['R (mag)'][0]

    # go through rows and select r and i band observation and put it into the ouput DataFrame
    for i, row in dat.iterrows():
        r = str(row['r" (mag)'])
        i = str(row['i" (mag)'])

        for band, b in zip(['r', 'i'], [r, i]):

            if b != nan_proxy and b != '-':
                if not '>' in b:
                    ul = False
                    mag = float(b.split('(')[0])
                    emag = float('0.' + b.split('(')[1].strip(')'))
                else:
                    ul = True
                    mag = float(b.split('>')[1])
                    emag = 1
                app = pd.DataFrame([(row['MJD (day)'], band, mag, emag, ul, 'FLWO')], columns=new.columns)
                new = new.append(app, ignore_index=True)

    # convert the DataFrame to np.array
    arr = new.to_numpy()
    new_arr = np.array([tuple(row) for row in arr],
                       dtype={'names': list(new.columns), 'formats': ['<f8', '<U15', '<f8', '<f8', '?', '<U15']})
    return new_arr


def make_pickle_file():

    # set up empty data dictionary
    data = {
        'lcs': [],
        'meta': {
            'z': [],
            'z_err': [],
            't_exp': []
            'idx_orig': [],
        }
    }

    # read the meta data from file
    meta_data = pd.read_csv(meta_file)

