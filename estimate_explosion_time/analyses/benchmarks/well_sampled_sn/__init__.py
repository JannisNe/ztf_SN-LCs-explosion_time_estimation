import os
import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd
from estimate_explosion_time.core.data_prep.data import DataHandler, DataImportError
import logging
from estimate_explosion_time.shared import get_custom_logger, main_logger_name, es_scratch_dir


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
# logger.setLevel(logging.DEBUG)

raw_data_dir = es_scratch_dir + '/../benchmark_lcs'
lightcurve_dir = raw_data_dir + '/lcs'
meta_file = raw_data_dir + '/meta.csv'
pickle_filename = raw_data_dir + '/benchmark_lcs.pkl'

event_get_lc_fct = {}


def register_event(name):
    logger.debug(f'registering {name}')

    def dec(fct):
        event_get_lc_fct[name] = fct
        return fct

    return dec


@register_event('SN2008D')
def get_lc_sn2008d():

    fn = f'{lightcurve_dir}/SN2008D.txt'

    # load the data into a pandas DataFrame
    dat = pd.read_csv(fn, delimiter='\t')

    # set up a new blank DataFrame with the desired columns
    new = pd.DataFrame(columns=['mjd', 'band', 'mag', 'emag', 'upper_limit', 'instrument'])

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


class benchmarkDH(DataHandler):

    name = 'benchmark_lcs'

    def __init__(self):
        logger.debug('initialising')
        self.make_pickle_file()
        DataHandler.__init__(self, pickle_filename, benchmarkDH.name, simulation=False)
        self.save_me()

    @staticmethod
    def make_pickle_file():
        # set up empty data dictionary
        data = {
            'lcs': [],
            'meta': {
                'z': [],
                'z_err': [],
                't_exp': [],
                'idx_orig': [],
            }
        }

        # read the meta data from file
        meta_data = pd.read_csv(meta_file, delimiter=', ')
        logger.debug(f'meta data:\n{meta_data}')

        for event_name, lc_fct in event_get_lc_fct.items():
    
            logger.debug(f'adding data for {event_name} to list')

            # add lightcurve data to list
            data['lcs'].append(lc_fct())

            # load meta data
            meta_mask = meta_data.ID == event_name
            logger.debug(f'mask is \n{meta_mask}')

            meta_data_ind = np.where(meta_mask)[0]
            # raise Error if meta data is amigiuous
            if len(meta_data_ind) > 1:
                raise Exception(f'More than one entry found for {event_name}:\n{meta_data}')
            elif len(meta_data_ind) < 1:
                raise Exception(f'No info for {event_name}:\n{meta_data}')
            else:
                meta_data_ind = int(meta_data_ind)

            logger.debug(f'indice is {meta_data_ind}')

            # add to meta data list
            for key in data['meta']:
                logger.debug(f'key is {key}')
                data['meta'][key].append(meta_data[key if key != 'idx_orig' else 'ID'][meta_data_ind])

        # save as pickle
        logger.debug(f'saving under {pickle_filename}')
        with open(pickle_filename, 'wb+') as f:
            pickle.dump(data, f)

    @staticmethod
    def get_dhandler(**kwargs):
        if os.path.isfile(DataHandler.dh_path(benchmarkDH.name)):
            return DataHandler.load_dh(benchmarkDH.name)
        else:
            return benchmarkDH()