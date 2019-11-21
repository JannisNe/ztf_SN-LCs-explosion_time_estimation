from estimate_explosion_time.shared import simulation_dir, real_data_dir, dh_dict_dir
from estimate_explosion_time.core.analyse_fits_from_simulation.results import SNCosmoResultHandler, MosfitResultHandler
import os
from warnings import warn
from shutil import copytree, copy2
import logging
import pickle
from astropy.table import Table
from astropy.io import ascii


class DataHandler:

    def __init__(self, path, name, simulation=True):

        self.name = name
        self.orig_path = path
        self.data = None
        self.nlcs = None
        self.pickle_dir = None
        self.method = None
        self.dh_dict = None
        self.collected_data = None

        if simulation:
            diri = simulation_dir
        else:
            diri = real_data_dir

        diri += f'/{name}'
        logging.info(f'data directory will be {diri}')

        iadd = 2

        # if data has not been copied to input directory, do so
        if diri not in path:
            newdir1 = diri

            while os.path.isdir(newdir1):

                newdir2 = f'{diri}_{iadd}'

                warn('the directory \n'
                     f'{newdir1} \n'
                     f'already exists! Saving data to \n'
                     f'{newdir2}',
                     DataImportWarning)

                iadd +=1
                newdir1 = newdir2

            self.dir = newdir1

            logging.info("Making Directory: {0}".format(self.dir))
            os.makedirs(self.dir)

            # copy file(s)
            if os.path.isfile(path):
                ending = path.split('/')[-1].split('.')[-1]
                dst = f'{self.dir}/{self.name}.{ending}'
                logging.info(f'copying {path} to {dst}')
                copy2(path, dst)

            elif os.path.isdir(path):
                logging.info(f'copying data from {path} to {self.dir}')
                copytree(path, self.dir)

            else:
                raise DataImportError(
                    f'No data found in {path}'
                )

        else:

            if diri == path:
                self.dir = path

            else:
                raise DataImportError(
                    f'Data already imported, but to wrong directory!'
                )

        self._sncosmo_data_ = f'{self.dir}/{self.name}.pkl'
        self._mosfit_data_ = f'{self.dir}/{self.name}_csv'

    def write_pkl_to_csv(self):

        logging.info('converting .pkl to .csv\'s')

        add_columns = {
            'instrument': 'ZTF_camera',
            'telescope': 'ZTF',
            'name': 'arb',
            'reference': 'JannisNecker',
            'u_time': 'MJD',
            'redshift': 'arb',
            'ebv': 'arb',
            'ID': 'arb'
        }

        with open(self._sncosmo_data_, 'rb') as fin:

            data = pickle.load(fin, encoding='latin1')
            lcs = data['lcs']
            meta = data['meta']
            self.nlcs = len(lcs)

        if not os.path.exists(self._mosfit_data_):
            logging.info(f'making directory {self._mosfit_data_}')
            os.mkdir(self._mosfit_data_)

        else:
            raise DataImportError('Folder with CSV files already exists!')

        for ind in range(len(lcs)):

            lc = Table(lcs[ind])
            lc['band'][lc['band'] == 'desi'] = 'ztfi'
            fname = f'{self._mosfit_data_}/{ind + 1}.csv'

            for col in add_columns:

                if col not in lc.keys():

                    lc[col] = [add_columns[col]] * len(lc)
                    if col == 'name': lc[col] = [f'{ind}'] * len(lc)
                    if col == 'redshift': lc[col] = [meta['z'][ind]] * len(lc)
                    if col == 'ebv': lc[col] = [meta['hostebv'][ind]] * len(lc)
                    if col == 'ID': lc[col] = [int(meta['idx_orig'][ind])] * len(lc)
                    if col == 'lumdist': lc[col] = [meta['lumdist'][ind]] * len(lc)

                else:
                    raise IndexError(f'Column {col} already exists!')

            logging.debug(f'writing file {fname}')
            with open(fname, 'w') as fout:
                ascii.write(lc, fout)

    def use_method(self, method):

        logging.info(f'DataHandler for {self.name} configured to use {method}')
        self.method = method

        if 'sncosmo' in method:

            self.data = self._sncosmo_data_

            if not self.nlcs:
                with open(self.data, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                    self.nlcs = len(data['lcs'])

                logging.info(f'{self.nlcs} lightcurves found in data.')

        elif 'mosfit' in method:
            self.data = self._mosfit_data_

            if not os.path.isdir(self._mosfit_data_):
                self.write_pkl_to_csv()

            self.nlcs = len(os.listdir(self.data))

        logging.debug(f'using data stored in {self.data}')

    def save_dh_dict(self):

        dh_dict = {}
        for key in self.__dict__.keys():
            dh_dict[key] = self.__dict__[key]

        if not self.dh_dict:
            self.dh_dict = dh_dict_path(self.name, self.method)

        logging.debug(f'saving the DataHandler dictionary to {self.dh_dict}')

        with open(self.dh_dict, 'wb') as fout:
            pickle.dump(dh_dict, fout)

    def results(self):

        if 'sncosmo' in self.method:
            rhandler = SNCosmoResultHandler(self)
        elif 'mosfit' in self.method:
            rhandler = MosfitResultHandler(self)
        else:
            raise ValueError(f'method {self.method} not known')

        rhandler.collect_results()
        self.save_dh_dict()


def dh_dict_path(name, method):
    return f'{dh_dict_dir}/{name}_{method}.pkl'


def load_dh(name, method):

    name = dh_dict_path(name, method)
    with open(name, 'rb') as fin:
        dh_dict = pickle.load(fin)

    dhandler = DataHandler(dh_dict['orig_path'], dh_dict['name'])
    for k in dh_dict.keys():
        dhandler.__dict__[k] = dh_dict[k]

    return dhandler


class DataImportWarning(UserWarning):
    def __init__(self, msg):
        self.msg = msg


class DataImportError(Exception):
    def __init__(self, msg):
        self.msg = msg
