import os
from warnings import warn
from shutil import copytree, copy2
import logging
import pickle
from astropy.table import Table
from astropy.io import ascii
import numpy as np
from estimate_explosion_time.shared import simulation_dir, real_data_dir, dh_dict_dir,\
    get_custom_logger, main_logger_name
from estimate_explosion_time.core.analyse_fits_from_simulation.results import SNCosmoResultHandler, MosfitResultHandler
from estimate_explosion_time.core.analyse_fits_from_simulation.plots import Plotter
from estimate_explosion_time.cluster import wait_for_cluster


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
logger.debug('logging in data.py is also debug')


class DataHandler:

    def __init__(self, path, name, simulation=True):

        self.name = name
        self.orig_path = path
        self.data = None
        self.nlcs = None
        self.pickle_dir = None
        self.latest_method = None
        self.dh_dict = None
        self.save_path = dh_path(self.name)
        self.collected_data = None
        self.rhandlers = {}
        self.plotter = Plotter(self)

        if simulation:
            diri = simulation_dir
        else:
            diri = real_data_dir

        diri += f'/{name}'
        logger.info(f'data directory will be {diri}')

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

            logger.info("Making Directory: {0}".format(self.dir))
            os.makedirs(self.dir)

            # copy file(s)
            if os.path.isfile(path):
                ending = path.split('/')[-1].split('.')[-1]
                dst = f'{self.dir}/{self.name}.{ending}'
                logger.info(f'copying {path} to {dst}')
                copy2(path, dst)

            elif os.path.isdir(path):
                logger.info(f'copying data from {path} to {self.dir}')
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
        self.nlcs = len(os.listdir(self._mosfit_data_))
        logger.info(f'{self.nlcs} found in data')

    def write_pkl_to_csv(self):

        logger.info('converting .pkl to .csv\'s')

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
            logger.info(f'making directory {self._mosfit_data_}')
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

            logger.debug(f'writing file {fname}')
            with open(fname, 'w') as fout:
                ascii.write(lc, fout)

    def get_data(self, method):
        if 'sncosmo' in method:
            return self._sncosmo_data_
        elif 'mosfit' in method:
            if not os.path.isdir(self._mosfit_data_):
                self.write_pkl_to_csv()
            return self._mosfit_data_

    def use_method(self, method, job_id):

        logger.info(f'DataHandler for {self.name} configured to use {method}')
        self.latest_method = method

        # initialize ResultHandler for the method
        if self.latest_method not in self.rhandlers.keys():
            if 'sncosmo' in self.latest_method:
                rhandler = SNCosmoResultHandler(self, job_id)
            elif 'mosfit' in self.latest_method:
                rhandler = MosfitResultHandler(self, job_id)
            else:
                raise ValueError(f'method {self.latest_method} not known')
            self.rhandlers[self.latest_method] = rhandler
        self.data = self.get_data(method)
        logger.debug(f'using data stored in {self.data}')

    def save_dh_dict(self):

        dh_dict = {}
        for key in self.__dict__.keys():
            dh_dict[key] = self.__dict__[key]

        if not self.dh_dict:
            self.dh_dict = dh_dict_path(self.name, self.latest_method)

        logger.debug(f'saving the DataHandler dictionary to {self.dh_dict}')

        with open(self.dh_dict, 'wb') as fout:
            pickle.dump(dh_dict, fout)

    def save_me(self):

        logger.debug(f'saving the DataHandler to {self.save_path}')

        with open(self.save_path, 'wb') as fout:
            pickle.dump(self, fout)

    def results(self, method=None, cl=0.9):
        logger.info(f'getting results for {self.name} analyzed by {method}')
        if not method:
            method = self.latest_method
        rhandler = self.rhandlers[method]

        rhandler.collect_results()
        rhandler.get_t_exp_dif_distribution()
        self.plotter.hist_t_exp_dif(rhandler, cl)

    def get_good_lcIDs(self,
                       req_prepeak=None,
                       req_postpeak=None,
                       req_max_timedif=None,
                       req_std=None,
                       check_band='any'):
        """
        Selects lightcurves based on the number of detections before and after the peak,
        a maximum time difference between detections and a measure for the spread of the data points.
        :param req_prepeak: int (same value for all bands) or dict (with the bands as keys), requested
                            number of data points before the observed peak (in each band)
        :param req_postpeak: same as req_prepeak but for datapoints after the peak
        :param req_max_timedif: float, if any two detections are further apart than this value, the LC will be cut
        :param req_std: float, a minimum spread that the LC datapoints need to have to enter the selection
        :param check_band: either 'any' or 'all', specify if all bands have to fulfill the requests or just any
                            use 'all' if either of the requests is a dict
        :return: list of IDs that comply with the requests
                 list of IDs that don't comply
        """
        # TODO: test this

        with open(self._sncosmo_data_, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        IDs = []
        cut_IDs = []

        for req in [req_prepeak, req_postpeak]:
            if req is not None:
                if type(req) is int or type(req) is dict:
                    pass
                else:
                    raise TypeError('Input type has to be int or dict')

        for req in [req_max_timedif, req_std]:
            if req is not None:
                if type(req) in [int, float, dict]:
                    pass
                else:
                    raise (TypeError('Input type has to be int, float or dict'))

        for lc, ID in zip(data['lcs'], data['meta']['idx_orig']):

            bands = np.unique([lc['band']])
            band_masks = {}
            for band in bands: band_masks[band] = lc['band'] == band
            comply_prepeak = {}
            comply_postpeak = {}
            comply_max_timedif = {}
            comply_std = {}

            for i, band in enumerate(bands):

                lc_masked = lc[band_masks[band]]
                nepochs = len(lc_masked)
                peak_phase = lc_masked['time'][np.argmax(lc_masked['flux'])]

                # check for compliance with required pre peak epochs
                npre_peak = len(lc_masked[lc_masked['time'] < peak_phase])
                if req_prepeak is not None and type(req_prepeak) == dict and band in req_prepeak.keys():
                    comply_prepeak[band] = npre_peak >= req_prepeak[band]
                elif req_prepeak is not None and type(req_prepeak) == int:
                    comply_prepeak[band] = npre_peak >= req_prepeak
                else:
                    comply_prepeak[band] = True

                # check for compliance with required post peak epochs
                npost_peak = len(lc_masked[lc_masked['time'] > peak_phase])
                if req_postpeak is not None and type(req_postpeak) == dict and band in req_postpeak.keys():
                    comply_postpeak[band] = npost_peak >= req_postpeak[band]
                elif req_postpeak is not None and type(req_postpeak) == int:
                    comply_postpeak[band] = npost_peak >= req_postpeak
                else:
                    comply_postpeak[band] = True

                # check for compliance with required maximal time difference
                timedif = [lc_masked['time'][j + 1] - lc_masked['time'][j] for j in range(nepochs - 1)]
                if req_max_timedif is not None and len(timedif) != 0:
                    max_timedif = max(timedif)
                    if type(req_max_timedif) is dict and band in req_max_timedif.keys():
                        comply_max_timedif[band] = max_timedif <= req_max_timedif[band]
                    elif type(req_max_timedif) is int or type(req_max_timedif) is float:
                        comply_max_timedif[band] = max_timedif <= req_max_timedif
                else:
                    comply_max_timedif[band] = True

                # check for compliance with required custom standard deviation / variance
                med_flux = np.median(lc_masked['flux'])
                std = np.median((np.array(lc_masked['flux']) - med_flux) ** 2 / np.array(lc_masked['fluxerr'] ** 2))
                if req_std is not None:
                    if type(req_std) is dict and band in req_std.keys():
                        comply_std[band] = std >= req_std[band]
                    elif type(req_std) is int or type(req_std) is float:
                        comply_std[band] = std >= req_std
                else:
                    comply_std[band] = True

            if check_band == 'any':
                if np.any(list(comply_prepeak.values())) & np.any(list(comply_postpeak.values())) & \
                        np.any(list(comply_max_timedif.values())) & np.any(list(comply_std.values())):
                    IDs.append(ID)
                else:
                    cut_IDs.append(ID)
            elif check_band == 'all':
                if np.all(list(comply_prepeak.values())) & np.all(list(comply_postpeak.values())) & \
                        np.all(list(comply_max_timedif.values())) & np.all(list(comply_std.values())):
                    IDs.append(ID)
                else:
                    cut_IDs.append(ID)
            else:
                raise TypeError(f'Input {check_band} for check_band not understood!')
        return IDs, cut_IDs


def dh_dict_path(name, method):
    return f'{dh_dict_dir}/{name}_{method}.pkl'


def dh_path(name):
    return f'{dh_dict_dir}/{name}.pkl'


def load_dh(name):
    name = dh_path(name)
    with open(name, 'rb') as fin:
        dhandler = pickle.load(fin)
    return dhandler


class DataImportWarning(UserWarning):
    def __init__(self, msg):
        self.msg = msg


class DataImportError(Exception):
    def __init__(self, msg):
        self.msg = msg
