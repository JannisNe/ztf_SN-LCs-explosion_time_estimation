import os
from warnings import warn
from shutil import copytree, copy2
import logging
import pickle
from astropy.table import Table
import astropy
from astropy.io import ascii
import astropy.cosmology as cosmo
import numpy as np
import multiprocessing
from tqdm import tqdm
import sncosmo
from estimate_explosion_time.shared import simulation_dir, real_data_dir, dh_dir,\
    get_custom_logger, main_logger_name, es_scratch_dir, TqdmToLogger
from estimate_explosion_time.core.analyse_fits_from_simulation.results import SNCosmoResultHandler, \
    MosfitResultHandler, ResultError
from estimate_explosion_time.core.analyse_fits_from_simulation.plots import Plotter
from estimate_explosion_time.core.analyse_fits_from_simulation.get_source_explosion_time.find_explosion_time \
    import get_explosion_time, ExplosionTimeError
from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import FitterError


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())


class DataHandler:

    def __init__(self, path, name, simulation=True, **explosion_time_kwargs):

        self.name = name
        self.orig_path = path
        self.data = None
        self.nlcs = None
        self.pickle_dir = None
        self.latest_method = None
        self.dh_dict = None
        self.save_path = DataHandler.dh_path(self.name)
        self.collected_data = None
        self.rhandlers = {}
        self.selected_indices = None
        self.selection_string = 'all'

        self.meta_data = None
        self.explosion_time_kwargs = None
        self.queue = None

        if simulation:
            diri = simulation_dir
        else:
            diri = real_data_dir

        diri += f'/{name}'
        logger.debug(f'data directory will be {diri}')

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

                iadd += 1
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
                raise NotImplementedError

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

        # test get_data
        self.get_data('sncosmo')
        self.get_data('mosfit')

        self.nlcs = len(os.listdir(self._mosfit_data_))
        logger.info(f'{self.nlcs} lightcurves found in data')
        self.save_me()

    def write_pkl_to_csv(self):

        logger.info('converting .pkl to .csv\'s')

        with open(self._sncosmo_data_, 'rb') as fin:

            data = pickle.load(fin, encoding='latin1')
            lcs = data['lcs']
            meta = data['meta']
            self.nlcs = len(lcs)

        if 'lumdist' not in meta:
            logger.warning('Luminosity distance not given! Calculating it using Planck15.')

        if not os.path.exists(self._mosfit_data_):
            logger.info(f'making directory {self._mosfit_data_}')
            os.mkdir(self._mosfit_data_)

        else:
            raise DataImportError('Folder with CSV files already exists!')

        itr = range(len(lcs))

        add_columns = {
            'instrument': 'ZTF_camera',
            'telescope': 'ZTF',
            'name': [f'{ind}' for ind in itr],
            'reference': 'JannisNecker',
            'u_time': 'MJD',
            'redshift':[meta['z'][ind] if 'z' in meta else None for ind in itr],
            'ebv': [meta['hostebv'][ind] if 'hostebv' in meta else None for ind in itr],
            'lumdist': [meta['lumdist'][ind] if 'lumdist' in meta else \
                        cosmo.Planck15.luminosity_distance(meta['z'][ind]) if 'z' in meta else \
                        None for ind in itr],
            'ID': [meta['idx_orig'][ind] if 'idx_orig' in meta else None for ind in itr]
        }

        for ind in itr if logger.getEffectiveLevel() > logging.INFO else tqdm(itr, desc='writing to csv'):

            lc = Table(lcs[ind])
            lc['band'][lc['band'] == 'desi'] = 'ztfi'
            fname = f'{self._mosfit_data_}/{ind}.csv'

            this_add_columns = {key: val if type(val) != list else val[ind]
                                for key, val in add_columns.items()}

            write_to_csv(lc, fname, this_add_columns)

    def get_data(self, method):

        if 'sncosmo' in method:
            logger.debug(f'getting {self._sncosmo_data_}')
            ret = self._sncosmo_data_

        elif 'mosfit' in method:
            if not os.path.isdir(self._mosfit_data_):
                self.write_pkl_to_csv()
            logger.debug(f'getting {self._mosfit_data_}')
            ret = self._mosfit_data_
        else:
            raise Exception(f'Method {method} not known!')

        if es_scratch_dir in ret:
            return ret
        elif 'input' in ret:
            return es_scratch_dir + '/input' + ret.split('input')[1]
        else:
            raise DataImportError('No idea where to look for the data :(')

    def get_spectral_time_evolution_file(self, indice):
        """ to be implemented in subclasses """
        raise NotImplementedError

    def get_peak_band(self, ind, meta_data=None):

        logger.debug(f'getting peak band for {ind}')

        if not meta_data:
            with open(self.get_data('sncosmo'), 'rb') as f:
                sndata = pickle.load(f, encoding='latin1')

            meta_data = sndata['meta']

        peak_mags = np.array([
            (f'peak_mag_{b}', meta_data[f'peak_mag_{b}'][ind] if f'peak_mag_{b}' in meta_data else None)
            for b in {'r', 'g'}
        ], dtype={'names': ['band', 'peak_mag'], 'formats': ['<U15', '<f8']})

        logger.debug(f'peak mags: {peak_mags}')

        peak_mags_wo_nans = peak_mags[~np.isnan(peak_mags['peak_mag'])]

        logger.debug(f'without np.nan: {peak_mags_wo_nans}')

        peak_band_ind = None if len(peak_mags_wo_nans) == 0 else \
            0 if len(peak_mags_wo_nans) == 1 else \
            np.argmin(peak_mags_wo_nans['peak_mag'])

        logger.debug(f'peak band indice: {peak_band_ind}')

        peak_band_from_meta = None if peak_band_ind is None else peak_mags_wo_nans['band'][peak_band_ind]
        peak_mag_from_meta = None if peak_band_ind is None else peak_mags_wo_nans['peak_mag'][peak_band_ind]

        logger.debug(f'peak band is {peak_band_from_meta}: {peak_mag_from_meta} mag')

        # if the peak magnitude is no instrument filter name, assume ZTF filters
        peak_band_from_meta_old = None

        if 'peak' in peak_band_from_meta and ('r' in peak_band_from_meta or 'g' in peak_band_from_meta):
            peak_band_from_meta_old = peak_band_from_meta
            peak_band_from_meta = 'ztfr' if 'r' in peak_band_from_meta_old else \
                'ztfg' if 'g' in peak_band_from_meta_old else \
                 None

        if not peak_band_from_meta and peak_band_from_meta_old:
            raise ExplosionTimeError(f'Couldn\'t determine peak band {peak_band_from_meta_old}!')

        logger.debug(f'peak band is {peak_band_from_meta}')

        return peak_band_from_meta, peak_mag_from_meta

    def get_explosion_times_from_template(self, ncpu=10, force_all=False, **kwargs):
        """
        Get the explosion times for all SNe from the correspodning template and save it to the meta data.
        :param ncpu: number of cpus to use in multiprocessing
        :param force_all: bool, if True all explosion times will be estimated, default False
        :param kwargs: to be passed to get_explosion_time_from_template_single()
        """

        logger.info('getting explosion times')
        with open(self.get_data('sncosmo'), 'rb') as f:
            sndata = pickle.load(f, encoding='latin1')

        meta_data = sndata['meta']

        missing_t0_mask = np.equal(np.array(meta_data['t0']), None) if not force_all else [True] * self.nlcs
        missing_t0_indices = np.where(missing_t0_mask)[0]

        logger.debug(f'{len(missing_t0_indices)} missing explosion times')

        if len(missing_t0_indices) == 0:
            logger.info('No missing explosion times.')
            return

        # set these variable so they can be used by get_explosion_time_from_template_single
        self.meta_data = meta_data
        self.explosion_time_kwargs = kwargs

        logger.debug('starting multiprocessing')
        with multiprocessing.Pool(ncpu) as p:

            # case difference so that for INFO a nice progress bar is displayed
            if logger.getEffectiveLevel() == logging.INFO:
                results = list()
                pbar_length = len(missing_t0_indices)
                with tqdm(total=pbar_length, desc='calculating explosion times') as pbar:
                    for i, _ in enumerate(p.imap_unordered(
                            self.get_explosion_time_from_template_single,
                            missing_t0_indices)):
                        results.append(_)
                        pbar.update()

            else:
                results = p.map(self.get_explosion_time_from_template_single,
                                missing_t0_indices)

        results_array = np.array(results)

        for t_exp, ind, idx_orig in results_array:

            idx_orig_orig = sndata['meta']['idx_orig'][int(ind)]
            if idx_orig != idx_orig_orig:
                raise DataInconsistencyError(f'Original Index {idx_orig_orig}: Result array has index {idx_orig}!')

            sndata['meta']['t0'][int(ind)] = t_exp

        if np.any(np.array(sndata['meta']['t0']) == None):
            raise DataImportError('Still some explosion times missing!')

        # save data
        logger.info('saving data to ' + self.get_data('sncosmo'))
        with open(self.get_data('sncosmo'), 'wb') as f:
            pickle.dump(sndata, f)

        # unset variables
        self.meta_data = None
        self.explosion_time_kwargs = None

        self.save_me()

    def get_explosion_time_from_template_single(self, ind, meta_data=None, **explosion_time_kwargs):
        """
        get the explosion time for a single Supernova. Arguments can be set directly or by setting self.meta_data
        and self.explosion_time_kwargs respectively. If both are set an Error is raised
        :param ind: int, indice of the supernova
        :param meta_data: dict, the meta data for the supernova
        :param explosion_time_kwargs: dict, keyword arguments passed to get_explosion_time()
        :return: tuple of floats, explosion time and indice
        """

        logger.debug(f'indice = {ind}')

        if meta_data and self.meta_data:
            raise DataImportError('Two arguments for meta data found!')

        if explosion_time_kwargs and self.explosion_time_kwargs:
            raise DataImportError('Two sets of keyword arguments found!')

        if (not meta_data) and (not self.meta_data):
            logger.debug('Variable meta_data not set. Getting data...')
            with open(self.get_data('sncosmo'), 'rb') as f:
                sndata = pickle.load(f, encoding='latin1')
            meta_data = sndata['meta']

        if not meta_data:
            meta_data = self.meta_data

        if not explosion_time_kwargs and self.explosion_time_kwargs:
            explosion_time_kwargs = self.explosion_time_kwargs

        peak_band_from_meta, _ = self.get_peak_band(ind, meta_data=meta_data)

        if ('peak_band' in explosion_time_kwargs) and peak_band_from_meta:
            raise ExplosionTimeError('Found peak band in meta data but a band was also explicitly given for the '
                                     'explosion time estimation! \n '
                                     f' explosion_time_kwargs: {explosion_time_kwargs.keys()}')

        peak_band = peak_band_from_meta if 'peak_band' not in explosion_time_kwargs else \
            explosion_time_kwargs['peak_band']

        if not peak_band:
            logger.warning(f'No peak band given! Assuming peak date refers to peak in bolometric flux!')

        t_exp = get_explosion_time(self.get_spectral_time_evolution_file(ind),
                                   redshift=meta_data['z'][ind],
                                   peak_mjd=meta_data['t_peak'][ind],
                                   peak_band=peak_band,
                                   **explosion_time_kwargs)

        idx_orig = meta_data['idx_orig'][ind]

        return t_exp, ind, idx_orig

    def use_method(self, method, job_id, force_new_rhandler=False):

        logger.debug(f'DataHandler for {self.name} configured to use {method}')
        self.latest_method = method

        # initialize ResultHandler for the method
        if self.latest_method not in self.rhandlers.keys() or force_new_rhandler:

            if 'sncosmo' in self.latest_method:
                rhandler = SNCosmoResultHandler(self, job_id)
            elif 'mosfit' in self.latest_method:
                rhandler = MosfitResultHandler(self, job_id)
            else:
                raise ValueError(f'method {self.latest_method} not known')

            self.rhandlers[self.latest_method] = rhandler

        # If the result Handler already exists, get it
        else:
            rhandler = self.rhandlers[self.latest_method]

        # if Result Handler job ID is set, check if it's the same as the Fitter's
        if rhandler.job_id:

            if rhandler.job_id != job_id:

                logger.warning(f'Data Handler {self.name}: '
                               f'Result Handler {rhandler.method} '
                               f'still waiting on results from method {method}!')

                inpt = input('continue and overwrite? [y/n] ')

                if inpt in ['y', 'yes']:
                    pass
                else:
                    raise FitterError(f'Data Handler {self.name}: '
                                      f'Result Handler {rhandler.method} '
                                      f'still waiting on results from method {method}!')

        # if Result Handler job ID is not set, do so
        logger.debug(f'setting Result Handler\'s job ID to {job_id}')
        rhandler.job_id = job_id

        # reset attributes that indicate that data was already collected
        rhandler.collected_data = None
        rhandler.t_exp_dif = None
        rhandler.t_exp_dif_error = None

    def save_me(self):

        logger.debug(f'saving the DataHandler to {self.save_path}')

        with open(self.save_path, 'wb') as fout:
            pickle.dump(self, fout)

    def results(self, method=None, cl=0.9, force=False, **kwargs):

        if not method:
            method = self.latest_method
        logger.info(f'getting results for {self.name} analyzed by {method}')

        if force:
            logger.info('forcing re-collecting of results!')
            self.use_method(method=method, job_id=None, force_new_rhandler=True)

        rhandler = self.rhandlers[method]
        plotter = Plotter(self, method)

        if not self.selected_indices:
            raise ResultError('No selection has been made!')

        try:
            rhandler.collect_results(force=force)
            rhandler.get_t_exp_dif_distribution()
            plotter.hist_t_exp_dif(cl)
            plotter.hist_ic90_deviaton()
            plotter.plot_tdif_tdife()
            plotter.plot_lcs_where_fit_fails(**kwargs)
        except KeyboardInterrupt:
            pass

        finally:
            self.save_me()

    def select_and_adjust_selection_string(self, select_all=False, **kwargs):

        if (len(kwargs.keys()) < 1) or select_all:
            logger.debug('selecting all SNe')
            self.selection_string = 'all'
            self.selected_indices = list(range(self.nlcs))
        else:
            logger.debug('making selection')
            self.selection_string = ''
            for kw_item in kwargs.items():
                if kw_item[1] is not None:
                    if kw_item[0] == 'req_texp_dif':
                        self.selection_string += f'tdif{kw_item[1][0]}{kw_item[1][1]}'
                    elif 'print' not in kw_item[0]:
                        self.selection_string += f'{kw_item[0]}{kw_item[1]}_'

            logger.debug(f'selection string is {self.selection_string}')

            self.select(**kwargs)

    def select(self,
               req_prepeak=None,
               req_postpeak=None,
               req_max_timedif=None,
               req_peak_mag=None, use_magnitude_system='AB',
               req_std=None,
               req_texp_dif=None,
               check_band='any',
               sigma=4,
               print_selected_indices=False):
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

        with open(self._sncosmo_data_, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        meta = data['meta']
        IDs = meta['idx_orig']
        peak_mags_r = meta.get('peak_mag_r', [])
        peak_mags_g = meta.get('peak_mag_g', [])
        lcs = data['lcs']

        indices = []
        comply_IDs = []
        cut_IDs = []

        for req in [req_prepeak, req_postpeak]:
            if req is not None:
                if type(req) is int or type(req) is dict:
                    pass
                else:
                    raise TypeError('Input type has to be int or dict')

        for req in [req_max_timedif, req_std, req_peak_mag]:
            if req is not None:
                if type(req) in [int, float, dict]:
                    pass
                else:
                    raise (TypeError('Input type has to be int, float or dict'))

        for j, lc in enumerate(lcs):

            logger.debug('checking indice ' + str(j))
            ID = IDs[j]
            peak_mag_r = peak_mags_r[j] if len(peak_mags_r) > 0 else None
            peak_mag_g = peak_mags_g[j] if len(peak_mags_g) > 0 else None

            noise_mask = lc['flux']/lc['fluxerr'] > sigma
            lc = lc[noise_mask]

            bands = np.unique([lc['band']])
            band_masks = {}

            for band in bands:
                band_masks[band] = lc['band'] == band

            comply_prepeak = {}
            comply_postpeak = {}
            comply_max_timedif = {}
            comply_peak_mag = {}
            comply_std = {}

            comply_dictionaries = [comply_prepeak, comply_postpeak, comply_max_timedif, comply_peak_mag, comply_std]

            for i, band in enumerate(bands):

                logger.debug('checking band ' + band)

                lc_masked = lc[band_masks[band]]
                if len(lc_masked) == 0:
                    continue

                nepochs = len(lc_masked)
                peak_phase = lc_masked['time'][np.argmax(lc_masked['flux'])]

                # check for compliance with required pre peak epochs
                if req_prepeak:
                    npre_peak = len(lc_masked[lc_masked['time'] < peak_phase])
                    logger.debug(f'{npre_peak} pre peak detections')
                    if type(req_prepeak) == dict and band in req_prepeak.keys():
                        comply_prepeak[band] = npre_peak >= req_prepeak[band]
                    elif type(req_prepeak) == int:
                        comply_prepeak[band] = npre_peak >= req_prepeak

                # check for compliance with required post peak epochs
                if req_postpeak:
                    npost_peak = len(lc_masked[lc_masked['time'] > peak_phase])
                    logger.debug(f'{npost_peak} post peak detections')
                    if type(req_postpeak) == dict and band in req_postpeak.keys():
                        comply_postpeak[band] = npost_peak >= req_postpeak[band]
                    elif type(req_postpeak) == int:
                        comply_postpeak[band] = npost_peak >= req_postpeak

                # check for compliance with required maximal time difference
                if req_max_timedif:
                    timedif = [lc_masked['time'][j + 1] - lc_masked['time'][j] for j in range(nepochs - 1)]
                    if len(timedif) != 0:
                        max_timedif = max(timedif)
                        logger.debug(f'{max_timedif} maximal difference between any following epochs')
                        if type(req_max_timedif) is dict and band in req_max_timedif.keys():
                            comply_max_timedif[band] = max_timedif <= req_max_timedif[band]
                        elif type(req_max_timedif) is int or type(req_max_timedif) is float:
                            comply_max_timedif[band] = max_timedif <= req_max_timedif
                    else:
                        logger.warning(f'Length of timedif = 0!')

                # check for compliance with required peak magnitude
                if req_peak_mag:
                    logger.debug(f'peak_mag_r: {peak_mag_g}, peak_mag_g: {peak_mag_g}')
                    if not peak_mag_g and not peak_mag_r:
                        raise SelectionError(f'No peak magnitude given! Can\'t do magnitude selection!')
                    peak_mag = min((m for m in [peak_mag_r, peak_mag_g] if m))
                    logger.debug(f'Peak Magnitude is {peak_mag}.')
                    if type(req_peak_mag) is dict and band in req_peak_mag.keys():
                        comply_peak_mag[band] = peak_mag <= req_peak_mag[band]
                    else:
                        comply_peak_mag[band] = peak_mag <= req_peak_mag

                # check for compliance with required custom standard deviation / variance
                if req_std is not None:
                    med_flux = np.median(lc_masked['flux'])
                    std = np.median((np.array(lc_masked['flux']) - med_flux) ** 2 / np.array(lc_masked['fluxerr'] ** 2))
                    logger.debug(f'{std} spread')
                    if type(req_std) is dict and band in req_std.keys():
                        comply_std[band] = std >= req_std[band]
                    elif type(req_std) is int or type(req_std) is float:
                        comply_std[band] = std >= req_std

            crit_str = np.array(['prepeak', 'postpeak', 'max timedif', 'peak mag', 'std'])
            comply_lists = [list(dictionary.values()) for dictionary in comply_dictionaries]

            if check_band == 'any':
                bool_list = [np.any(l) if len(l) > 0 else True for l in comply_lists]
                if np.all(bool_list):
                    logger.debug('all good')
                    comply_IDs.append(ID)
                    indices.append(j)
                else:
                    fail_str = str(crit_str[np.invert(bool_list)])
                    logger.debug('failed because ' + fail_str)
                    cut_IDs.append(ID)

            elif check_band == 'all':
                bool_list = [np.all(l) if len(l) > 0 else True for l in comply_lists]
                if np.all(bool_list):
                    logger.debug('all good')
                    comply_IDs.append(ID)
                    indices.append(j)
                else:
                    fail_str = str(crit_str[np.invert(bool_list)])
                    logger.debug('failed because ' + fail_str)
                    cut_IDs.append(ID)

            else:
                raise TypeError(f'Input {check_band} for check_band not understood!')

        self.selected_indices = indices

        # select based on the error on the explosion time fit
        if req_texp_dif:

            logger.debug('checking for required difference in texplosion estimation')

            if not len(req_texp_dif) == 2:
                raise ValueError('req_texp_dif has to be [method, value]!')
            error = self.rhandlers[req_texp_dif[0]].t_exp_dif_error
            val = req_texp_dif[1]

            indices = []
            for indice in self.selected_indices:
                logger.debug('checking indice ' + str(indice))
                if error[indice] <= val:
                    indices.append(indice)
                else:
                    logger.debug('failed')

            self.selected_indices = indices

        selected_percentage = len(indices)/len(data['lcs'])
        logger.debug('selected {0:.2f}% of all lightcurves'.format(selected_percentage*100))

        if print_selected_indices:
            logger.info('selected indices: \n' + str(self.selected_indices))

    @staticmethod
    def dh_dict_path(name, method):
        return f'{dh_dir}/{name}_{method}.pkl'

    @staticmethod
    def get_dhandler(name, path=None, simulation=True):
        if os.path.isfile(DataHandler.dh_path(name)):
            logger.info(f'DataHandler for {name} already exists, loading it ...')
            return DataHandler.load_dh(name)
        elif path:
            logger.info(f'creating DataHandler for {name}')
            return DataHandler(path, name, simulation)
        else:
            raise DataImportError(f'ResultHandler for {name} doesn\'t exist. '
                                  f'Please sepcify path to the data!')

    @staticmethod
    def load_dh(name):
        name = DataHandler.dh_path(name)
        with open(name, 'rb') as fin:
            dhandler = pickle.load(fin)
        logger.debug(f'loaded DataHandler {dhandler.name}')
        return dhandler

    @staticmethod
    def dh_path(name):
        return f'{dh_dir}/{name}.pkl'


def write_to_csv(lc, filename, add_columns=dict()):

    if type(lc) != astropy.table.table.Table:
        lc = Table(lc)

    for col, val in add_columns.items():

        if col not in lc.keys():

            lc[col] = [add_columns[col]] * len(lc)

            # if col == 'name':
            #     lc[col] = [f'{ind}'] * len(lc)
            #
            # if col == 'redshift':
            #     lc[col] = [meta['z'][ind] if 'z' in meta else None] * len(lc)
            #
            # if col == 'ebv':
            #     lc[col] = [meta['hostebv'][ind] if 'hostebv' in meta else None] * len(lc)
            #
            # if col == 'ID':
            #     try:
            #         lc[col] = [int(meta['idx_orig'][ind]) if 'idx_orig' in meta else None] * len(lc)
            #     except ValueError:
            #         lc[col] = [meta['idx_orig'][ind] if 'idx_orig' in meta else None] * len(lc)
            #
            # if col == 'lumdist':
            #     lc[col] = [meta['lumdist'][ind] if 'lumdist' in meta else
            #                cosmo.Planck15.luminosity_distance(meta['z'][ind]) if 'z' in meta else
            #                None] * len(lc)
            #
            # if col == 'zpsys':
            #     lc[col] = [meta['zpsys'][ind] if 'zpsys' in meta else None] * len(lc)

        else:
            logger.warning(f'Column {col} already exists!')
            # raise IndexError(f'Column {col} already exists!')

    with open(filename, 'w') as fout:
        ascii.write(lc, fout)


class DataImportWarning(UserWarning):
    def __init__(self, msg):
        self.msg = msg


class DataImportError(Exception):
    def __init__(self, msg):
        self.msg = msg


class DataInconsistencyError(Exception):
    def __init__(self, msg):
        self.msg = msg


class SelectionError(Exception):
    def __init__(self, msg):
        self.msg = msg