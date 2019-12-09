import numpy as np
import os
from astropy.table import Table, vstack
import logging
import pickle
import json
from tqdm import tqdm
from estimate_explosion_time.shared import get_custom_logger, main_logger_name, TqdmToLogger
from estimate_explosion_time.cluster import wait_for_cluster
from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import Fitter


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
tqdm_deb = TqdmToLogger(logger, level=logging.DEBUG)
tqdm_info = TqdmToLogger(logger, level=logging.INFO)


class ResultHandler:

    def __init__(self, dhandler, job_id):

        logger.debug(f'configuring ResultHandler for dhandler {dhandler.name} '
                      f'using the method {dhandler.latest_method}')

        if dhandler.pickle_dir:
            self.pickle_dir = dhandler.pickle_dir
        else:
            raise ResultError('No results for this data!')

        self.dhandler = dhandler
        self.method = dhandler.latest_method
        self.collected_data = None
        self.t_exp_dif = None
        self.t_exp_dif_error = None
        self.job_id = job_id

    def save_data(self):
        logger.debug(f'saving collected data data to {self.dhandler.collected_data}')
        with open(self.dhandler.collected_data, 'wb') as f:
            pickle.dump(self.collected_data, f)

    def get_t_exp_dif_distribution(self):

        # check the first entry in collected data for t_exp_dif to check if it has been calculated
        if 't_exp_dif' not in self.collected_data[0].keys():
            raise ResultError('fitted explosion time is missing!')

        logger.info('getting distribution of the difference between true and fitted explosion time')

        self.t_exp_dif = np.array([data['t_exp_dif'] for data in self.collected_data])
        self.t_exp_dif_error = np.array([data['t_exp_dif_error'] for data in self.collected_data])

    def collect_results(self):

        if not self.collected_data:

            if self.job_id:
                job_status = wait_for_cluster(self.job_id)
            else:
                inp = input('No job id specified. Go on trying to collect fit results? ')
                if inp in ['y', 'yes']:
                    job_status = True
                else:
                    raise ResultError('no job_id specified')

            # if all jobs are done, unset job id and continue with collecting results
            if job_status:

                self.job_id = None

                if len(os.listdir(self.pickle_dir)) is not 0:
                    self.sub_collect_results()
                    collected_data_filename = f'{self.pickle_dir}.pkl'
                    self.dhandler.collected_data = collected_data_filename
                    self.save_data()
                    self.dhandler.save_me()

                else:
                    raise ResultError(f'No result files in {self.pickle_dir}!')

            else:
                raise ResultError(f'Exiting before all tasks were done.')

        else:
            logger.debug('results were already collected')

    def sub_collect_results(self):
        """
        implemented in subclasses
        """
        raise NotImplementedError


class SNCosmoResultHandler(ResultHandler):

    def sub_collect_results(self):

        logger.info('collecting fit results')

        listed_pickle_dir = os.listdir(self.pickle_dir)

        # indices are file name minus one as file names start at 1, indices at 0!
        indices = [int(file.split('.')[0])-1 for file in listed_pickle_dir]

        # get a list of indices, that are not a file in result directory
        # if this list is not empty, raise Error
        missing_indices = []
        for i in range(self.dhandler.nlcs):
            if i not in indices:
                missing_indices.append(i)

        if len(missing_indices) > 0:
            raise ResultError(f'No result files for indices {missing_indices}')

        data = [{}] * (max(indices) + 1)
        for file in tqdm(listed_pickle_dir, desc='collecting fit results', file=tqdm_info, mininterval=30):

            if file.startswith('.'):
                continue

            ind = int(file.split('.')[0])-1

            full_file_path = f'{self.pickle_dir}/{file}'
            with open(full_file_path, 'rb') as f:
                dat = pickle.load(f)

            t_exp_true = np.unique(dat['t_exp_true'])
            if len(t_exp_true) is not 1:
                raise ResultError(f'different explosion times for the same lightcurve')
            else:
                t_exp_true = t_exp_true[0]

            data[ind] = {
                'fit_output': Table(dat),
                't_exp_true': t_exp_true
            }

            # os.remove(full_file_path)

        self.collected_data = data
        self.combine_best_fits()

    def combine_all_model_fits(self):
        """
        Calculates a weighted mean with weights based on the reduced chi2
        """
        logger.info('fitted explosion time is weighted mean of all fits')
        if not self.collected_data:
            self.collect_results()

        for res in self.collected_data:

            fit_output = res['fit_output']

            if 'simsurvey' in self.dhandler.name:
                fit_output = fit_output[['nugent' not in model for model in fit_output['model']]]

            weights = fit_output['red_chi2'] ** 2 / sum(fit_output['red_chi2'] ** 2)
            res['t_exp_fit'] = np.average(fit_output['t_exp_fit'], weights=weights)
            res['t_exp_dif_error'] = np.average(fit_output['t0_e'], weights=weights)
            res['t_exp_dif'] = res['t_exp_true'] - res['t_exp_fit']
            logger.debug(f'explosion time difference: {res["t_exp_dif"]}')

        self.dhandler.save_me()

    def combine_best_fits(self):
        """
        calculates
        """
        logger.info('fitted explosion time is median of best fits')
        if not self.collected_data:
            self.collect_results()

        for res in self.collected_data:

            fit_output = res['fit_output']

            nugent_mask = \
                ['nugent' not in model for model in fit_output['model']] \
                if 'simsurvey' in self.dhandler.name \
                else [True] * len(fit_output)

            chi2 = fit_output['red_chi2']
            mask = chi2 <= (min(chi2[nugent_mask]) * (3))

            res['t_exp_fit'] = np.median(fit_output['t_exp_fit'][mask])
            res['mask'] = np.array(mask) & np.array(nugent_mask)

            cl = 0.9
            error_quantile = np.quantile(fit_output['t0_e'][mask], [0.5-cl/2, 0.5+cl/2])
            stat_error = error_quantile[1] - error_quantile[0]
            res['t_exp_dif_error'] = max(max(fit_output['t0_e'][mask]), stat_error)

            res['t_exp_dif'] = res['t_exp_fit'] - res['t_exp_true']

        self.dhandler.save_me()


class MosfitResultHandler(ResultHandler):

    def sub_collect_results(self, cl=0.9):

        logger.info('collecting fit results')

        data = []
        with open(self.dhandler._sncosmo_data_, 'rb') as f:
            sim = pickle.load(f, encoding='latin1')

        t_exp_true = sim['meta']['t0']

        for file in tqdm(os.listdir(self.pickle_dir), desc='collecting fit results', file=tqdm_info, mininterval=30):

            if file.startswith('.'):
                continue

            full_file_path = f'{self.pickle_dir}/{file}'

            # logger.debug(f'opening {full_file_path}')

            with open(full_file_path, 'r') as f:
                dat = json.loads(f.read())

            if 'name' not in dat:
                dat = dat[list(dat.keys())[0]]

            name = dat['name']
            indice = int(name) # TODO: change this to int(name)-1 for future imports
            model = dat['models'][0]

            posterior_t_exp = []
            for rs in model['realizations']:
                rspars = rs['parameters']
                posterior_t_exp.append(rspars['texplosion']['value'] + rspars['reference_texplosion']['value'])

            data += [{
                't_exp_posterior': posterior_t_exp,
                't_exp_fit': np.median(posterior_t_exp),
                't_exp_true': t_exp_true[indice],
                't_exp_dif': np.median(posterior_t_exp) - t_exp_true[indice],
                't_exp_dif_error': np.std(posterior_t_exp),
                't_exp_dif_ic': np.quantile(posterior_t_exp, [0.5-cl/2, 0.5+cl/2])
            }]

        self.collected_data = data


class ResultError(Exception):
    def __init__(self, msg):
        self.msg = msg
