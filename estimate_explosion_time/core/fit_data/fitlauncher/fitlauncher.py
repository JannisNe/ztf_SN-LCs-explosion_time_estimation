from estimate_explosion_time.cluster import submit_to_desy, n_tasks
from estimate_explosion_time.shared import pickle_dir, cache_dir, get_custom_logger, main_logger_name, TqdmToLogger, \
    fh_dir
import os
import logging
import shutil
from tqdm import tqdm
import pickle


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
tqdm_info = TqdmToLogger(logger, level=logging.DEBUG)
tqdm_deb = TqdmToLogger(logger, level=logging.INFO)


fitter_file = f'{fh_dir}/fitter_dict.pkl'


def read_fitter_dict():
    if os.path.isfile(fitter_file):
        logger.debug('loading fitter dictionary')
        with open(fitter_file, 'rb') as f:
            fitter_dict = pickle.load(f)
    else:
        logger.debug('no fitter dictionary found. Making new one')
        fitter_dict = {}
    return fitter_dict


class Fitter:

    fitter_dict = {}

    @classmethod
    def get_fitter(cls, method_name):
        cls.fitter_dict = read_fitter_dict()
        if method_name in cls.fitter_dict.keys():
            return cls.fitter_dict[method_name]
        else:
            fitter = cls(method_name)
            cls.fitter_dict[method_name] = fitter
            cls.save_fh_dict()
            return fitter

    @classmethod
    def save_fh_dict(cls):
        logger.debug('saving fitter dictionary')
        with open(fitter_file, 'wb') as f:
            pickle.dump(cls.fitter_dict, f)

    def __init__(self, method_name):

        self.method_name = method_name
        self.cache_dir = self.get_cache_root()
        self.update_cache_dir()
        self.job_id = None

    def fit_lcs(self, dhandler):


        dh_job_id = None
        if self.method_name in dhandler.rhandlers.keys():
            dh_job_id = dhandler.rhandlers[self.method_name].job_id

        # quiere cluster to see if there are still tasks from previuos fit running
        if dh_job_id and (n_tasks(dh_job_id) > 0):
            N = n_tasks(dh_job_id, print_full=True)
            logger.info(f'Still {N} tasks from previous fit in queue! Continuing this fit')
            return

        outdir_name = f'{pickle_dir}/{dhandler.name}'
        outdir = f'{outdir_name}/{self.method_name}'

        # If directory doesn't exist, make one.
        for this_dir in [outdir_name, outdir]:
            if not os.path.exists(this_dir):
                logger.debug(f'Making Fitter output directory {this_dir}')

        # If outdir does contain stuff, clear it. That deletes old fit results, that stay after
        # collect_results in the case of Mosfit
        for file in tqdm(os.listdir(outdir), desc='clearing fitter output directory', file=tqdm_info, mininterval=30):
            os.remove(f'{outdir}/{file}')

        self.cache_dir = f'{self.get_cache_root()}/{dhandler.name}'
        self.update_cache_dir()

        logger.info('submitting jobs to DESY cluster')

        # TODO: set njobs to dhandler.nlcs
        self.job_id = submit_to_desy(self.method_name,
                                     dhandler.get_data(self.method_name),
                                     outdir,
                                     cache=self.cache_dir,
                                     njobs=dhandler.nlcs,
                                     simulation_name=dhandler.name)

        logger.info(f'job-ID is {self.job_id}')

        dhandler.pickle_dir = outdir
        dhandler.use_method(self.method_name, self.job_id)
        dhandler.save_me()

        Fitter.save_fh_dict()

    def update_cache_dir(self):

        logger.debug(f'cache directory for Fitter using {self.method_name} is {self.cache_dir}')

        if not os.path.isdir(self.cache_dir):
            logger.debug('making cache directory')
            os.mkdir(self.cache_dir)

        else:

            if self.cache_dir != self.get_cache_root():
                logger.debug('cache directory already exists')

                for obj in tqdm(os.listdir(self.cache_dir + '/'),
                                desc='clearing cache directory',
                                file=tqdm_info,
                                mininterval=30):

                    full_path = f'{self.cache_dir}/{obj}'

                    if os.path.isfile(full_path):
                        os.remove(full_path)
                    elif os.path.isdir(full_path):
                        shutil.rmtree(full_path)
                    else:
                        raise FitterError(f'Could not remove {full_path}')

    def get_cache_root(self):
        this_cache_dir = f'{cache_dir}/{self.method_name}'
        return this_cache_dir


class FitterError(Exception):
    def __init__(self, msg):
        self.msg = msg
