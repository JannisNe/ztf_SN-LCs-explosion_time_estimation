from estimate_explosion_time.cluster import submit_to_desy, wait_for_cluster
from estimate_explosion_time.shared import pickle_dir, cache_dir, get_custom_logger, main_logger_name
import os
import logging


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())


class Fitter:

    def __init__(self, method_name):

        self.method_name = method_name
        self.cache_dir = self.get_cache_root()
        self.update_cache_dir()
        self.job_id = None

    def fit_lcs(self, dhandler):

        if self.job_id:
            raise FitterError('Still tasks from previous fit in queue!')

        outdir_name = f'{pickle_dir}/{dhandler.name}'
        outdir = f'{outdir_name}/{self.method_name}'

        for dir in [outdir_name, outdir]:
            if not os.path.exists(dir):
                os.mkdir(dir)

        self.cache_dir = f'{self.get_cache_root()}/{dhandler.name}'
        self.update_cache_dir()

        logger.info('submitting jobs to DESY cluster')

        #TODO: set njobs to dhandler.nlcs
        self.job_id = submit_to_desy(self.method_name,
                                     dhandler.get_data(self.method_name),
                                     outdir,
                                     cache=self.cache_dir,
                                     njobs=4,
                                     simulation_name=dhandler.name)

        logger.info(f'job-ID is {self.job_id}')

        dhandler.pickle_dir = outdir
        dhandler.use_method(self.method_name, self.job_id)
        dhandler.save_me()

    def update_cache_dir(self):
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)
        logger.debug(f'cache directory for Fitter using {self.method_name} is {self.cache_dir}')

    def get_cache_root(self):
        this_cache_dir = f'{cache_dir}/{self.method_name}'
        return this_cache_dir


class FitterError(Exception):
    def __init__(self, msg):
        self.msg = msg
