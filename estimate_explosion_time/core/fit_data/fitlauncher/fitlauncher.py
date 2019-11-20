from estimate_explosion_time.cluster import submit_to_desy, wait_for_cluster
from estimate_explosion_time.shared import pickle_dir, cache_dir
import os
import logging


class Fitter:

    def __init__(self, method_name):

        self.method_name = method_name
        self.cache_dir = self.get_cache_root()
        self.update_cache_dir()

    def fit_lcs(self, dhandler):

        dhandler.use_method(self.method_name)

        self.cache_dir = f'{self.get_cache_root()}/{dhandler.name}'
        self.update_cache_dir()

        outdir = f'{pickle_dir}/{dhandler.name}_{self.method_name}'
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        logging.info('submitting jobs to DESY cluster')

        #TODO: set njobs to dhandler.nlcs
        submit_to_desy(self.method_name, dhandler.data, outdir, cache=self.cache_dir,
                       njobs=10, simulation_name=dhandler.name)

        logging.info('waiting on cluster')
        wait_for_cluster()

        dhandler.pickle_dir = outdir
        dhandler.save_dh_dict()

    def update_cache_dir(self):
        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)
        logging.debug(f'cache directory for Fitter using {self.method_name} is {self.cache_dir}')

    def get_cache_root(self):
        this_cache_dir = f'{cache_dir}/{self.method_name}'
        return this_cache_dir
