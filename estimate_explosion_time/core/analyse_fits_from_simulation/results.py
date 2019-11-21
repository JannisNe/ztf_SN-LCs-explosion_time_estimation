import numpy as np
import os
from astropy.table import Table, vstack
import logging
import pickle
from tqdm import tqdm
from estimate_explosion_time.core.analyse_fits_from_simulation.get_source_explosion_time.find_explosion_time\
    import get_explosion_time
from estimate_explosion_time.shared import output_dir, plots_dir


class ResultHandler:

    def __init__(self, dhandler):

        logging.debug(f'configuring ResultHandler for dhandler {dhandler.name} '
                      f'using the method {dhandler.method}')

        if dhandler.pickle_dir:
            self.pickle_dir = dhandler.pickle_dir
        else:
            raise ResultError('No results for this data!')

        self.dhandler = dhandler
        self.method_used = dhandler.method
        self.collected_data = None


class SNCosmoResultHandler(ResultHandler):

    def collect_results(self):

        logging.info('collecting fit results')

        data = []
        for file in tqdm(os.listdir(self.pickle_dir), desc='collecting fit results'):

            full_file_path = f'{self.pickle_dir}/{file}'
            with open(full_file_path, 'rb') as f:
                dat = pickle.load(f)

            data += [{'fit_output': Table(dat)}]
            os.remove(full_file_path)

        self.collected_data = data

        collected_data_filename = f'{self.pickle_dir}.pkl'

        with open(collected_data_filename, 'wb') as fout:
            pickle.dump(self.collected_data, fout)

        self.dhandler.collected_data = collected_data_filename
        self.dhandler.save_dh_dict()









class MosfitResultHandler(ResultHandler):

    def collect_results(self):
        # TODO: implement this
        pass

    def get_fitted_explosion_times(self):
        # TODO: implement this
        pass


class ResultError(Exception):
    def __init__(self, msg):
        self.msg = msg
