from estimate_explosion_time.shared import output_dir
import os
from astropy.table import Table, vstack
import logging
import pickle
from tqdm import tqdm


class ResultHandler:

    def __init__(self, dhandler):

        logging.debug(f'configuring ResultHandler for dhandler {dhandler.name} '
                      f'using the method {dhandler.method}')

        if dhandler.pickle_dir:
            self.pickle_dir = dhandler.pickle_dir
        else:
            raise ResultError('No results for this data!')

        self.method_used = dhandler.method
        self.collected_data = None

    def collect_results(self):

        logging.info('collecting fit results')

        if 'sncosmo' in self.method_used:

            data = []
            for file in tqdm(os.listdir(self.pickle_dir), desc='collecting fit results'):

                full_file_path = f'{self.pickle_dir}/{file}'
                with open(full_file_path, 'rb') as f:
                    dat = pickle.load(f)

                data += [Table(dat)]
                os.remove(full_file_path)

            self.collected_data = vstack(data)

        elif 'mosfit' in self.method_used:
            # TODO: implement for mosfit!
            raise NotImplementedError

        else:
            raise ValueError(f'The method {self.method_used} not known!')

        collected_data_filename = f'{self.pickle_dir}/collected_data.pkl'

        with open(collected_data_filename, 'wb') as fout:
            pickle.dump(self.collected_data, fout)

        return collected_data_filename



class ResultError(Exception):
    def __init__(self, msg):
        self.msg = msg
