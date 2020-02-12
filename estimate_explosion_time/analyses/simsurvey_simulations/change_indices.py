from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

import os
from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import Fitter
from estimate_explosion_time.core.data_prep.data import DataHandler

error = False

dh = DataHandler.get_dhandler('simsurvey_simulation')
fi = Fitter.get_fitter('mosfit')
fi_outdir = fi.get_output_directory(dh)

for diri in [fi_outdir]:
    for file in os.listdir(diri):
        if not file.startswith('.') and not 'missing' in file and not file.startswith('new__'):
            old_ind = int(file.split('.')[0])
            ending = file.split('.')[1]
            new_ind = old_ind + 1
            logger.info(f'renaming {file} to indice {new_ind}')
            os.rename(diri + '/' + file, f'{diri}/new__{new_ind}.{ending}')

    if not error:
        for file in os.listdir(diri):
            if file.startswith('new__'):
                fn = file.split('new__')[1]
                logger.info(f'renaming {file}')
                os.rename(diri + '/' + file, diri + '/' + fn)
    else:
        raise Exception('sth went wrong')
