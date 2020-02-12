import os
import logging
from pathlib import Path
import io
from tqdm import tqdm
import sncosmo
from numpy import loadtxt


# ========================== #
# = create a common logger = #
# ========================== #


# create a handler class to deal with tqdm progress bars
class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


main_logger_name = 'main script'
handler = logging.StreamHandler()
# format = logging.Formatter('%(name)s:\n %(levelname)s - %(message)s')
format = logging.Formatter('%(levelname)s - %(name)s: %(message)s')
handler.setFormatter(format)


def get_custom_logger(name):
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    return logger


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())


# ================================= #
# = create directory substructure = #
# ================================= #


class DirectoryError(Exception):
    def __init__(self, msg):
        self.msg = msg


es_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

root_dir = ''
for s in es_dir.split('/')[1:-2]:
    root_dir += '/'+s

try:
    es_scratch_dir = os.environ['EXPLOSION_TIME_ESTIMATION_SCRATCH_DIR']
except KeyError:
    es_scratch_dir = str(Path.home())
    raise DirectoryError('No scratch directory has been set!')
    # logger.warning("No scratch directory has been set. Using home directory as default.")

output_dir = f'{es_scratch_dir}/output'
storage_dir = f'{es_scratch_dir}/storage'
input_dir = f'{es_scratch_dir}/input'

cache_dir = f'{storage_dir}/cache'
pickle_dir = f'{storage_dir}/pickles'
log_dir = f'{storage_dir}/logs'

dh_dir = f'{storage_dir}/DataHandler'
fh_dir = f'{storage_dir}/Fitter'

simulation_dir = f'{input_dir}/simulations'
real_data_dir = f'{input_dir}/real_data'

plots_dir = f'{output_dir}/plots'
output_data = f'{output_dir}/data'


all_dirs = [output_dir, storage_dir, input_dir,
            cache_dir, pickle_dir, log_dir,
            dh_dir, fh_dir,
            simulation_dir,real_data_dir,
            plots_dir, output_data]


for dirname in all_dirs:
    if not os.path.isdir(dirname):
        logger.info("Making Directory: {0}".format(dirname))
        os.makedirs(dirname)
    else:
        logger.info("Found Directory: {0}".format(dirname))

activate_path = '/lustre/fs23/group/icecube/necker/init_anaconda3.sh'
environment_path = '/afs/ifh.de/user/n/neckerja/scratch/envs/estimate_explosion_time_env'
mosfit_environment_path = '/afs/ifh.de/user/n/neckerja/scratch/envs/mosfit_env'


# =========== #
# = methods = #
# =========== #

all_methods = ['sncosmo_chi2', 'sncosmo_mcmc', 'sncosmo_nester', 'mosfit']


# ============================================ #
# = set magnitude system and flux zero point = #
# ============================================ #
#
# magnitude_system = 'ab'
# flux_zp = 15


# ======================================= #
# = set colors for the wavelength bands = #
# ======================================= #

def bandcolors(band):

    color_dict = {
        'ztfg': 'orange',
        'ztfr': 'blue',
        'ztfi': 'green',
        'desi': 'green'
    }

    if band in color_dict.keys():
        return color_dict[band]
    else:
        for key in color_dict.keys():
            if band in key:
                return color_dict[key]

    raise Exception(f'No color specified for band {band}, type(band)={type(band)}!')


# ======================= #
# = limiting magnitudes = #
# ======================= #
#
# def get_mag_limit(band):
#
#     mag_dict = {
#         'ztfg': 21,
#         'ztfr': 21,
#         'ztfi': 21,
#         'desi': 21
#     }
#
#     if band in mag_dict.keys():
#         return mag_dict[band]
#     else:
#         return Exception(f'No limiting magnitude specified for band {band}')
