import os
import logging
from pathlib import Path


# ================================= #
# = create directory substructure = #
# ================================= #

es_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

root_dir = ''
for s in es_dir.split('/')[1:-2]:
    root_dir += '/'+s

try:
    es_scratch_dir = os.environ['EXPLOSION_TIME_ESTIMATION_SCRATCH_DIR']
except KeyError:
    es_scratch_dir = str(Path.home())
    logging.warning("No scratch directory has been set. Using home directory as default.")

output_dir = f'{es_scratch_dir}/output'
storage_dir = f'{es_scratch_dir}/storage'
input_dir = f'{es_scratch_dir}/input'

cache_dir = f'{storage_dir}/cache'
pickle_dir = f'{storage_dir}/pickles'
log_dir = f'{storage_dir}/logs'
dh_dict_dir = f'{storage_dir}/DataHandler_dict'

simulation_dir = f'{input_dir}/simulations'
real_data_dir = f'{input_dir}/real_data'

plots_dir = f'{output_dir}/plots'

all_dirs = [output_dir, storage_dir, input_dir,
            cache_dir, pickle_dir, log_dir, dh_dict_dir,
            simulation_dir,real_data_dir, plots_dir]


for dirname in all_dirs:
    if not os.path.isdir(dirname):
        logging.info("Making Directory: {0}".format(dirname))
        os.makedirs(dirname)
    else:
        logging.info("Found Directory: {0}".format(dirname))

activate_path = '/lustre/fs23/group/icecube/necker/init_anaconda3.sh'
environment_path = '/afs/ifh.de/user/n/neckerja/scratch/envs/estimate_explosion_time_env'
mosfit_environment_path = '/afs/ifh.de/user/n/neckerja/scratch/envs/mosfit_env'
