import os
import logging
from pathlib import Path


# ================================= #
# = create directory substructure = #
# ================================= #

es_dir = os.path.dirname(os.path.realpath(__file__)) + "/"

try:
    es_scratch_dir = os.environ['EXPLOSION_TIME_ESTIMATION_SCRATCH_DIR']
except KeyError:
    es_scratch_dir = str(Path.home())
    logging.warning("No scratch directory has been set. Using home directory as default.")

output_dir = f'{es_scratch_dir}/output'
storage_dir = f'{es_scratch_dir}/storage'
input_dir = f'{es_scratch_dir}/input'

simulation_dir = f'{input_dir}/simulations'
real_data_dir = f'{input_dir}/real_data'

plots_dir = f'{output_dir}/plots'