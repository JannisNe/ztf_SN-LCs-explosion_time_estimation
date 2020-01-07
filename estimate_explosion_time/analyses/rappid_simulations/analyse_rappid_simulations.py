from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.shared import simulation_dir
from estimate_explosion_time.core.data_prep.data import DataHandler
from estimate_explosion_time.cluster import n_tasks


methods = all_methods[:-1]
simulation_name = 'rappid_sim_model03'

rappid03_path = '/afs/ifh.de/user/n/neckerja/scratch/ZTF_20190512/ZTF_MSIP_MODEL03.pkl'

rappid03DH = DataHandler.get_dhandler(simulation_name, rappid03_path)