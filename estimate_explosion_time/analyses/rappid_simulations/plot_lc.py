from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)

from estimate_explosion_time.core.analyse_fits_from_simulation.plots import Plotter
from estimate_explosion_time.analyses.rappid_simulations import rappidDH
from estimate_explosion_time.shared import get_custom_logger, main_logger_name
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('ind', type=int, nargs='+')
parser.add_argument('model', type=str)
args = parser.parse_args()



logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
dh = rappidDH.get_dhandler(args.model)



plotter = Plotter(dh, 'mosfit')
for ind in args.ind:
    plotter.plot_lc(ind)
