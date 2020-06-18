from estimate_explosion_time.shared import get_custom_logger, main_logger_name, pickle_dir
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

import argparse
from estimate_explosion_time.analyses.ZTF20abdnpdo import ZTF20abdnpdoDataHandler
from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import Fitter
from estimate_explosion_time.core.analyse_fits_from_simulation.plots import Plotter


parser = argparse.ArgumentParser()
parser.add_argument('-fn', '--force_new', type=bool, default=False)
args = parser.parse_args()

method = 'mosfit'
dh = ZTF20abdnpdoDataHandler.get_dhandler(force_new=args.force_new)
fitter = Fitter(method)

missing_indice_file = f'{pickle_dir}/{dh.name}/{fitter.method_name}/missing_indices.txt'
fitter.fit_lcs(dh, reduce_mosfit_output=False, mock_fit=True)

dh.select_and_adjust_selection_string(select_all=True)
dh.results(force=True, make_plots=False)

plotter = Plotter(dh, method)
for i in range(dh.nlcs):
    plotter.plot_lc(i, plot_corner=True, plot_orig_template=False, reduce_data=True)
