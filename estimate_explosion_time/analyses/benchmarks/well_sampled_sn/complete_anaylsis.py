from estimate_explosion_time.shared import get_custom_logger, main_logger_name, pickle_dir
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.analyses.benchmarks.well_sampled_sn import benchmarkDH, event_get_lc_fct
from estimate_explosion_time.core.fit_data.fitlauncher.fitlauncher import Fitter
from estimate_explosion_time.core.analyse_fits_from_simulation.plots import Plotter
from estimate_explosion_time.cluster import wait_for_cluster


logger.info(f'Events: {[n for n in event_get_lc_fct]}')

# produces the pickle file and loads the DataHandler
dh = benchmarkDH.get_dhandler()

# fit the lightcurves with the desired method (only 'mosfit' is good!)
method = 'mosfit'
fitter = Fitter.get_fitter(method)

logger.debug(
    f'fitter method {fitter.method_name} \n'
    f'job-id {fitter.job_id}'
)

missing_indice_file = f'{pickle_dir}/{dh.name}/{fitter.method_name}/missing_indices.txt'
fitter.fit_lcs(dh, reduce_mosfit_output=False
               # missing_indice_file=missing_indice_file  # to be used when repeating the fit
               )

wait_for_cluster(fitter.job_id)

plotter = Plotter(dh, method)
for i in range(dh.nlcs):
    plotter.plot_lc(i, plot_corner=True)
