from estimate_explosion_time.shared import get_custom_logger, main_logger_name, all_methods
import logging

logger = get_custom_logger(main_logger_name)
logger.setLevel(logging.DEBUG)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.shared import simulation_dir, es_scratch_dir
from estimate_explosion_time.core.data_prep.data import DataHandler
from estimate_explosion_time.cluster import n_tasks
from estimate_explosion_time.analyses.dummy import DummyDH
from estimate_explosion_time.core.analyse_fits_from_simulation.plots import Plotter
import pickle
import matplotlib.pyplot as plt
import numpy as np


method = 'mosfit'
simulation_name = 'dummy'

simsurvey_path = f'{es_scratch_dir}/../dummy.pkl'

dh = DummyDH.get_dhandler()
dh.select_and_adjust_selection_string(
    # req_prepeak=2,
    # req_std=None,
    # check_band='any',
    # req_texp_dif=['mosfit', 10],
    # print_selected_indices=True
)

logger.debug(
    f'Name: {dh.name} \n'
    f'ResultHandler: {dh.rhandlers.keys()} \n'
    f'rhandlers: \n '
)

for rhandler in dh.rhandlers.values():
    logger.debug(
        f'method {rhandler.method} \n'
        f'job ID {rhandler.job_id} \n'
        f'tasks left {n_tasks(rhandler.job_id)}'
    )

logger.debug('selection: ' + dh.selection_string)

input('continue? ')

with open(dh.get_data('sncosmo'), 'rb') as f:
    data = pickle.load(f, encoding='latin1')

dh.results(method)
plotter = Plotter(dh, method)

meta = data['meta']
t0_true_true = np.array(meta['t0_true'])
t0_true_estimated = np.array(meta['t0'])
t0_fit = np.array([dh.rhandlers[method].collected_data[i]['t_exp_fit'] for i in range(dh.nlcs)])

fig = plt.figure(figsize=[5.85, 3.6154988341868854 * 3])
axs = fig.subplots(nrows=3)

axs[0].hist(t0_true_true - t0_true_estimated, label=r'true true - true estimated')
axs[0].legend()

axs[1].hist(t0_true_true - t0_fit, label=r'true true - fit')
axs[1].legend()

axs[2].hist(t0_true_estimated - t0_fit, label=r'true estimated - fit')
axs[2].legend()
axs[2].set_xlabel(r'$\Delta$t')

figname = f'{plotter.dir}/delta_ts.pdf'
logger.debug(f'saving figure under {figname}')
fig.savefig(figname)
plt.close()
