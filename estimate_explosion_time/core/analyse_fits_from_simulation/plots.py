import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from estimate_explosion_time.shared import plots_dir, get_custom_logger, main_logger_name


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
logging.getLogger('matplotlib').setLevel('INFO')


class Plotter:

    def __init__(self, dhandler):
        self.dhandler = dhandler
        self.dir = self.get_my_root_dir()
        self.update_dir()

    def hist_t_exp_dif(self, rhandler, cl):

        # def plot_dist(tdif, tdif_e, tmean, IC, cl, chi2=None, filename=None, xlabel=None, ylabel=None, title=None):
            # plot the distribution of a variable (namely Delta t0) and optionally its relation to
            # red_chi2
            # Input
            #   filename: filename to be passed to savefig, if not given figure is returned
            #   tdif, tmean, IC: the variable to be plotted with its mean an confidence interval
            #   cl: the confidence level of the confidence interval
            #   xlabel, ylabel, title: to be passed to pyplot
            #   chi2: if given the relation tdif-chi2 is plotted
            # Output
            #   figure if filename is not given

        self.dir = f'{self.get_my_root_dir()}/{rhandler.method}'
        self.update_dir()

        tdif = rhandler.t_exp_dif
        tdif_e = rhandler.t_exp_dif_error

        ic = [0, 0]
        ic[0], tmean, ic[1] = np.quantile(tdif, [0.5-cl/2, 0.5, 0.5+cl/2])

        fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        xlabel = f'median($\Delta t$)'

        ax[1].set_xlabel(xlabel)
        ax[1].plot(tdif, tdif_e, 'k.', label=f'median = {round(np.median(tdif_e), 2)}', ms=0.5, alpha=0.5)
        ax[1].set_yscale('log')
        ax[1].set_ylabel(xlabel + '$_{,err}$')
        ax[1].legend()

        ax[0].hist(tdif, bins=50)
        ax[0].axvline(tmean, c='orange', label=f'median = {round(tmean,2)}')
        ax[0].axvline(ic[0], linestyle='--', c='orange', label=f'IC$_{ {cl} }$ = {round(ic[0],2), round(ic[1],2)}')
        ax[0].axvline(ic[1], linestyle='--', c='orange')
        ax[0].set_ylabel('a.u.')
        ax[0].set_title(f'{self.dhandler.name}\n{rhandler.method}')
        ax[0].legend()

        plt.savefig(f'{self.dir}/t_exp_dif.pdf')
        plt.close()

    def update_dir(self):
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        logger.debug(f'directory for Plotter is {self.dir}')

    def get_my_root_dir(self):
        return f'{plots_dir}/{self.dhandler.name}'
