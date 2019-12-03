import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from scipy.stats import kde
from scipy.optimize import curve_fit
from estimate_explosion_time.shared import plots_dir, get_custom_logger, main_logger_name


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
logging.getLogger('matplotlib').setLevel('INFO')


class Plotter:

    def __init__(self, dhandler):
        self.dhandler = dhandler
        self.use_indices = None
        self.dir = self.get_my_root_dir()
        self.update_dir()

    def hist_t_exp_dif(self, rhandler, cl):
        """

        :param rhandler:
        :param cl:
        :return:
        """
        logger.debug('plotting histogramm of elta t_exp')

        tdif = np.array(rhandler.t_exp_dif)[self.dhandler.selected_indices]
        tdif_e = np.array(rhandler.t_exp_dif_error)[self.dhandler.selected_indices]

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

        logger.debug(f'saving figure under {self.dir}/t_exp_dif.pdf')

        plt.savefig(f'{self.dir}/t_exp_dif.pdf')
        plt.close()

    def plot_tdif_tdife(self, rhandler):
        """

        :param rhandler:
        :return:
        """

        logger.debug(f'plotting delta t_exp against fitting error')

        xref = np.array([1e-3, 1e2])
        yref = xref

        x = np.log(abs(np.array(rhandler.t_exp_dif)[self.dhandler.selected_indices]))
        y = np.log(np.array(rhandler.t_exp_dif_error)[self.dhandler.selected_indices])
        nbins = 200

        def fitfct(xdat, a, b): return a * xdat + b

        pop, pcov = curve_fit(fitfct, x, y)

        k = kde.gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        fig, ax = plt.subplots()
        ax.plot(np.log(xref), np.log(yref), '--r', label='reference linear relation')
        ax.plot(x, pop[0] * x + pop[1], '-r', label='fitted linear relation')
        ax.contour(xi, yi, zi.reshape(xi.shape), color='k', label='contour')
        ax.set_xlim([min(x), max(x)])
        ax.set_ylim([min(y), max(y)])
        ax.plot(x, y, 'ko', ms=0.5, alpha=0.5, label='data')
        ax.set_xlabel('$log(|\Delta t_{exp}|)$')
        ax.set_ylabel('$log(t_{exp,err})$')
        ax.legend()

        logger.debug(f'figure saved to {self.dir}/difference_error_relation.pdf')

        plt.savefig(f'{self.dir}/difference_error_relation.pdf')
        plt.close()

    def update_dir(self):
        if not os.path.isdir(self.dir):
            os.mkdir(self.dir)
        logger.debug(f'directory for Plotter is {self.dir}')

    def get_my_root_dir(self):
        return f'{plots_dir}/{self.dhandler.name}'
