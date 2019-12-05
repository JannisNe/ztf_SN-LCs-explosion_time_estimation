import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import sncosmo
import pickle
from scipy.stats import kde
from scipy.optimize import curve_fit
from estimate_explosion_time.shared import plots_dir, get_custom_logger, main_logger_name


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
logging.getLogger('matplotlib').setLevel('INFO')


class Plotter:

    def __init__(self, dhandler, method):
        self.dhandler = dhandler
        self.use_indices = None

        self.update_dir(self.get_my_root_dir())
        self.update_dir(f'{self.get_my_root_dir()}/{method}')
        self.update_dir(f'{self.get_my_root_dir()}/{method}/{self.dhandler.selection_string}')

        self.dir = f'{self.get_my_root_dir()}/{method}/{self.dhandler.selection_string}'
        self.lc_dir = self.dir + '/lcs'
        self.rhandler = self.dhandler.rhandlers[method]

    def hist_t_exp_dif(self, cl):
        """

        :param cl:
        """
        logger.debug('plotting histogramm of delta t_exp')

        tdif = np.array(self.rhandler.t_exp_dif)[self.dhandler.selected_indices]
        tdif_e = np.array(self.rhandler.t_exp_dif_error)[self.dhandler.selected_indices]

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
        ax[0].set_title(f'{self.dhandler.name}\n{self.rhandler.method}')
        ax[0].legend()

        logger.debug(f'saving figure under {self.dir}/t_exp_dif.pdf')

        plt.savefig(f'{self.dir}/t_exp_dif.pdf')
        plt.close()

    def plot_tdif_tdife(self):
        """

        """

        logger.debug(f'plotting delta t_exp against fitting error')

        xref = np.array([1e-3, 1e2])
        yref = xref

        x = np.log(abs(np.array(self.rhandler.t_exp_dif)[self.dhandler.selected_indices]))
        y = np.log(np.array(self.rhandler.t_exp_dif_error)[self.dhandler.selected_indices])
        nbins = 200

        def fitfct(xdat, a, b): return a * xdat + b

        pop, pcov = curve_fit(fitfct, x, y)

        k = kde.gaussian_kde([x, y])
        xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        fig, ax = plt.subplots()
        ax.plot(np.log(xref), np.log(yref), '--r', label='reference linear relation')
        ax.plot(x, pop[0] * x + pop[1], '-r', label='fitted linear relation')
        ax.contour(xi, yi, zi.reshape(xi.shape))
        ax.set_xlim([min(x), max(x)])
        ax.set_ylim([min(y), max(y)])
        ax.plot(x, y, 'ko', ms=0.5, alpha=0.5, label='data')
        ax.set_xlabel('$log(|\Delta t_{exp}|)$')
        ax.set_ylabel('$log(t_{exp,err})$')
        ax.legend()

        logger.debug(f'figure saved to {self.dir}/difference_error_relation.pdf')

        plt.savefig(f'{self.dir}/difference_error_relation.pdf')
        plt.close()

    def plot_lcs_where_fit_fails(self, dt, n):

        dt_indices = np.where(abs(self.rhandler.t_exp_dif[self.dhandler.selected_indices]) >= dt)[0]

        for indice in dt_indices[:n]:
            self.plot_lc(indice)

    def plot_lc(self, indice):

        if not os.path.isdir(self.lc_dir):
            logger.info(f'making directory {self.lc_dir}')
            os.mkdir(self.lc_dir)

        if 'sncosmo' in self.rhandler.method:
            self.plot_lc_with_sncosmo_fit(indice)
        elif 'mosfit' in self.rhandler.method:
            self.plot_lc_with_mosfit_fit(indice)
        else:
            raise PlotterError('Jesus, what\'s that method, you crazy dog?! You\'re a dog, man!')

    def plot_lc_with_sncosmo_fit(self, indice):

        logger.debug(f'plotting lightcurve number {indice}')
        data_path = self.dhandler._sncosmo_data_
        with open(data_path, 'rb') as fin:
            data = pickle.load(fin, encoding='latin1')

        lc = data['lcs'][indice]
        id_orig = data['meta']['idx_orig'][indice]

        fit_results = self.rhandler.collected_data[indice]['fit_output']
        mask = self.rhandler.collected_data[indice]["mask"]

        if fit_results[0]['ID'] != id_orig:
            raise PlotterError(f'original ID and ID of fitted event are different: \n '
                               f'{fit_results[0]["ID"]} and {id_orig}')

        # logger.debug(f'{mask} {fit_results}')
        # logger.debug(f'{fit_results[mask]}')

        models = []
        for fit_result in fit_results[mask]:
            model = sncosmo.Model(source=fit_result['model'])
            model.set(z=fit_result['z'], t0=fit_result['t0'], amplitude=fit_result['amplitude'])
            models.append(model)

        sncosmo.plot_lc(lc, fname=f'{self.lc_dir}/{indice}.pdf', model=models,
                        model_label=fit_results['model'][mask])

    def plot_lc_with_mosfit_fit(self, indice):
        pass

    def get_dir(self, method):
        return f'{self.get_my_root_dir()}/{method}/{self.dhandler.selection_string}/'

    @staticmethod
    def update_dir(this_dir):
        if not os.path.isdir(this_dir):
            os.mkdir(this_dir)
        logger.debug(f'directory for Plotter is {this_dir}')

    def get_my_root_dir(self):
        return f'{plots_dir}/{self.dhandler.name}'


class PlotterError(Exception):
    def __init__(self, msg):
        self.msg = msg
