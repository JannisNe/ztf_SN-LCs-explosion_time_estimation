import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import sncosmo
import pickle
import json
import seaborn as sns
import corner
from collections import OrderedDict
from scipy.stats import kde
from scipy.optimize import curve_fit
from estimate_explosion_time.shared import plots_dir, get_custom_logger, main_logger_name, \
    pickle_dir, bandcolors


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
logging.getLogger('matplotlib').setLevel('INFO')


class Plotter:

    def __init__(self, dhandler, method):
        self.dhandler = dhandler
        self.use_indices = None
        self.dir = f'{self.get_my_root_dir()}/{method}/{self.dhandler.selection_string}'
        self.lc_dir = self.dir + '/lcs'
        self.corner_dir = self.dir + '/corners'
        self.rhandler = self.dhandler.rhandlers[method]

        self.update_dir(self.get_my_root_dir())
        self.update_dir(f'{self.get_my_root_dir()}/{method}')
        self.update_dir(self.dir)
        self.update_dir(self.corner_dir)

    def hist_t_exp_dif(self, cl):
        """

        :param cl:
        """
        logger.debug('plotting histogramm of delta t_exp')

        # logger.debug(f'rhandler data = {self.rhandler.t_exp_dif[:10]} ...')
        # logger.debug(f'indices = {self.dhandler.selected_indices}')

        tdif = np.array(self.rhandler.t_exp_dif)[self.dhandler.selected_indices]
        tdif_e = np.array(self.rhandler.t_exp_dif_error)[self.dhandler.selected_indices]

        # logger.debug(f't_dif = {tdif[:10]} ...')
        # logger.debug(f't_dif error = {tdif_e[:10]} ...')

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

        fig, ax = plt.subplots()
        plt.gca().invert_yaxis()

        if 'sncosmo' in self.rhandler.method:
            self.plot_lc_with_sncosmo_fit(indice, ax)
        elif 'mosfit' in self.rhandler.method:
            self.plot_lc_with_mosfit_fit(indice, ax)
        else:
            raise PlotterError('Jesus, what\'s that method, you crazy dog?! You\'re a dog, man!')

        if self.rhandler.collected_data:
            collected_data = self.rhandler.collected_data[indice]

            t_exp_true = collected_data['t_exp_true']
            t_exp_fit = collected_data['t_exp_fit']
            t_exp_fit_error = collected_data['t_exp_dif_error']

            ax.axvline(t_exp_true, color='red', linestyle='--', label='true $t_{exp}$')
            ax.axvline(t_exp_fit, color='red', label='fitted $t_{exp}$')
            ax.fill_between(np.array([-t_exp_fit_error / 2, t_exp_fit_error / 2]) + t_exp_fit,
                            y1=30, color='red', alpha=0.2)

        else:
            logger.info('Data from fits has not been collected! Can\'t draw explosion time estimate ...')

        ax.set_xlabel('Phase in MJD')
        ax.set_ylabel('Apparent Magnitude')

        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.lc_dir}/{indice}.pdf')
        plt.close()

    def plot_lc_with_sncosmo_fit(self, indice, ax):

        logger.debug(f'plotting lightcurve number {indice}')
        data_path = self.dhandler._sncosmo_data_
        with open(data_path, 'rb') as fin:
            data = pickle.load(fin, encoding='latin1')

        lc = data['lcs'][indice]
        id_orig = data['meta']['idx_orig'][indice]
        bands = np.unique(lc['band'])

        collected_data = self.rhandler.collected_data[indice]

        fit_results = collected_data['fit_output']
        mask = collected_data["mask"]

        if fit_results[0]['ID'] != id_orig:
            raise PlotterError(f'original ID and ID of fitted event are different: \n '
                               f'{fit_results[0]["ID"]} and {id_orig}')

        models = []
        for fit_result in fit_results[mask]:
            model = sncosmo.Model(source=fit_result['model'])
            model.set(z=fit_result['z'], t0=fit_result['t0'], amplitude=fit_result['amplitude'])
            models.append(model)

        plot_times = np.linspace(
            min(lc['time'])-20,
            max(lc['time'])+20,
            round((max(lc['time']) - min(lc['time']) + 40))
        )

        zp = np.unique(lc['zp'])
        zpsys = np.unique(lc['zpsys'])

        if (len(zp) > 1) or (len(zpsys) > 1):
            raise PlotterError('Different zeropoints and zeropoint systems!')
        else:
            zp = 15 # zp[0]
            zpsys = zpsys[0]
            sn_mag_sys = sncosmo.get_magsystem(zpsys)

        logger.debug(f'zp: {zp}, zpsys: {zpsys}')
        data = sncosmo.photdata.PhotometricData(lc)
        normed_data = data.normalized(zp=zp, zpsys=zpsys)
        ylims = [20,20]

        for band in bands:

            logger.debug(f'plotting band {band}')
            band_color = bandcolors(band)
            band_mag = {'mag': list(), 'mag_err_u': list(), 'mag_err_l': list(), 'time': list()}
            band_mag_ul = {'mag': list(), 'mag_err_u': list(), 'mag_err_l': list(), 'time': list()}

            for flux, fluxerr, time, bandpass in zip(normed_data.flux,
                                                     normed_data.fluxerr,
                                                     normed_data.time,
                                                     normed_data.band):

                if bandpass.name != band:
                    continue

                print(flux, fluxerr, time)

                if (flux > 0) and (flux/fluxerr >= 5):
                    band_mag['mag'].append(sn_mag_sys.band_flux_to_mag(flux, bandpass))
                    band_mag['mag_err_u'].append(
                        sn_mag_sys.band_flux_to_mag(flux + fluxerr, bandpass) - band_mag['mag'][-1]
                    )
                    band_mag['mag_err_l'].append(
                        sn_mag_sys.band_flux_to_mag(flux - fluxerr, bandpass) - band_mag['mag'][-1]
                    )
                    band_mag['time'].append(time)

                else:
                    band_mag_ul['mag'].append(
                        sn_mag_sys.band_flux_to_mag(
                            max([flux+fluxerr, fluxerr]),
                            bandpass)
                    )
                    band_mag_ul['mag_err_u'].append(0)
                    band_mag_ul['mag_err_l'].append(-1)
                    band_mag_ul['time'].append(time)

            ylims = [max([ylims[0], max(band_mag['mag'])]), min([ylims[1], min(band_mag_ul['mag'])])]

            # ax.plot(lc['time'][band_mask], band_mag, 'o', label=band, color=band_color)
            # ax.plot(lc['time'][band_mask], band_flux, 'o', label=band, color=band_color)
            i = 0
            for symb, dic in zip(['o', 'v'], [band_mag, band_mag_ul]):
                if len(dic['mag']) > 0:
                    ax.errorbar(dic['time'], dic['mag'], yerr=(dic['mag_err_u'], dic['mag_err_l']), color=band_color,
                                fmt=symb, label=[band, ''][i], markeredgecolor='black', markeredgewidth=1, capsize=1,
                                elinewidth=1.5, capthick=2, zorder=10
                                )
                    i += 1

            for model in models:

                model_mag = model.bandmag(band, zpsys, plot_times)
                # model_flux = model.bandflux(band, plot_times)
                ax.plot(plot_times, model_mag, color=band_color, alpha=1/len(models))
                # ax.plot(plot_times, model_flux, color=band_color, alpha=1/len(models))

        ax.set_ylim([ylims[0]+2, ylims[1]-3])

    def plot_lc_with_mosfit_fit(self, indice, ax, ylim=[20, 20], plot_corner=False,):

        json_name = f'{pickle_dir}/{self.dhandler.name}/mosfit/{indice}.json'

        sns.reset_orig()
        with open(json_name, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
            if 'name' not in data:
                data = data[list(data.keys())[0]]
        photo = data['photometry']
        model = data['models'][0]
        real_data = len([x for x in photo if 'band' in x and 'magnitude' in x and (
                'realization' not in x or 'simulated' in x)]) > 0
        ci_data = [x for x in photo if 'band' in x and 'confidence_level' in x]
        band_attr = ['band', 'instrument', 'telescope', 'system', 'bandset']
        band_list = list(set([tuple(x.get(y, '')
                                    for y in band_attr) for x in photo
                              if 'band' in x and 'magnitude' in x]))
        real_band_list = list(set([tuple(x.get(y, '')
                                         for y in band_attr) for x in photo
                                   if 'band' in x and 'magnitude' in x and (
                                           'realization' not in x or 'simulated' in x)]))

        confidence_intervals = {}
        for x in ci_data:
            if x['band'] not in confidence_intervals.keys():
                confidence_intervals[x['band']] = [[], [], []]
            confidence_intervals[x['band']][0].append(float(x['time']))
            confidence_intervals[x['band']][1].append(float(x['confidence_interval_upper']))
            confidence_intervals[x['band']][2].append(float(x['confidence_interval_lower']))

        used_bands = []
        for full_band in band_list:
            (band, inst, tele, syst, bset) = full_band

            extra_nice = ', '.join(list(filter(None, OrderedDict.fromkeys((inst, syst, bset)).keys())))
            nice_name = band + ((' [' + extra_nice + ']') if extra_nice else '')

            realizations = [[] for x in range(len(model['realizations']))]
            for ph in photo:
                rn = ph.get('realization', None)
                ci = ph.get('confidence_interval', False)
                si = ph.get('simulated', False)
                if rn and not si and not ci:
                    if tuple(ph.get(y, '') for y in band_attr) == full_band:
                        realizations[int(rn) - 1].append((
                            float(ph['time']), float(ph['magnitude']), [
                                float(ph.get('e_lower_magnitude', ph.get('e_magnitude', 0.0))),
                                float(ph.get('e_upper_magnitude', ph.get('e_magnitude', 0.0)))],
                            ph.get('upperlimit')))
            numrz = np.sum([1 for x in realizations if len(x)])

            if band in confidence_intervals.keys():
                logger.debug('plotting confidence intervals')
                ci = confidence_intervals[band]
                label = '' if full_band in used_bands or full_band in real_band_list else nice_name
                ax.fill_between(ci[0], ci[1], ci[2], color=bandcolors(band), edgecolor=None, alpha=0.3,
                                 label=label)
                if label:
                    used_bands = list(set(used_bands + [full_band]))

            for rz in realizations:
                if not len(rz):
                    continue
                logger.debug('plotting individual realizations')
                xs, ys, vs, us = zip(*rz)
                label = '' if full_band in used_bands or full_band in real_band_list else nice_name
                if max(vs) == 0.0:
                    ax.plot(xs, ys, color=bandcolors(band),
                             label=label, linewidth=0.5, alpha=0.1)
                else:
                    xs = np.array(xs)
                    ymi = np.array(ys) - np.array([np.inf if u else v[0] for v, u in zip(vs, us)])
                    yma = np.array(ys) + np.array([v[1] for v in vs])
                    ax.fill_between(xs, ymi, yma, color=bandcolors(band), edgecolor=None,
                                     label=label, alpha=1.0 / numrz, linewidth=0.0)
                    ax.plot(xs, ys, color=bandcolors(band),
                             label=label, alpha=0.1, linewidth=0.5)
                if label:
                    used_bands = list(set(used_bands + [full_band]))

            if real_data:
                for s in range(2):
                    if s == 0:
                        cond = False
                        symb = 'o'
                    else:
                        cond = True
                        symb = 'v'
                    vec = [(float(x['time']), float(x['magnitude']),
                            0.0 if 'upperlimit' in x else float(
                                x.get('e_lower_magnitude', x.get('e_magnitude', 0.0))),
                            float(x.get('e_upper_magnitude', x.get('e_magnitude', 0.0)))) for x in photo
                           if 'magnitude' in x and ('realization' not in x or 'simulated' in x) and
                           'host' not in x and 'includeshost' not in x and
                           x.get('upperlimit', False) == cond and
                           tuple(x.get(y, '') for y in band_attr) == full_band]
                    if not len(vec):
                        continue
                    xs, ys, yls, yus = zip(*vec)
                    ylim = [max([ylim[0], max(ys)+1]), min([ylim[1], min(ys)-1])]
                    label = nice_name if full_band not in used_bands else ''
                    ax.errorbar(xs, ys, yerr=(yus, yls), color=bandcolors(band), fmt=symb,
                                 label=label,
                                 markeredgecolor='black', markeredgewidth=1, capsize=1,
                                 elinewidth=1.5, capthick=2, zorder=10)
                    ax.errorbar(xs, ys, yerr=(yus, yls), color='k', fmt=symb, capsize=2,
                                 elinewidth=2.5, capthick=3, zorder=5)
                    if label:
                        used_bands = list(set(used_bands + [full_band]))

        ax.set_ylim(ylim)

        if plot_corner:
            cfig = self._plot_corners(model)
            cfig.savefig(f'{self.corner_dir}/{indice}.pdf')
            plt.close(cfig)

    @staticmethod
    def _plot_corners(model):
        """produce corner plot of posterior parameter distribution of a mosfit result"""

        logger.debug('producing corner plot')

        corner_input = []
        pars = [x for x in model['setup'] if model['setup'][x].get('kind') == 'parameter' and
                'min_value' in model['setup'][x] and 'max_value' in model['setup'][x]]
        weights = []
        for realization in model['realizations']:
            par_vals = realization['parameters']
            if 'weight' in realization:
                weights.append(float(realization['weight']))
            var_names = ['$' + ('\\log\\, ' if par_vals[x].get('log') else '') +
                         par_vals[x]['latex'] + '$' for x in par_vals if x in pars and 'fraction' in par_vals[x]]
            corner_input.append([np.log10(par_vals[x]['value']) if
                                 par_vals[x].get('log') else par_vals[x]['value'] for x in par_vals
                                 if x in pars and 'fraction' in par_vals[x]])
        weights = weights if len(weights) else None
        ranges = [0.999 for x in range(len(corner_input[0]))]
        cfig = corner.corner(corner_input, labels=var_names, quantiles=[0.16, 0.5, 0.84],
                             show_titles=True, weights=weights, range=ranges)
        return cfig

    def get_dir(self, method):
        return f'{self.get_my_root_dir()}/{method}/{self.dhandler.selection_string}/'

    @staticmethod
    def update_dir(this_dir):
        if not os.path.isdir(this_dir):
            os.mkdir(this_dir)
            logger.debug(f'making directory {this_dir}')

    def get_my_root_dir(self):
        return f'{plots_dir}/{self.dhandler.name}'


class PlotterError(Exception):
    def __init__(self, msg):
        self.msg = msg
