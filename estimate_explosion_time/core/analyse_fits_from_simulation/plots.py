import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import sncosmo
import pickle
import json
import seaborn as sns
import corner
import warnings
from collections import OrderedDict
from scipy.stats import kde
from scipy.optimize import curve_fit
from estimate_explosion_time.shared import plots_dir, get_custom_logger, main_logger_name, \
    pickle_dir, bandcolors
from estimate_explosion_time.core.analyse_fits_from_simulation.get_source_explosion_time.find_explosion_time import \
    get_explosion_time
from estimate_explosion_time.core.fit_data.mosfit.reduce_mosfit_result_file import make_reduced_output

warnings.simplefilter('ignore', matplotlib.MatplotlibDeprecationWarning)

logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
logging.getLogger('matplotlib').setLevel('INFO')


class Plotter:

    def __init__(self, dhandler, method, directory=None):
        self.dhandler = dhandler
        self.use_indices = None
        self.dir = f'{self.get_my_root_dir()}/{method}/{self.dhandler.selection_string}' if dhandler \
            else directory if directory \
            else input('Dude, where\'s my directory? ')
        self.lc_dir = self.dir + '/lcs'
        self.corner_dir = self.dir + '/corners'
        self.rhandler = self.dhandler.rhandlers[method] if dhandler else None

        # try creating default directory structure
        # if this fails, assume Plotter is used without a DataHandler and ask for directory
        if dhandler:
            self.update_dir(self.get_my_root_dir())
            self.update_dir(f'{self.get_my_root_dir()}/{method}')
            self.update_dir(self.dir)
            self.update_dir(self.corner_dir)
            self.update_dir(self.lc_dir)

        else:
            logger.warning('No DataHandler!')
            self.update_dir(self.dir)
            self.update_dir(self.corner_dir)
            self.update_dir(self.lc_dir)

    def hist_t_exp_dif(self, cl):
        """

        :param cl:
        """
        logger.debug('plotting histogramm of delta t_exp')

        tdif_with_nans = np.array(self.rhandler.t_exp_dif)[self.dhandler.selected_indices]
        tdif = tdif_with_nans[~np.isnan(tdif_with_nans)]
        tdif_e = np.array(self.rhandler.t_exp_dif_error)[self.dhandler.selected_indices]
        tdif_e = tdif_e[~np.isnan(tdif_with_nans)]

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

        if self.dhandler.selected_indices:
            ax[0].plot([], [], ' ', label='{:.2f}% of all LCs'.format(
                len(self.dhandler.selected_indices)*100/self.dhandler.nlcs))

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
        x = x[~np.isnan(x)]
        y = np.log(np.array(self.rhandler.t_exp_dif_error)[self.dhandler.selected_indices])
        y = y[~np.isnan(y)]
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

    def hist_ic90_deviaton(self):
        """

        :return:
        """

        tdif = np.array(self.rhandler.t_exp_dif)[self.dhandler.selected_indices]
        tdif_minus_ic = self.rhandler.tdif_minus_ic[self.dhandler.selected_indices]
        tdif_minus_ic = tdif_minus_ic[~np.isnan(tdif)]

        if np.any(np.isnan(tdif_minus_ic)):
            raise Exception  # TODO: make nice Exception

        fig, ax = plt.subplots()
        ax.hist(tdif_minus_ic, bins=50)
        ax.plot([], [], ' ',
                label=f'{len(tdif_minus_ic[tdif_minus_ic == 0]) * 100 / len(tdif_minus_ic):.2f}% contained in IC90')
        ax.set_ylabel('a.u.')
        ax.set_xlabel(r'deviation to IC$_{90}$')
        ax.set_title(f'{self.dhandler.name}\n{self.rhandler.method}')
        ax.legend()

        fname = f'{self.dir}/deviation_ic.pdf'
        logger.debug(f'saving figure under {fname}')

        plt.savefig(fname)
        plt.close()

    def plot_lcs_where_fit_fails(self, dt=5, n=10, **kwargs):

        # dt_indices = np.where(abs(self.rhandler.t_exp_dif[self.dhandler.selected_indices]) >= dt)[0]
        # dt_indices = np.array(self.dhandler.selected_indices)[
        #     abs(self.rhandler.t_exp_dif[self.dhandler.selected_indices]) >= dt
        # ]

        tdif = np.array(self.rhandler.t_exp_dif)[self.dhandler.selected_indices]

        dt_indices = np.array(self.dhandler.selected_indices)[~np.isnan(tdif)][
            abs(self.rhandler.tdif_minus_ic[self.dhandler.selected_indices][~np.isnan(tdif)]) != 0
        ]

        for indice in dt_indices[:n]:
            logger.debug(f'plotting lightcurve with indice {indice}')
            self.plot_lc(indice, f'{self.lc_dir}/bad_lcs', **kwargs)

    def plot_lc(self, indice, plot_in_dir='default', plot_orig_template=True, **kwargs):
        """

        Plot the data and fit results

        :param indice: int, indice of the event in the data
        :param plot_in_dir: str, the directory where to save the plot, defaults to self.lc_dir
        :param plot_orig_template: bool, if True the original temaplet will be plottes
        :param kwargs: to be passed self.to plot_lc_with_mosfit_fit

        """

        if plot_in_dir == 'default':
            plot_in_dir = self.lc_dir

        if not os.path.isdir(plot_in_dir):
            logger.info(f'making directory {plot_in_dir}')
            os.mkdir(plot_in_dir)

        with open(self.dhandler.get_data('sncosmo'), 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        original_id = data['meta']['idx_orig'][indice]
        neutrino_time = data['meta']['neutrino_time'][indice] if 'neutrino_time' in data['meta'] else None

        logger.debug(f'plotting {original_id}')

        fig, ax = plt.subplots()
        plt.gca().invert_yaxis()

        if 'sncosmo' in self.rhandler.method:
            raise DeprecationWarning
            # self.plot_lc_with_sncosmo_fit(indice, ax)
        elif 'mosfit' in self.rhandler.method:
            self.plot_lc_with_mosfit_fit(indice, ax, **kwargs)
        else:
            raise PlotterError('Jesus, what\'s that method, you crazy dog?! You\'re a dog, man!')

        if not isinstance(self.rhandler.collected_data, type(None)):
            collected_data = self.rhandler.collected_data[indice]

            t_exp_true = collected_data['t_exp_true']
            t_exp_fit = collected_data['t_exp_fit']
            t_exp_ic = collected_data['t_exp_dif_0.9']
            tdif = collected_data['t_exp_dif']

            ax.axvline(t_exp_fit, color='red', label='fitted $t_{{exp}}=$' + f'{t_exp_fit:.2f}')

            if not np.isnan(t_exp_true):
                ax.plot([], [], ' ', label=r'$t_{dif} = $' + '{:.1f}'.format(tdif))
                ax.axvline(t_exp_true, color='red', linestyle='--', label='true $t_{exp}$')
                ic = np.array(t_exp_ic) + t_exp_true
            else:
                ic = t_exp_ic
            ax.fill_between(ic,
                            y1=30, color='red', alpha=0.2,
                            label=r'IC$_{90\%}=$' + f'[{ic[0]:.2f}, {ic[1]:.2f}]')

            if neutrino_time:
                ax.axvline(neutrino_time, color='gold', linestyle='--', label=r't$_{\nu}$=' + f'{neutrino_time:.2f}')

        else:
            logger.info('Data from fits has not been collected! Can\'t draw explosion time estimate ...')

        if plot_orig_template:
            peak_band, peak_mag = self.dhandler.get_peak_band(indice, meta_data=data['meta'])
            _ = get_explosion_time(self.dhandler.get_spectral_time_evolution_file(indice),
                                   peak_band=peak_band,
                                   peak_mjd=data['meta']['t_peak'][indice],
                                   redshift=data['meta']['z'][indice],
                                   full_output=True)

            time, flux = _[1], _[2]

            magsys = sncosmo.get_magsystem('AB')
            mags = np.array([magsys.band_flux_to_mag(f, peak_band) if f > 0 else np.nan for f in flux])
            mags += peak_mag - min(mags)

            ax.plot(time, mags, ls='-.', color=bandcolors(peak_band), label=f'original template {peak_band}')

            ax.plot(data['meta']['t_peak'][indice], peak_mag,
                    color=bandcolors(peak_band), marker='x', label=f'peak')

        ax.set_title(original_id)
        ax.set_xlabel('Phase in MJD')
        ax.set_ylabel('Apparent Magnitude')

        plt.legend()
        plt.tight_layout()
        fn = f'{plot_in_dir}/{indice}.pdf'
        logger.debug(f'saving to {fn}')
        plt.savefig(fn)
        plt.close()

    # def plot_lc_with_sncosmo_fit(self, indice, ax):
    #
    #     logger.warning('THIS PART OF THE CODE IS NOT MAINTAINED!!')
    #     logger.debug(f'plotting lightcurve number {indice}')
    #     data_path = self.dhandler._sncosmo_data_
    #     with open(data_path, 'rb') as fin:
    #         data = pickle.load(fin, encoding='latin1')
    #
    #     lc = data['lcs'][indice]
    #     id_orig = data['meta']['idx_orig'][indice]
    #     bands = np.unique(lc['band'])
    #
    #     collected_data = self.rhandler.collected_data[indice]
    #
    #     fit_results = collected_data['fit_output']
    #     mask = collected_data["mask"]
    #
    #     if fit_results[0]['ID'] != id_orig:
    #         raise PlotterError(f'original ID and ID of fitted event are different: \n '
    #                            f'{fit_results[0]["ID"]} and {id_orig}')
    #
    #     models = []
    #     for fit_result in fit_results[mask]:
    #         model = sncosmo.Model(source=fit_result['model'])
    #         model.set(z=fit_result['z'], t0=fit_result['t0'], amplitude=fit_result['amplitude'])
    #         models.append(model)
    #
    #     plot_times = np.linspace(
    #         min(lc['time'])-20,
    #         max(lc['time'])+20,
    #         round((max(lc['time']) - min(lc['time']) + 40))
    #     )
    #
    #     zp = np.unique(lc['zp'])
    #     zpsys = np.unique(lc['zpsys'])
    #
    #     if (len(zp) > 1) or (len(zpsys) > 1):
    #         raise PlotterError('Different zeropoints and zeropoint systems!')
    #     else:
    #         zp = 15 # zp[0]
    #         zpsys = zpsys[0]
    #         sn_mag_sys = sncosmo.get_magsystem(zpsys)
    #
    #     logger.debug(f'zp: {zp}, zpsys: {zpsys}')
    #     data = sncosmo.photdata.PhotometricData(lc)
    #     normed_data = data.normalized(zp=zp, zpsys=zpsys)
    #     ylims = [20,20]
    #
    #     for band in bands:
    #
    #         logger.debug(f'plotting band {band}')
    #         band_color = bandcolors(band)
    #         band_mag = {'mag': list(), 'mag_err_u': list(), 'mag_err_l': list(), 'time': list()}
    #         band_mag_ul = {'mag': list(), 'mag_err_u': list(), 'mag_err_l': list(), 'time': list()}
    #
    #         for flux, fluxerr, time, bandpass in zip(normed_data.flux,
    #                                                  normed_data.fluxerr,
    #                                                  normed_data.time,
    #                                                  normed_data.band):
    #
    #             if bandpass.name != band:
    #                 continue
    #
    #             print(flux, fluxerr, time)
    #
    #             if (flux > 0) and (flux/fluxerr >= 5):
    #                 band_mag['mag'].append(sn_mag_sys.band_flux_to_mag(flux, bandpass))
    #                 band_mag['mag_err_u'].append(
    #                     sn_mag_sys.band_flux_to_mag(flux + fluxerr, bandpass) - band_mag['mag'][-1]
    #                 )
    #                 band_mag['mag_err_l'].append(
    #                     sn_mag_sys.band_flux_to_mag(flux - fluxerr, bandpass) - band_mag['mag'][-1]
    #                 )
    #                 band_mag['time'].append(time)
    #
    #             else:
    #                 band_mag_ul['mag'].append(
    #                     sn_mag_sys.band_flux_to_mag(
    #                         max([flux+fluxerr, fluxerr]),
    #                         bandpass)
    #                 )
    #                 band_mag_ul['mag_err_u'].append(0)
    #                 band_mag_ul['mag_err_l'].append(-1)
    #                 band_mag_ul['time'].append(time)
    #
    #         ylims = [max([ylims[0], max(band_mag['mag'])]), min([ylims[1], min(band_mag_ul['mag'])])]
    #
    #         # ax.plot(lc['time'][band_mask], band_mag, 'o', label=band, color=band_color)
    #         # ax.plot(lc['time'][band_mask], band_flux, 'o', label=band, color=band_color)
    #         i = 0
    #         for symb, dic in zip(['o', 'v'], [band_mag, band_mag_ul]):
    #             if len(dic['mag']) > 0:
    #                 ax.errorbar(dic['time'], dic['mag'], yerr=(dic['mag_err_u'], dic['mag_err_l']), color=band_color,
    #                             fmt=symb, label=[band, ''][i], markeredgecolor='black', markeredgewidth=1, capsize=1,
    #                             elinewidth=1.5, capthick=2, zorder=10
    #                             )
    #                 i += 1
    #
    #         for model in models:
    #
    #             model_mag = model.bandmag(band, zpsys, plot_times)
    #             # model_flux = model.bandflux(band, plot_times)
    #             ax.plot(plot_times, model_mag, color=band_color, alpha=1/len(models))
    #             # ax.plot(plot_times, model_flux, color=band_color, alpha=1/len(models))
    #
    #     ax.set_ylim([ylims[0]+2, ylims[1]-3])

    def plot_lc_with_mosfit_fit(self, indice, ax, ylim=[20, 20], plot_corner=False, **kwargs):

        json_name = f'{pickle_dir}/{self.dhandler.name}/mosfit/{indice}.json'

        self.mosfit_plot(file=json_name, ax=ax, ylim=ylim, **kwargs)

        if plot_corner:
            cfig = self.plot_corners(json_name)
            cfig_fn = f'{self.corner_dir}/{indice}.pdf'
            logger.debug(f'saving under {cfig_fn}')
            cfig.savefig(cfig_fn)
            plt.close(cfig)

    @staticmethod
    def mosfit_plot(file, ax=None, fig=None, ylim=[20, 20], reduce_data=False):

        if not ax and not fig:
            logger.debug('neither axes nor figure given')
            fig, ax = plt.subplots()
            plt.gca().invert_yaxis()
        elif fig and not ax:
            logger.debug('only figure given')
            ax = fig.add_subplot()

        sns.reset_orig()

        logger.debug(f'type of file is {type(file)}')
        if type(file) is str:
            logger.debug(f'file is {file}.')
            if not os.path.isfile(file):
                logger.warning('Not a file path!')

        if type(file) is str and os.path.isfile(file):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())
                if 'name' not in data:
                    data = data[list(data.keys())[0]]
        else:
            data = file

        if reduce_data:
            data = make_reduced_output(data)

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

            logger.debug(f'plotting {band}')

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

            rz_mask = [False if not len(rz) else True for rz in realizations]

            if np.any(rz_mask):
                logger.debug('plotting individual realizations')
                for rz in np.array(realizations)[rz_mask]:

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
                    ys_wo_nans = np.array(ys)
                    ys_wo_nans = ys_wo_nans[~np.isnan(ys_wo_nans)]
                    ylim = [max([ylim[0], max(ys_wo_nans) + 1]), min([ylim[1], min(ys_wo_nans) - 1])]
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

        return fig, ax

    @staticmethod
    def plot_corners(file):
        """produce corner plot of posterior parameter distribution of a mosfit result"""

        logger.debug('producing corner plot')

        if type(file) is str and os.path.isfile(file):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())
                if 'name' not in data:
                    data = data[list(data.keys())[0]]

        else:
            data = file

        model = data['models'][0]

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
        cfig = corner.corner(corner_input, labels=var_names, quantiles=[0.05, 0.5, 0.95],
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
