import numpy as np
import matplotlib.pyplot as plt
import json
import sncosmo_utils
from time import gmtime, strftime
import pickle
import sncosmo
from astropy.table import vstack, Table
import os
import sncosmo_register_ztf_bands
from scipy import stats
from scipy.stats import kde
from scipy.optimize import curve_fit
from tqdm import tqdm


#####################################
#       SNCosmo stuff               #
#####################################


def texp_from_t0(t0, model_name, z):
    # gives the explosion time calculated by the sncosmo model parameter t0
    # Input
    #   t0, model_name and z of the object to be evaluated. t0 and z from the fit
    # Output
    #   the explosion time
    if len(t0) == 0: raise ValueError('Empty input!')
    else:
        with open('sncosmo_texp.json', 'rb') as fin:
            texp_model = json.load(fin)
        texp = np.empty(len(t0))
        for i in range(len(t0)):
            texp[i] = texp_model[model_name[i]] * (1+np.array(z[i])) + np.array(t0[i])
        return texp


def get_saved_tdif(savename, **kwargs):
    if savename is not None:
        if os.path.isfile(savename):
            with open(savename, 'rb') as fin: tabs = pickle.load(fin)
        else:
            tabs = get_tdif(**kwargs)
            with open(savename, 'wb') as f: pickle.dump(tabs, f)
    else: tabs = get_tdif(**kwargs)
    return tabs


def get_tdif(fitres, model_name=None, zfix=True, include_model=None, fit_qual_level=False, include_IDs=None):
    # gives a set of parameters of the fits read with get_tables()
    # Input
    #   fitres: the output of get_tables()
    #   model_name: if given, only fits to this model are returned. If not given, for each
    #       lightcurve the best fitting fit is returned
    #   zfix: if True, fits with fixed redshift are returned, with fitted redshift otherwise
    #   include_model: list of strings, include only these models
    #   fit_qual_level: float, specifies the intervall in red_chi2 in units of the minimal value found, up to which
    #                   other fits should be included
    #   include_IDs: list of IDs to be included, used to implement exterior cuts
    # Outout
    #   dictionary with keys tdif (texp_fit-texp_true), tdif_e (error in tdif), ID, red_chi2, model
    ind = get_ind_table(fitres, model_name=model_name, zfix=zfix, include_model=include_model,
                        fit_qual_level=fit_qual_level, include_IDs=include_IDs)
    if (len(np.unique(ind)) == 1) & (np.unique(ind)[0] == False):
        raise ValueError('No elements in selection!')
    t0_fit = texp_from_t0(np.array(fitres['t0'][ind]), np.array(fitres['model'][ind]), np.array(fitres['z'][ind]))
    t0_true = np.array(fitres['t0_true'][ind])
    outtab = fitres[ind]
    outtab['tdif'] = t0_fit - t0_true
    return outtab


def get_ind_table(table, model_name=None, zfix=True, include_model=None, fit_qual_level=None, include_IDs=None):
    # get the indeces where {model_name} was used or, if not given, where red_chi2 is minimal
    # Input
    #   table: astropy table with at least zfix, error, model (if model_name is given),
    #       red_chi2, ID
    #   model_name: if given, index where model_name was used is returned. If not, index where
    #       red_chi2 is minimal is returned
    #   zfix: get results for fits with fixed redshift if True
    #   include_model: list of strings, include only these models
    #   fit_qual_level: float, specifies the interval in red_chi2 in units of the minimal value found, up to which
    #                   other fits should be included
    #   include_IDs: list of IDs to be included, used to implement exterior cuts
    # Output
    #   the indices for the above conditions
    zind = table['zfix'] == zfix
    errind = table['error'] == False
    if model_name:
        case_ind = table['model'] == model_name
    else:
        with open('logs/NoSuccsessfulFit_log.txt', 'w') as f:
            case_ind = np.array([False]*len(table['zfix']))
            chi2 = np.array(table['red_chi2'])
            tmpID = np.array(table['ID'])
            if include_IDs is None: include_IDs = tmpID
            for IDind in [tmpID == id for id in np.unique(include_IDs)]:
                tmpind = zind & errind & IDind
                if include_model:
                    include_tmp = np.array([False]*len(chi2))
                    for itr in range(len(include_model)):
                        include_tmp = include_tmp | (table['model'] == include_model[itr])
                    tmpind = tmpind & include_tmp
                try:
                    if fit_qual_level:
                        q = [min(chi2[tmpind]) * (1 + fit_qual_level / 2 * i) for i in [-1, 1]]
                        maxmask = (chi2 <= q[1]) & (chi2 >= q[0])
                    else: maxmask = chi2 == min(chi2[tmpind])
                except ValueError:
                    f.write('lightcurve ' + str(np.unique(np.array(table['ID'][IDind]))) +
                            ' has no succsesful fit with zfix is ' + str(zfix) + '\n')
                    maxmask = np.array([True]*len(table['ID']))
                case_ind[maxmask & tmpind] = True

    ind = case_ind & zind & errind
    return ind


def include_other_fits(xtract):
    # calculates the combined values for tdif, t0_e and chi2 from the result of get_tdif()
    # when a fit_qual_level was given
    med_chi2 = []
    med_tdif = []
    med_t0e = []
    med_best_model = []

    for id in np.unique(xtract['ID']):
        IDmask = xtract['ID'] == id

        other_chi2 = xtract['red_chi2'][IDmask]
        weights = np.array(other_chi2) ** (-2) / sum(np.array(other_chi2) ** (-2))

        # med_chi2.append(np.average(other_chi2, weights=weights))
        med_chi2.append(np.median(other_chi2))

        other_tdif = xtract['tdif'][IDmask]
        # med_tdif.append(np.average(other_tdif, weights=weights))
        med_tdif.append(np.median(other_tdif))

        other_t0e = xtract['t0_e'][IDmask]
        # med_t0e.append(np.average(other_t0e, weights=weights))
        med_t0e.append(np.median(other_t0e))

        med_best_model.append(xtract['model'][IDmask][np.argmax(other_chi2)])

    return {'red_chi2': med_chi2, 'tdif': med_tdif, 't0_e': med_t0e,
            'ID': np.unique(xtract['ID']), 'model': med_best_model}


def get_tables(savename=None, *args):
    # combines the results of all lightcurve fits
    # Input:
    #   filename_base, ind, appendix: the filename of the fit result files have
    #   the name filename_base + ind + appendix
    # Output
    #   table for chi2 minimaziation fit
    #   table for mcmc fit

    if savename is not None:
        if os.path.isfile(savename):
            with open(savename, 'rb') as fin: tabs = pickle.load(fin)
        else:
            tabs = calc_tables(*args)
            with open(savename, 'wb') as f: pickle.dump(tabs, f)
    else: tabs = calc_tables(*args)
    return tabs


def calc_tables(filename_base, ind, appendix):
    if appendix == '.pkl':
        chi_list = []
        mcmc_list = []
        with open('logs/FilesNotFound_log.txt', 'w') as logf:
            for i in tqdm(ind, desc='concatenating tables'):
                try:
                    with open(filename_base+str(i)+appendix, 'rb') as f:
                        dat = pickle.load(f)
                    # for ii in range(nmethods): lists[ii] += [dat[ii]]
                    chi_list += [Table(dat[0])]
                    mcmc_list += [Table(dat[1])]
                except FileNotFoundError as err: logf.write(str(err)+'\n')
        chi_tab = vstack(chi_list)
        mcmc_tab = vstack(mcmc_list)
        if len(chi_tab)==0 and len(mcmc_tab)==0:
            raise FileNotFoundError(f'Length of concatenated tables is zero! Probably {filename_base} does\'n exist.')
        return chi_tab, mcmc_tab
    else: raise NotImplementedError('data type not implemented')


#####################################
#     main evaluate function        #
#####################################


def eval_dist(array, array_e, chi2=None, cl=0.9, filename=None, xlabel=None, title=None):
    # calculate statistical properties of an input array
    # Input
    #   array: array to be investigated
    #   cl: the confidence level used for the confidence interval
    #   filename: if given, a histogram is saved under that name
    # Outout
    #   dictionary with keys median, IC and sigma
    if len(np.array(array)) == 0:
        raise ValueError('No elements in array!')
    try:
        if chi2 is not None: tdif, tdif_e, chi2 = zip(*sorted(list(zip(array, array_e, chi2)), key=lambda x: x[0]))
        else: tdif, tdif_e = zip(*sorted(list(zip(array, array_e)), key=lambda x: x[0]))
        q = np.quantile(tdif, [0.5-cl/2,0.5,0.5+cl/2])
        tmean = q[1]
        IC = [q[0], q[2]]
        sigma = np.std(tdif)
        if filename: plot_dist(tdif, tdif_e, tmean, IC, cl, chi2=chi2, filename=filename, xlabel=xlabel, title=title)
        return {'median': tmean, 'IC': IC, 'sigma': sigma}
    except IndexError:
        print('No elements in tdif')


##########################################
#  util functions for analyzing LC cuts  #
##########################################


def weird_lcs(criterion, get_tdif_output, simulation, table, txt_filename=False, plot_filenames=None, zfix=True,
              cstd=None, N=20, plot_model=False):
    """
    Plot lightcurves (lcs)
    :param criterion: array like, the indices of the output of get_tdif() that satisfy the criterion
    :param get_tdif_output: table, the corresonding output of get_tdif()
    :param simulation: the origibnal simulation file
    :param table: the outbut of get_tables()
    :param txt_filename:
    :param plot_filenames: to be passed to savefig
    :param zfix: if True, fit with fixed redshift are analized and vv
    :param cvar: table or array, contains the custom variance for each filter
    :param N: int, how many lcs should be printed
    :param plot_model: bool, should the best fit model be drawn as well
    :return: list of figures, only if plot_filenames are not given
    """

    ID = np.array(get_tdif_output['ID'])[criterion]
    simid = [sncosmo_utils.find_index(simulation['meta']['idx_orig'], ii) for ii in ID]

    figs = []
    for i, iID in enumerate(ID[:N]):
        model_name = get_tdif_output['model'][criterion[i]]
        tab_ind = (table['ID'] == iID) & (table['model'] == model_name) & (table['zfix'] == zfix) \
                  & (table['error'] == False)
        model = sncosmo.Model(source=model_name)
        model.set(z=table['z'][tab_ind], t0=table['t0'][tab_ind],
                  amplitude=table['amplitude'][tab_ind])
        lc = simulation['lcs'][simid[i]]
        errdic = {'z': float(table['z_e'][tab_ind]), 't0': float(table['t0_e'][tab_ind]),
                  'amplitude': float(table['amplitude_e'][tab_ind])}

        figtext = '$\Delta t_{exp} = $' + f'{get_tdif_output["tdif"][criterion[i]]}'
        if cstd is not None:
            cstd_i = cstd[cstd['ID'] == iID]
            for k in cstd_i.keys()[2:]:
                if not np.isnan(float(cstd_i[k])): figtext += f'\n$cstd_{ {k} } = {float(cstd_i[k])}$'

        if plot_model:
            mdl = model
        else:
            mdl = None

        if plot_filenames is not None:
            sncosmo.plot_lc(lc, fname=plot_filenames[i], model=mdl, errors=errdic,
                            figtext=figtext)
        else:
            figs.append(sncosmo.plot_lc(lc, model=mdl, errors=errdic,
                                        figtext=figtext))
    if plot_filenames is None:
        return figs

    if txt_filename:
        with open(txt_filename, 'w') as f:
            mods, counts = np.unique(get_tdif_output["model"][criterion], return_counts=True)
            f.write('models: \n')
            [f.write(f'{mods[p]}    {counts[p]}\n') for p in range(len(mods))]
            f.write(f'red chi2: { get_tdif_output["red_chi2"][criterion] }')


def get_good_lcIDs(sim,
                   req_prepeak=None,
                   req_postpeak=None,
                   req_max_timedif=None,
                   req_std=None,
                   req_unc=None,
                   check_band='any'):
    # cuts based on the number of data points before and after observed maximum
    # Input
    #   sim: the original simulation data
    #   req_prepeak: int (same value for all bands) or dict (with the bands as keys), requested
    #               number of data points before the observed peak (in each band)
    #   req_postpeak: same as req_prepeak but for datapoints after the peak
    #   check_band: either 'any' or 'all', specify if all bands have to fulfill the requests or just any
    #               use 'all' if either of the requests is a dict
    # Output
    #   list of IDs that comply with the requests
    #   list of IDs that don't comply
    IDs = []
    cut_IDs = []

    for req in [req_prepeak, req_postpeak]:
        if req is not None:
            if type(req) is int or type(req) is dict:
                pass
            else:
                raise TypeError('Input type has to be int or dict')

    for req in [req_max_timedif, req_std, req_unc]:
        if req is not None:
            if type(req) in [int, float, dict]:
                pass
            else:
                raise (TypeError('Input type has to be int, float or dict'))

    for lc, ID in zip(sim['lcs'], sim['meta']['idx_orig']):

        bands = np.unique([lc['band']])
        band_masks = {}
        for band in bands: band_masks[band] = lc['band'] == band
        comply_prepeak = {}
        comply_postpeak = {}
        comply_max_timedif = {}
        comply_std = {}

        for i, band in enumerate(bands):

            lc_masked = lc[band_masks[band]]
            nepochs = len(lc_masked)
            peak_phase = lc_masked['time'][np.argmax(lc_masked['flux'])]


            # check for compliance with required pre peak epochs
            npre_peak = len(lc_masked[lc_masked['time'] < peak_phase])
            if req_prepeak is not None and type(req_prepeak) == dict and band in req_prepeak.keys():
                comply_prepeak[band] = npre_peak >= req_prepeak[band]
            elif req_prepeak is not None and type(req_prepeak) == int:
                comply_prepeak[band] = npre_peak >= req_prepeak
            else:
                comply_prepeak[band] = True

            # check for compliance with required post peak epochs
            npost_peak = len(lc_masked[lc_masked['time'] > peak_phase])
            if req_postpeak is not None and type(req_postpeak) == dict and band in req_postpeak.keys():
                comply_postpeak[band] = npost_peak >= req_postpeak[band]
            elif req_postpeak is not None and type(req_postpeak) == int:
                comply_postpeak[band] = npost_peak >= req_postpeak
            else:
                comply_postpeak[band] = True

            # check for compliance with required maximal time difference
            timedif = [lc_masked['time'][j + 1] - lc_masked['time'][j] for j in range(nepochs - 1)]
            if req_max_timedif is not None and len(timedif) != 0:
                max_timedif = max(timedif)
                if type(req_max_timedif) is dict and band in req_max_timedif.keys():
                    comply_max_timedif[band] = max_timedif <= req_max_timedif[band]
                elif type(req_max_timedif) is int or type(req_max_timedif) is float:
                    comply_max_timedif[band] = max_timedif <= req_max_timedif
            else:
                comply_max_timedif[band] = True

            # check for compliance with required custom standard deviation / variance
            med_flux = np.median(lc_masked['flux'])
            std = np.median((np.array(lc_masked['flux']) - med_flux) ** 2 / np.array(lc_masked['fluxerr'] ** 2))
            if req_std is not None:
                if type(req_std) is dict and band in req_std.keys():
                    comply_std[band] = std >= req_std[band]
                elif type(req_std) is int or type(req_std) is float:
                    comply_std[band] = std >= req_std
            else:
                comply_std[band] = True

        if check_band == 'any':
            if np.any(list(comply_prepeak.values())) & np.any(list(comply_postpeak.values())) & \
                    np.any(list(comply_max_timedif.values())) & np.any(list(comply_std.values())):
                IDs.append(ID)
            else:
                cut_IDs.append(ID)
        elif check_band == 'all':
            if np.all(list(comply_prepeak.values())) & np.all(list(comply_postpeak.values())) & \
                    np.all(list(comply_max_timedif.values())) & np.all(list(comply_std.values())):
                IDs.append(ID)
            else:
                cut_IDs.append(ID)
        else:
            raise TypeError(f'Input {check_band} for check_band not understood!')
    return IDs, cut_IDs


def exclude_lcs(get_tdif_output, include_IDs):
    """
    include only certain IDs
    :param get_tdif_output: table, the complete output of get_tdif()
    :param include_IDs: array like, IDs to be included
    :return: table with the form of get_tdif_output
    """
    out = get_tdif_output[ [get_tdif_output['ID'][i] in include_IDs for i in range(len(get_tdif_output))] ]
    return out


#########################
#   plot functions      #
#########################


def plot_cut(cut_tab, cuts, cl=0.9, cutfct=lambda x,y: x>y, filename=None):
    """
    Plots cuts
    :param cut_tab: table, includes ID and value for every lc and band
    :param cuts: array like, values of the cuts
    :param cl: float, confidence level for calculating the interval of confidence, default 0.9
    :param cutfct: callable, function of two variables, e.g. x<y, used to apply the cuts
    :return: figure, only if filename is not given
    """

    c = 'blue'; cperc = 'red'
    dt = {}; dtic = {}; perc = {}
    for band in cut_tab.keys()[2:]:
        if band not in dt.keys():
            dt[band] = [np.nan]*len(cuts)
            dtic[band] = [np.nan]*len(cuts)
            perc[band] = [np.nan]*len(cuts)
        for i, cut in enumerate(cuts):
            msk = cutfct(np.array(cut_tab[band]), cut)
            q = np.quantile(np.array(cut_tab['tdif'][msk]), [0.5-cl/2,0.5,0.5+cl/2])
            dt[band][i] = q[1]
            dtic[band][i] = np.array([q[0], q[2]])
            perc[band][i] = len(cut_tab[msk])/len(cut_tab)
    fig, ax = plt.subplots(len(cut_tab.keys())-2, sharex=True)
    if type(ax) not in [list, np.ndarray]: ax = [ax]
    for i, band in enumerate(cut_tab.keys()[2:]):
        if len(cuts) < 5: m = 's'
        else: m = None
        ax[i].plot(cuts, dt[band], color=c, label=f'median {band}', marker=m)
        ax[i].fill_between(cuts, y1=np.array(dtic[band]).T[0], y2=np.array(dtic[band]).T[1],
                           color=c, alpha=0.2, label=f'$IC_{ {cl} }$')
        ax[i].set_ylabel('$\Delta t_{exp}$', color=c)
        ax[i].tick_params(axis='y', labelcolor=c)
        newax = ax[i].twinx()
        newax.set_ylabel('perc.', color=cperc)
        newax.plot(cuts, perc[band], color=cperc, label='kept part', marker=m)
        newax.set_ylim([0,1])
        newax.tick_params(axis='y', labelcolor=cperc)
        newax.legend(loc='upper right')
        ax[i].legend(loc='upper left')
    ax[-1].set_xlabel('cut')

    if filename is not None: fig.savefig(filename)
    else: return fig


def plot_tdif_z(tab, filename, mode):
    # plot relation between redshift and error in texp prediction
    fig, ax = plt.subplots()
    ax.plot(tab['z'], tab['tdif'], 'k.', ms=0.5)
    ax.set_xlabel('z')
    ax.set_ylabel('$\Delta t_{exp}$')
    ax.set_yscale('log')
    ax.set_title(f'{mode} fit')
    plt.savefig(filename)


def plot_cor_with_nobs(sim, xtracted_tab, var_names, filename, hist_nobs_fn= False, log=False, abs_v=True):
    # plot the relation to the number of observed epochs with specified variables
    # Input:
    #   sim: the original simulation data
    #   xtracted_tab: the output of get_tdif()
    #   var_names: A list of variable names to be plotted against the number of observed epochs
    #   filenama: to be passes to savefig, if not given figure is returned
    #   hist_nobs_fn: if given a histogram of the nuber of observed epochs ist saved to this string
    #   log: list or bool, specifies which variable in var_names should be plotted logarithmically
    #   abs_v: list or bool, specifies of which variable in var_names the absolute value should be plotted
    # Output
    #   list containing the number of observed epochs and the corresponding number of lightcurves
    #   figure if filename is not given
    nobs = xtracted_tab['nobs']
    nobs_sort, nobs_sort_times = np.unique(nobs, return_counts=True)

    if type(log) == bool: log = [log]*len(var_names)
    if type(abs_v) == bool: abs_v = [abs_v]*len(var_names)

    fig, (axs) = plt.subplots(len(var_names), sharex=True)

    for nax, ax in enumerate(axs):
        nobs_av = np.empty(len(nobs_sort))

        if abs_v[nax]: var = abs(xtracted_tab[var_names[nax]])
        else: var = xtracted_tab[var_names[nax]]
        for i, nnobs in enumerate(nobs_sort):
            nobs_mask = nobs == nnobs
            nobs_av[i] = np.median(var[nobs_mask])

        ax.plot(nobs, var, 'k.', ms=0.5, alpha=0.1)
        ax.plot(nobs_sort, nobs_av, 'r.', ms=0.7)
        ax.set_ylabel(var_names[nax])
        ax.set_xscale("log")
        if log[nax]: ax.set_yscale('log')
        if nax == len(axs)-1: ax.set_xlabel('observed epochs')
    plt.savefig(filename)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(nobs_sort, nobs_sort_times, ds='steps-mid', label=f'mdeian = {np.median(nobs_sort_times)}')
    ax.set_xlabel('observed epochs')
    ax.set_xscale('log')
    ax.set_ylabel('number of lightcurves')
    if hist_nobs_fn:
        plt.savefig(hist_nobs_fn)
        plt.close()
    else: return fig

    return [nobs_sort, nobs_sort_times]


def plot_chi2_td(tab, filename, mode):
    # plot the relation dt - red_chi2
    fig, ax = plt.subplots()
    ax.plot(tab['t0_e'], tab['red_chi2'], 'k.', ms=0.5)
    ax.set_xlabel('$\Delta t_{exp, fit}$')
    ax.set_ylabel('$\chi^2$/doF')
    ax.set_title(f'{mode} fit')
    ax.set_xscale('log'); ax.set_yscale('log')
    plt.savefig(filename)


def plot_model_specifics(tab, filename_prefix='', only_1bc=True):
    # plot the relation model-specifics
    if only_1bc:
        model_names = ['s11-2005hl', 's11-2005hm', 's11-2006fo','s11-2006jo', 'snana-04d1la', 'snana-04d4jv',
                       'snana-2004gv', 'snana-2006ep', 'snana-2007y', 'snana-2004ib', 'snana-2005hm', 'snana-2006jo',
                       'snana-2007nc', 'nugent-sn1bc']
    else:
        with open('sncosmo_model_names.json', 'rb') as fin:
            model_names = json.load(fin)
    model_tdif = {}
    model_redchi2 = {}
    for model_name in model_names:
        xtract_i = get_tdif(tab, model_name=model_name)
        dic = eval_dist(xtract_i['tdif'], xtract_i['t0_e'], xtract_i['red_chi2'])
        if dic:
            model_tdif[model_name] = np.median(xtract_i['tdif'])
            model_redchi2[model_name] = np.median(xtract_i['red_chi2'])
        else:
            print(f'Non output for {model_name}')
    _plot_spec_(model_tdif, f'../plots/{filename_prefix}model_tdif.pdf')
    _plot_spec_(model_redchi2, f'../plots/{filename_prefix}model_redchi2.pdf')


def _plot_spec_(spec, figname=None, pltind=False):
    # plot a dict conteining model names a s keys
    # returns figure if figname is not given
    if not pltind: pltind = range(len(spec.keys()))
    fig, ax = plt.subplots()
    plotdic = dict(sorted(spec.items(), key=lambda i: i[1]))
    ax.plot(np.array(list(plotdic.keys()))[pltind], np.array(list(plotdic.values()))[pltind], ds='steps-mid')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(5)
        tick.label.set_rotation(45)
    ax.grid(b=True)
    if figname:
        plt.savefig(figname)
        plt.close()
    else: return fig


def plot_dist(tdif, tdif_e, tmean, IC,  cl, chi2=None, filename=None, xlabel=None, ylabel=None, title=None):
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
    if chi2 is not None: fig, ax = plt.subplots(nrows=3, sharex = True, gridspec_kw={'height_ratios':[3,1,1]})
    else: fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios':[3,1]})

    if not xlabel: xlabel = 'variable'

    if chi2 is not None:
        ax[2].plot(tdif, chi2, 'k.', label=f'median = {round(np.median(chi2),2)}', ms=0.5, alpha=0.5)
        ax[2].set_ylabel('$\chi^2$/doF')
        ax[2].set_yscale('log')
        ax[2].axhline(1, linestyle='--', ms=1)
        ax[2].legend()
        ax[2].set_xlabel(xlabel)
    else: ax[1].set_xlabel(xlabel)

    ax[1].plot(tdif, tdif_e, 'k.', label=f'median = {round(np.median(tdif_e), 2)}', ms=0.5, alpha=0.5)
    ax[1].set_yscale('log')
    ax[1].set_ylabel(xlabel+'$_{,err}$')
    ax[1].legend()

    ax[0].hist(tdif, bins=50)
    ax[0].axvline(tmean, c='orange', label=f'median = {round(tmean,2)}')
    ax[0].axvline(IC[0], linestyle='--', c='orange', label=f'IC$_{ {cl} }$ = {round(IC[0],2), round(IC[1],2)}')
    ax[0].axvline(IC[1], linestyle='--', c='orange')

    if ylabel: ax[0].set_ylabel(ylabel)
    else: ax[0].set_ylabel('a.u.')
    if title: ax[0].set_title(title)
    ax[0].legend()

    if filename:
        plt.savefig(filename)
        plt.close()
    else: return fig


def plot_tdif_tdife(get_tdif_output, filename=None):
    xref = np.array([1e-3, 1e2])
    yref = xref

    x = np.log(abs(np.array(get_tdif_output['tdif'])))
    y = np.log(np.array(get_tdif_output['t0_e']))
    nbins = 200

    def _fitfct_(xdat, a, b): return a*xdat+b
    pop, pcov = curve_fit(_fitfct_, x, y)

    k = kde.gaussian_kde([x, y])
    xi, yi = np.mgrid[x.min():x.max():nbins * 1j, y.min():y.max():nbins * 1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    fig, ax = plt.subplots()
    ax.plot(np.log(xref), np.log(yref), '--r', label='reference linear relation')
    ax.plot(x, pop[0]*x+pop[1], '-r', label='fitted linear relation')
    ax.contour(xi, yi, zi.reshape(xi.shape), color='k', label='contour')
    ax.set_xlim([min(x), max(x)])
    ax.set_ylim([min(y), max(y)])
    ax.plot(x, y, 'ko', ms=0.5, alpha=0.5, label='data')
    ax.set_xlabel('$log(|\Delta t_{exp}|)$')
    ax.set_ylabel('$log(t_{exp,err})$')
    ax.legend()

    if filename is not None:
        fig.savefig(filename)
    else:
        return fig
