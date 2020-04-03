from estimate_explosion_time.shared import get_custom_logger, main_logger_name, test_plots_dir, storage_dir, bandcolors
import logging

logger = get_custom_logger(main_logger_name)
logger_level = logging.DEBUG
logger.setLevel(logger_level)
logger.debug('logging level is DEBUG')

from estimate_explosion_time.analyses.simsurvey_simulations import SimsurveyDH
from estimate_explosion_time.core.analyse_fits_from_simulation.get_source_explosion_time.find_explosion_time import \
    get_explosion_time, ExplosionTimeError, bolometric_bandpass
import pickle
import os
import numpy as np
import sncosmo
from tqdm import tqdm
from matplotlib import pyplot as plt
import multiprocessing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--force', type=bool, default=False)
args = parser.parse_args()

raw = f'{test_plots_dir}/test_explosion_time_estimation_from_template'

if not os.path.exists(raw):
    os.mkdir(raw)

storage_file = f'{storage_dir}/test_explosion_time_estimation_from_template.npy'
cl = 0.9
time_steps = 5000

# load the data handler
dh = SimsurveyDH.get_dhandler()

# load the data
logger.debug('loading data')
with open(dh.get_data('sncosmo'), 'rb') as f:
    sndata = pickle.load(f, encoding='latin1')

meta_data = sndata['meta']
lcs = sndata['lcs']

ind = []
itr = range(dh.nlcs)


def multiprocess_function(i):
    """
    just a function for multiprocessing
    passes the meta data and peak mjd on to estimate_explosion_time and returns the result
    that's done once for the case weher we know in which band the maximum occurs and once where we assume
    the maximum refers to the bolometric flux
    """

    logger.debug(f'{i}/{dh.nlcs - 1}')
    lc = lcs[i]
    max_ind = np.argmax(lc['flux'])
    peak_band = lc['band'][max_ind]

    model = sncosmo.Model('nugent-sn1bc')
    model.set(z=meta_data['z'][i], amplitude=1e-11, t0=meta_data['t0'][i])
    time = np.linspace(model.mintime(), model.maxtime(), time_steps)

    bandpass_band = sncosmo.get_bandpass(peak_band)
    bandflux_band = model.bandflux(bandpass_band, time)
    peak_mjd_band = time[np.argmax(bandflux_band)]

    bandpass_bol = bolometric_bandpass(model)
    bandflux_bol = model.bandflux(bandpass_bol, time)
    peak_mjd_bol = time[np.argmax(bandflux_bol)]

    estimated_explosion_time = get_explosion_time(
        'nugent-sn1bc',
        redshift=meta_data['z'][i],
        peak_mjd=peak_mjd_bol,
        band=None
    )

    estimated_explosion_time_with_band = get_explosion_time(
        'nugent-sn1bc',
        redshift=meta_data['z'][i],
        peak_mjd=peak_mjd_band,
        band=peak_band
    )

    return i, estimated_explosion_time, estimated_explosion_time_with_band


if args.force or not os.path.isfile(storage_file):
    # estimate the explosion times

    logger.debug('estimating explosion times')
    logger.debug('starting multiprocessing')

    with multiprocessing.Pool(30) as p:

        if logger.getEffectiveLevel() != logging.INFO:
            estimated_explosion_times = p.map(multiprocess_function, itr)

        else:
            estimated_explosion_times = list()
            pbar_length = len(itr)
            with tqdm(total=pbar_length, desc='calculating explosion times') as pbar:
                for i, _ in enumerate(p.imap_unordered(multiprocess_function,itr)):
                    estimated_explosion_times.append(_)
                    pbar.update()

    estimated_explosion_times = np.array(estimated_explosion_times,
                                         dtype={'names': ['indice', 'without_band', 'with_band'],
                                                'formats': ['<f8', '<f8', '<f8']})

    logger.debug(f'saving results to {storage_file}')
    np.save(storage_file, estimated_explosion_times)

else:
    # get the previously saved estimated explosion times
    logger.debug(f'loading previously estimated explosion times from {storage_file}')
    estimated_explosion_times = np.load(storage_file)


explosion_time_dif = estimated_explosion_times.copy()

# visualize the results
for bandstr in ['with_band', 'without_band']:

    this_dir = f'{raw}/{bandstr}'
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)

    logger.debug(bandstr)
    logger.debug('calculating difference between estimated and true explosion time')

    # calculate the error in the estimation
    explosion_time_dif[bandstr][np.argsort(explosion_time_dif['indice'])] -= meta_data['t0']

    # plot the distribution of the error

    arr = explosion_time_dif[bandstr]
    ic_lower, tmean, ic_upper = np.quantile(arr, [0.5 - cl / 2, 0.5, 0.5 + cl / 2])

    logger.debug(f'tmean = {tmean}, ic = [{ic_lower:.2f}, {ic_upper:.2f}]')

    fig, ax = plt.subplots()

    ax.hist(arr)
    ax.axvline(ic_lower, color='orange' ,ls='--', label=fr'IC$_{ {cl} }$ = [{ic_lower:.1f}, {ic_upper:.1f}]')
    ax.axvline(ic_upper, color='orange', ls='--')
    ax.axvline(tmean, color='orange', label=f'median = {tmean:.3f}')

    ax.legend()

    ax.set_xlabel(r'$\Delta t_{\mathrm{explosion}}$ [d]')
    ax.set_title(r'$\Delta t_{\mathrm{explosion}}$ ' + f'{bandstr.strip("_band")} provinding peak band')
    plt.tight_layout()
    fname = f'{this_dir}/estimation_error_distribution.pdf'
    logger.debug(f'saving figure under {fname}')
    fig.savefig(fname)
    plt.close()


    # plot the 20 worst cases
    sorted_arr = explosion_time_dif[np.argsort(abs(explosion_time_dif[bandstr]))]
    logger.debug(f'sorted explosion time estimation errors: {sorted_arr[bandstr]}')

    for i in sorted_arr['indice'][-20:]:

        i = int(i)

        prev_texp = estimated_explosion_times[bandstr][estimated_explosion_times['indice'] == i]

        lc = lcs[i]
        max_ind = np.argmax(lc['flux'])
        peak_band = lc['band'][max_ind]
        band = None if 'without' in bandstr else peak_band
        peak_mjd = lc['time'][max_ind]
        plot_lc = lc if 'without' in bandstr else lc[lc['band'] == peak_band]

        # find true peak time
        model = sncosmo.Model('nugent-sn1bc')
        model.set(z=meta_data['z'][i], amplitude=1e-11, t0=meta_data['t0'][i])
        bandpass = bolometric_bandpass(model) if not band else sncosmo.get_bandpass(band)
        time = np.linspace(model.mintime(), model.maxtime(), time_steps)
        error = (max(time) - min(time))/time_steps
        bandflux = model.bandflux(bandpass, time)
        peak_mjd = time[np.argmax(bandflux)]

        texp_from_template, t, f = get_explosion_time('nugent-sn1bc',
                                                      band=band,
                                                      peak_mjd=peak_mjd,
                                                      redshift=meta_data['z'][i],
                                                      full_output=True)

        if texp_from_template-prev_texp != 0:
            raise ExplosionTimeError('got different result this time!')

        # logger.debug(f'\ntexp_true = {float(meta_data["t0"][i]):.1f}'
        #              f'\ntexp_prev = {float(prev_texp):.1f}'
        #              f'\ntexp_from_template = {float(texp_from_template):.1f}'
        #              f'\ntdif = {float(explosion_time_dif[bandstr][explosion_time_dif["indice"] == i]):.2f}')

        fig, ax = plt.subplots()
        color = bandcolors(band)
        ax.plot(t, f, label='lc used in estimation', color=color, ls='-.')

        for plot_band in np.unique(plot_lc['band']):
            band_lc = plot_lc[plot_lc['band'] == plot_band]
            ax.errorbar(band_lc['time'], band_lc['flux'] / max(band_lc['flux']),
                        yerr=band_lc['fluxerr'] / max(band_lc['flux']),
                        color=bandcolors(plot_band), ls='', marker='o', label='simulated data ' + plot_band)

        ax.axvline(meta_data['t0'][i], ls='--', color=color, label=r'$t_{exp,true}$')
        ax.axvline(texp_from_template, ls=':', color=color, label=r'$t_{exp,from template}$')
        ax.axvline(prev_texp, ls='', color=color,
                   label=r'$t_{dif} = $' +
                         f'{texp_from_template - meta_data["t0"][i]:.1f}' +
                         r' $\pm$ ' +
                         f'{error:.2f}'
                   )

        ax.set_xlabel('phase in days')
        ax.set_ylabel('flux')
        title = f"{bandstr.strip('_band')} providing peak band"
        if band:
            title += f"\nflux in {band}"
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        fname = f'{this_dir}/{i}.pdf'
        logger.debug(f'saving figure under {fname}')
        fig.savefig(fname)
        plt.close()
