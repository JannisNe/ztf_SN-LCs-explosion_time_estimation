import sncosmo
import os
import logging
from astropy.table import Table
import numpy as np
import json
import scipy.optimize as opt
from estimate_explosion_time.shared import get_custom_logger, main_logger_name


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())


find_explosion_time_dir = os.path.dirname(os.path.realpath(__file__))
explosion_time_dic_path = f'{find_explosion_time_dir}/explosion_times.json'


def shock_breakout(dt, a1, a2):
    return a1 * dt ** 1.6 /(np.exp(a2 * dt ** 0.5) - 1)


def black_body_expansion(dt, a3):
    return a3 * dt ** 2


def lightcurve_model(t, t0, a1, a2, a3):
    delta_t = t - t0

    F = [0 if dt <= 0 else
         shock_breakout(dt, a1, a2) +
         black_body_expansion(dt, a3)
         for dt in delta_t]

    return np.array(F)


def lightcurve_model_flux(t0, a1, a2, a3):
    def f(t):
        return lightcurve_model(t-t0, t0=t0, a1=a1, a2=a2, a3=a3)
    return f


def get_texp_from_model_fit(time, flux, flux_selection=(0.01, 0.8)):

    peak_ind = np.argmax(flux)
    logger.debug(f'peak_ind={peak_ind}, peak_time={time[peak_ind]}')
    pre_peak_time = time[:peak_ind]
    pre_peak_flux = flux[:peak_ind]
    logger.debug(f'len(time)={len(pre_peak_time)}, len(flux)={len(pre_peak_flux)}')

    maxtime_ind = max(np.where(pre_peak_flux < flux_selection[1])[0])
    mintime_ind = min(np.where(pre_peak_flux > flux_selection[0])[0])
    logger.debug(f'mintime_ind={mintime_ind}, maxtime_ind={maxtime_ind}')
    selected_inds = range(0, maxtime_ind + 1)

    selected_flux, selected_time = pre_peak_flux[selected_inds], pre_peak_time[selected_inds]

    guess_a = [0.0001, 0.01, 0.001]
    guess_t0 = [pre_peak_time[mintime_ind]]
    guess = guess_t0 + guess_a
    bounds = np.array([(min(selected_time) - 5, max(selected_time)),
                       (0, 1e4), (-1e2, 1e2), (0, 1e3)])

    try:
        best, cov = opt.curve_fit(lightcurve_model, selected_time, selected_flux,
                                  p0=guess, maxfev=100000, bounds=bounds.T)
        logger.debug(f'best fit parameters: {best}')
    except ValueError as e:
        print(f'bounds: {bounds}')
        print(f'guess: {guess}')
        raise ExplosionTimeError(e)

    return best, lightcurve_model_flux(*best)


def get_texp_from_flux_threshold(time, flux, flux_threshold=0.05):
    return min(time[flux > flux_threshold])


def get_explosion_time_from_template(template, band=None, peak_band=None, peak_mjd=None, redshift=0,
                                     steps=5000, full_output=False,
                                     method='threshold', **kwargs):
    """
    Get the explosion time from a spectral time evolution template.
    This will be the time when the flux reaches the flux_threshold.

    :param template: str, path to the template file or name of SNCosmo source

    :param band: str, opt, spectral bandpass that is used in explosion time estimation.
    As default the bolometric flux is used

    :param peak_band: str, opt,  spectral bandpass name to which the peak date refers.
    As default the bolometric flux is used

    :param peak_mjd: float, peak MJD of the lightcurve

    :param redshift: float, redshift

    :param steps: int, time steps used

    :param flux_threshold: float, flux threshold used to define the explosion time

    :param full_output: bool, if true the time and flux arrays used for the explosion time estimation is also returned

    :return: float, explosion time
             if full_output=True:
                ndarray, time array
                ndarray, flux array
    """

    # initialize SNCosmo Source from path or from built-in SNCosmo Source
    source = sncosmo_source(template) if '/' in template else sncosmo.get_source(template)

    # set arbitrary amplitude
    amp = 1e-1
    source.set(amplitude=amp)

    # get time and flux, normalize the flux
    # src_bandpass = bolometric_bandpass(source) if not band else sncosmo.get_bandpass(band)
    # time = np.linspace(source.minphase(), source.minphase() + 50, steps)
    # flux = source.bandflux(src_bandpass, time)
    # flux = flux / max(flux)

    # get the explosion time as phase when flux crosses flux threshold
    # as this is done without any redshift involved this refers to the restframe
    # tstart = min(time[flux > flux_threshold])
    # logger.debug('template starts at ' + str(tstart))
    # tstart_rel = tstart / (max(time) - min(time))
    # logger.debug(f'relative to the whole template time that\'s {tstart_rel:.3f}')

    # initialize SNCosmo Model in order to set the redshift
    # this now refers to the observer frame
    model = sncosmo.Model(source=source)
    logger.debug('setting redshift to ' + str(redshift))
    model.set(z=redshift, t0=0)

    mdl_peak_bandpass = bolometric_bandpass(model) if not peak_band else sncosmo.get_bandpass(peak_band)
    mdl_threshold_bandpass = mdl_peak_bandpass if peak_band and not band else \
        bolometric_bandpass(model) if not band and not peak_band else \
        sncosmo.get_bandpass(band)

    # get the time array and flux (used for finding where he threshold is exceeded) in the observer frame
    redtime = np.linspace(model.mintime(), model.maxtime(), steps)
    redshifted_flux = model.bandflux(mdl_threshold_bandpass, redtime)
    redshifted_flux = redshifted_flux / max(redshifted_flux)

    # get the time array and flux (used to shift the time array so peak flux occurs at peak mjd) in the observer frame
    redshifted_flux_peak_band = model.bandflux(mdl_peak_bandpass, redtime)
    redshifted_flux_peak_band = redshifted_flux_peak_band / max(redshifted_flux_peak_band)

    # shift the time array so the maximum flux is at peak MJD
    if peak_mjd:
        logger.debug('shift time array')
        redtime = redtime - redtime[np.argmax(redshifted_flux_peak_band)] + peak_mjd

    # determine explosion time
    if method == 'threshold':
        method_output = get_texp_from_flux_threshold(redtime, redshifted_flux, **kwargs)
        t_exp_pre = method_output

    elif method == 'model_fit':
        method_output = get_texp_from_model_fit(redtime, redshifted_flux, **kwargs)
        t_exp_pre = method_output[0][0]

    else:
        raise ExplosionTimeError(f'Method {method} unknown!')

    # if explosion time in template was after t=0 assume it is a theoretical model and that t=0 is the explosion time
    # IS THIS VALID ?!?!?!?!
    # t_exp = t_exp_pre - tstart_rel * (max(redtime) - min(redtime)) * (1 + redshift) if tstart > 0 else t_exp_pre
    t_exp = t_exp_pre

    if peak_mjd and (t_exp > peak_mjd):
        raise ExplosionTimeError('Explosion time {:.2f} after peak {:.2f}'.format(t_exp, peak_mjd))

    logger.debug('exploded {:.2f}d before peak'.format(peak_mjd - t_exp))

    if full_output:
        src_bandpass = bolometric_bandpass(source) if not band else sncosmo.get_bandpass(band)
        time = np.linspace(source.minphase(), source.minphase() + 50, steps)
        flux = source.bandflux(src_bandpass, time)
        flux = flux / max(flux)
        return t_exp, redtime, redshifted_flux, time, flux, model, method_output

    else:
        return t_exp


def bolometric_bandpass(sncosmo_source_or_model):
    """
    create a sncosmo.Bandpass instance that has transmission of 1
    for the whole spectral range od the model or the source given
    :param sncosmo_source_or_model:
    :return:
    """

    transmission = [1, 1]
    src_wavelength = [sncosmo_source_or_model.minwave(), sncosmo_source_or_model.maxwave()]
    src_bandpass = sncosmo.Bandpass(src_wavelength, transmission, name='complete band')
    return src_bandpass


def sncosmo_source(source_path):
    """
    create a sncosmo.TimeSeriesSource from a file
    :param source_path: str, path to the file
    :return: sncosmo.TimeSeriesSource
    """

    # read the file
    sed = Table(np.loadtxt(source_path), names=['phase', 'wl', 'flux'])

    # extract phase, wavelength and flux
    phase = np.unique(sed['phase'])
    disp = np.unique(sed['wl'])
    flux = sed['flux'].reshape(len(phase), len(disp))

    # create the TimeSeriesSource
    snc_source = sncosmo.TimeSeriesSource(phase, disp, flux, zero_before=True)

    return snc_source


def get_explosion_time(source, **explosion_time_kwargs):
    """
    Get the explosion time from a spectral time series source
    :param source: str, SNCosmo source name or path to Spectral time series
    :return: float, explosion time relative to the source's t_0
    """

    # if no extra information is given, this will get the explosion time at redshift zero and
    # relative to the models t0. This value can therefore be stored in a dictionary.
    if (not 'redshift' in explosion_time_kwargs) and (not 'peak_mjd' in explosion_time_kwargs):

        with open(explosion_time_dic_path, 'r') as f:
            explosion_time_dict = json.load(f)

        # if the source has no entry already in the explosion time dictionary, get it and add it to the dictionary
        if not source in explosion_time_dict.keys():
            logger.debug(f'getting explosion time for {source}')
            t_exp = get_explosion_time_from_template(source)
            explosion_time_dict[source] = t_exp

            with open(explosion_time_dic_path, 'w') as f:
                json.dump(explosion_time_dict, f)

        return explosion_time_dict[source]

    # if info is given, no entry can be made in the dictionary because for each redshift and peak_mjd
    # that will give a different result.
    else:
        return get_explosion_time_from_template(source, **explosion_time_kwargs)


class ExplosionTimeError(Exception):
    def __init__(self, msg):
        self.msg = msg


if __name__ == '__main__':

    logging.info(f'removing old explosion time dictionary!')
    if os.path.isfile(explosion_time_dic_path):
        os.remove(explosion_time_dic_path)
    else:
        logging.info(f'dictionary didn\'t exist!!')

    logging.info(f'creating empty dictionary in its place under {explosion_time_dic_path}')
    empty_dict = {}
    with open(explosion_time_dic_path, 'w') as f:
        json.dump(empty_dict, f)
