import sncosmo
import os
import logging
from astropy.table import Table
import numpy as np
import json
from estimate_explosion_time.shared import get_custom_logger, main_logger_name


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())


find_explosion_time_dir = os.path.dirname(os.path.realpath(__file__))
explosion_time_dic_path = f'{find_explosion_time_dir}/explosion_times.json'


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


def get_explosion_time_from_template(template, band=None, peak_mjd=None, redshift=0,
                                     steps=5000, flux_threshold=0.01, full_output=False):
    """
    Get the explosion time from a spectral time evolution template.
    This will be the time when the flux reaches the flux_threshold.
    :param template: str, path to the template file or name of SNCosmo source
    :param band: str, opt,  spectral bandpass name to which the peak date refers.
                            As default the bolometric flux is used
    :param peak_mjd: float, peak MJD of the lightcurve
    :param rredshift: float, redshift
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

    # construct bandpass
    # Use a completely transparent bandpass if no band is given
    transmission = [1, 1]
    if not band:
        src_wavelength = [source.minwave(), source.maxwave()]
        src_bandpass = sncosmo.Bandpass(src_wavelength, transmission, name='complete band')
    else:
        src_bandpass = sncosmo.get_bandpass(band)

    # set arbitrary amplitude
    amp = 1e-1
    source.set(amplitude=amp)

    # get time and flux, normalize the flux
    time = np.linspace(source.minphase(), source.minphase() + 50, steps)
    flux = source.bandflux(src_bandpass, time)
    flux = flux / max(flux)

    # get the explosion time as phase when flux crosses flux threshold
    # as this is done without any redshift involved this refers to the restframe
    tstart = min(time[flux > flux_threshold])
    logger.debug('template starts at ' + str(tstart))
    tstart_rel = tstart / (max(time) - min(time))
    logger.debug(f'relative to the whole template time that\'s {tstart_rel:.3f}')

    # initialize SNCosmo Model in order to set the redshift
    # this now refers to the observer frame
    model = sncosmo.Model(source=source)
    logger.debug('setting redshift to ' + str(redshift))
    model.set(z=redshift, t0=0)

    # as the redshift shifts the spectral range it's necesarry to make a new complete bandpass for the shifted model
    if not band:
        mdl_wavelength = [model.minwave(), model.maxwave()]
        mdl_bandpass = sncosmo.Bandpass(mdl_wavelength, transmission, name='complete band')
    else:
        mdl_bandpass = sncosmo.get_bandpass(band)

    # get the time array and flux in the observer frame
    redtime = np.linspace(model.mintime(), model.maxtime(), steps)
    redshifted_flux = model.bandflux(mdl_bandpass, redtime)
    redshifted_flux = redshifted_flux / max(redshifted_flux)

    # shift the time array so the maximum flux is at peak MJD
    if peak_mjd:
        logger.debug('shift time array')
        redtime = redtime - redtime[np.argmax(redshifted_flux)] + peak_mjd

    # determine explosion time
    t_exp_pre = min(redtime[redshifted_flux > 0.01])

    # if explosion time in template was after t=0 assume it is a theoretical model and that t=0 is the explosion time
    # IS THIS VALID ?!?!?!?!
    t_exp = t_exp_pre - tstart_rel * (max(redtime) - min(redtime)) * (1 + redshift) if tstart > 0 else t_exp_pre

    if peak_mjd and (t_exp > peak_mjd):
        raise ExplosionTimeError('Explosion time {:.2f} after peak {:.2f}'.format(t_exp, peak_mjd))

    logger.debug('exploded {:.2f}d before peak'.format(peak_mjd - t_exp))

    if full_output:
        return t_exp, redtime, redshifted_flux
    else:
        return t_exp


# def get_explosion_time_from_timeseries(source, redshift=None, peak_mjd=None):
#     """
#     Estimates the explosion time from a spectral time series
#     :param source: str, path to SED or sncosmo source name
#     :return: float, explosion time
#     """
#
#     # if input is a path to the time series, create the corresponding sncosmo source
#
#     if os.path.exists(source):
#
#         logging.debug(f'{source} is a path to the SED')
#
#         # load the sed from the file
#         logging.debug('loading the SED')
#         sed = Table(np.loadtxt(source), names=['phase', 'wl', 'flux'])
#         phase = np.unique(sed['phase'])
#         disp = np.unique(sed['wl'])
#         flux = sed['flux'].reshape(len(phase), len(disp))
#
#         #create sncosmo source
#         logging.debug('making SNCosmo source')
#         snc_source = sncosmo.TimeSeriesSource(phase, disp, flux)
#
#     # in the other case, the input is (hopefully) a already registered sncosmo source
#     else:
#         logging.debug(f'{source} is a SNCosmo source')
#         snc_source = sncosmo.get_source(source)
#
#     # to integrate the flux, construct a complete transparent bandpass
#     logging.debug('constructing transparent bandpass')
#     wavelength = [snc_source.minwave(), snc_source.maxwave()]
#     transmission = [1, 1]
#     complete_band = sncosmo.Bandpass(wavelength, transmission, name='complete band')
#
#     # integrate and normalize the flux at z=0 and with t0=0
#     logging.debug('integrating flux of wavelengths')
#     amp = 1e-1
#     snc_source.set(amplitude=amp)
#     step = 50000
#     time = np.linspace(snc_source.minphase(), snc_source.maxphase(), step)
#     flux = snc_source.bandflux(complete_band, time)
#     flux = flux / max(flux)
#
#     if (not redshift) and (not peak_mjd):
#         # guess the explosion time as the time when the flux exceeds a certain value
#         t_exp = min(time[flux > 0.001])
#
#         return t_exp
#
#     else:
#         snc_source.set(redshift=redshift)
        

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
