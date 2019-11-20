import sncosmo
import os
import logging
from astropy.table import Table
from estimate_explosion_time.core.analyse_fits_from_simulation.get_source_explosion_time import \
    explosion_time_dic_path
import numpy as np
import json


def get_explosion_time_from_timeseries(source):

    # if input is a path to the time series, create the corresponding sncosmo source

    if os.path.exists(source):

        logging.debug(f'{source} is a path to the SED')

        # load the sed from the file
        logging.debug('loading the SED')
        sed = Table(np.loadtxt(source), names=['phase', 'wl', 'flux'])
        phase = np.unique(sed['phase'])
        disp = np.unique(sed['wl'])
        flux = sed['flux'].reshape(len(phase), len(disp))

        #create sncosmo source
        logging.debug('making SNCosmo source')
        snc_source = sncosmo.TimeSeriesSource(phase, disp, flux)

    # in the other case, the input is (hopefully) a already registered sncosmo source
    else:
        logging.debug(f'{source} is a SNCosmo source')
        snc_source = sncosmo.get_source(source)

    # to integrate the flux, construct a complete transparent bandpass
    logging.debug('constructing transparent bandpass')
    wavelength = [snc_source.minwave(), snc_source.maxwave()]
    transmission = [1, 1]
    complete_band = sncosmo.Bandpass(wavelength, transmission, name='complete band')

    # integrate and normalize the flux at z=0 and with t0=0
    logging.debug('integrating flux of wavelengths')
    amp = 1e-1
    snc_source.set(amplitude=amp)
    step = 50000
    time = np.linspace(snc_source.minphase(), snc_source.maxphase(), step)
    flux = snc_source.bandflux(complete_band, time)
    flux = flux / max(flux)

    # guess the explosion time as the time when the flux exceeds a certain value
    t_exp = min(time[flux > 0.001])

    return t_exp


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
