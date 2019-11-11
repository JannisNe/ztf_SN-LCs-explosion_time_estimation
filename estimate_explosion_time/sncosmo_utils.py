"""
Version: 1
Author: JR
Last edited: 2019-07-23
"""

import pandas
import pickle
from astropy.io import ascii
from astropy.table import Table
import os
import sncosmo
import numpy as np
from numpy import where, array, ndarray, empty, sqrt, argmax, inf, argsort, sort, amax, amin, mean, std, log10, loadtxt
import matplotlib.pyplot as plt
import matplotlib
import sncosmo_register_ztf_bands
import json
from tqdm import tqdm_notebook
import get_texp_from_spectral_timeseries


def read_transient(dic, ID, ignore_zero_flux=True, bands=['ztfg', 'ztfr'], ignore_negative_flux=True, snr=5.0):
    """ reads one transient from a dictionary like this:
    {'ID': [list or numpy.ndarray(1-D) of int],
     'lcs': [list of astropy tables or numpy.ndarrays],
     ...}
    by ID, and returns the light curve object as a pandas dataframe.
    Always use this function (at least on experimental data) to be safe.
    Only returns data for photometric bands specified in 'bands' list
        (default=['ztfg', 'ztfr']).
    """
    # find out the index of ID:
    index = find_index(dic, ID)
    # get light curve object from dictionary:
    lc = dic['lcs'][index]
    
    # ignore photopoints with zero fluxerr:
    lc = lc[lc['fluxerr'] != 0.]
    
    # get rid of infs:
    lc = lc[lc['fluxerr'] != inf]
    # (because someone will always put at least one inf into your data...)
    
    # ignore datapoints with exactly zero flux:
    if ignore_zero_flux:
        lc = lc[lc['flux'] != 0.]
    
    # ignore datapoints with negative flux:
    if ignore_negative_flux:
        lc = lc[lc['flux'] > 0.]
    
    # significance cut:
    lc = lc[lc['flux']/lc['fluxerr'] >= snr]
        
    # this looks a bit clumsy, but can accept astropy tables as well as numpy.ndarrays
    return_dic = {
        'time': list(lc['time']),
        'band': list(lc['band']),
        'flux': list(lc['flux']),
        'fluxerr': list(lc['fluxerr']),
        'zp': list(lc['zp']),
        # 'zpsys': list(lc['zpsys']), # not really needed
    }
    
    # dictionary -> dataframe
    lc_df = pandas.DataFrame(return_dic)
    
    # filter by band: get boolean bitmask as numpy.(nd)array:
    bits = lc_df.isin({'band': bands})['band'].values
    # (because sometimes there's data from other experiments)
    
    # return filtered dict:
    return lc_df[bits]


def bandfilter(DF, filtr):
    """ Selects photopoints matching a specific band from a light curve
    in form of a pandas dataframe.
    """
    return (DF[(DF.band == filtr)])


def find_index(dic, ID):
    """ Finds out the index (position) of ID in dic.
    """
    index = None
    if type(dic) == ndarray:
        index = where(dic == ID)[0]
        if len(index) == 0:
            error_message = 'ID '+str(ID)+' not found!'
            raise KeyError(error_message)
        if len(index)>1:
            error_message = 'ID '+str(ID)+' occurs multiple times!'
            raise KeyError(error_message)
        index = int(index) # convert from array
    if type(dic) == list:
        if dic.count(ID) > 1:
            error_message = 'ID '+str(ID)+' occurs multiple times!'
            raise KeyError(error_message)
        index = dic.index(ID) # index() will raise an error by itself if not found
    if not index and index !=0: raise NotImplementedError(f'Input type not implemented: {type(dic)}, {type(index)}, {index}')
    return index


def find_index_list(dic, ID_list):
    """
    Finds the indices (positions)
    :param dic: numpy array or list
    :param ID_list: list like
    :return: list of indices
    """
    indexs = []
    for ID in ID_list: indexs.append(find_index(dic,ID))
    return indexs


def write_pkl_to_csv(filename, fileout_name=None, folder_suffix='', indices=None, encoding='latin1', add_columns=None):
    if not fileout_name:
        fileout_name = filename.split('.p',1)[0] + '.csv'
    with open(filename, 'rb') as fin:
        datain = pickle.load(fin, encoding=encoding)
        if 'lcs' in datain.keys():
            lcs = datain['lcs']
            meta = datain['meta']
        else: raise KeyError('Lightcurves not in keys of input dictionary')
    if not indices: indices = range(len(lcs))
    foldername = fileout_name[:-4] + folder_suffix
    if not os.path.exists(foldername): os.mkdir(foldername)
    for ind in tqdm_notebook(indices, desc='LCs', leave=False):
        lc = Table(lcs[ind])
        lc['band'][lc['band'] == 'desi'] = 'ztfi'
        fname = f'{foldername}/{ind}.csv'
        if add_columns:
            for col in add_columns:
                if col not in lc.keys():
                    lc[col] = [add_columns[col]]*len(lc)
                    if col == 'name': lc[col] = [f'{ind}']*len(lc)
                    if col == 'redshift': lc[col] = [meta['z'][ind]]*len(lc)
                    if col == 'ebv': lc[col] = [meta['hostebv'][ind]]*len(lc)
                    if col == 'ID': lc[col] = [int(meta['idx_orig'][ind])]*len(lc)
                    if col == 'lumdist': lc[col] = [meta['lumdist'][ind]]*len(lc)
                else: raise IndexError(f'Column {col} already exists!')
        with open(fname, 'w') as fout:
            ascii.write(lc, fout)


def plot_model_ts_and_lc(model_name, dir=None, band='ztfg', norm=None, linscale=1):
    # plot the spectral time series and the LC in a band of a SNCosmo model
    # Input
    #   model_name: str, model name of the SNCosmo model
    #   dir: str, optional, directory where to save the figures, if not given list of figures is returned
    #   band: str, band in which the lightcurve will be plotted
    # Output
    #   if dir is not given a list of two figures will be returned, first one containing the spectral TS,
    #   second one the LC

    if type(model_name) is str:
        with open('/Users/jannisnecker/Documents/Uni/masterthesis/software/sncosmo_texp.json') as d:
            model_times = json.load(d)
        alt_mintime = model_times[model_name]

        source = sncosmo.get_source(model_name)
        source.zero_before = True
    elif type(model_name) is sncosmo.models.TimeSeriesSource:
        source = model_name
        model_name = source.name
        source.zero_before = True
        alt_mintime = get_texp_from_spectral_timeseries.get_texp(source)[1]
    else:
        raise ValueError(
            f'Input type for model name has to be str or sncosmo.models.TimeSeriesSource, not {type(model_name)}')

    model = sncosmo.Model(source)

    figs = []

    nsteps = source.maxphase() - source.minphase() * 10
    w = np.linspace(source.minwave(), source.maxwave(), nsteps)
    p = np.linspace(source.minphase(), source.maxphase(), nsteps)
    flux = source.flux(p, w)
    phase, wave = np.meshgrid(p, w)


    fig1, ax = plt.subplots()
    # if norm is not None:
    norm = matplotlib.colors.SymLogNorm(linthresh=flux.max()/norm, vmin=flux.min(), vmax=flux.max(), linscale=linscale)
    pcm = ax.pcolor(phase, wave, flux, cmap='Greys_r', ec=None, lw=0, norm=norm)
    fig1.colorbar(pcm, ax=ax, extend='max')
    ax.axvline(alt_mintime, linestyle='--', color='orange', label='explosion time')

    bands = ['ztfg', 'ztfr', 'ztfi']
    for k, iband in enumerate(bands):
        obj = sncosmo.get_bandpass(iband)
        wave = obj.wave[obj.trans > 0.005]
        minmaxwave = [wave[0], wave[-1]]
        minmaxphase = [source.minphase(), source.maxphase()]
        ax.fill_between(minmaxphase, y1=minmaxwave[0], y2=minmaxwave[1],
                        color=['blue', 'green', 'yellow'][k], alpha=0.1, label=iband)

    ax.set_xlabel('phase in days')
    ax.set_ylabel('wavelength in angstrom')
    ax.set_title(f'spectral time evolution\n{model_name}')
    ax.legend()
    ax.grid(False)

    if dir is not None:
        try: os.mkdir(dir)
        except FileExistsError: pass
        fig1.savefig(f'{dir}/{model_name}_spectr_timeev.pdf')
    else: figs.append(fig1)

    # plot lightcurve
    amp = 1e-1
    mintime = source.minphase()
    model.set(z=0, t0=0, amplitude=amp)
    step = 10000
    time = np.linspace(mintime, source.maxphase(), step)
    flux = model.bandflux(band, time)

    fig2, ax = plt.subplots()
    ax.plot(time, flux / amp)
    ax.axvline(mintime, linestyle='--', label='model mintime')
    # if 'nugent' not in model_name:
    #     ax.axvline(alt_mintime, linestyle='--', color='orange', label='explosion time')
    plt.legend()
    ax.set_xlabel('phase in days')
    ax.set_ylabel('flux in a.u.')
    ax.set_title(f'flux in {band}\n{model_name}')

    if dir is not None: fig2.savefig(f'{dir}/{model_name}_{band}_lightcurve.pdf')
    else:
        figs.append(fig2)
        return figs

    plt.close()


def get_source_from_file(filename):
    """
    constructs a sncosmo.TimeSeriesSource from an external SED
    :param filename: path to SED file
    :return: sncosmo.TimeSeriesSource
    """
    sed = Table(np.loadtxt(filename), names=['phase', 'wl', 'flux'])
    phase = np.unique(sed['phase'])
    disp = np.unique(sed['wl'])
    flux = sed['flux'].reshape(len(phase), len(disp))
    source = sncosmo.TimeSeriesSource(phase, disp, flux)
    return source