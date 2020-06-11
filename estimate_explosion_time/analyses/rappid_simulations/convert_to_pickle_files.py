from estimate_explosion_time.analyses.rappid_simulations.read_light_curves_from_snana_fits \
    import read_light_curves_from_snana_fits_files as read_lc
from estimate_explosion_time.shared import get_custom_logger, main_logger_name, TqdmToLogger, es_scratch_dir
import logging
import tqdm
from astropy.table import Table
import pickle


logger = get_custom_logger(__name__)
logger.setLevel(logging.getLogger(main_logger_name).getEffectiveLevel())
tqdm_deb = TqdmToLogger(logger, level=logging.DEBUG)
tqdm_info = TqdmToLogger(logger, level=logging.INFO)


rappid_original_data = '/afs/ifh.de/group/amanda/scratch/neckerja/ZTF_20190512'


def rappid_pkl_name(model_number, peak_mag=None):
    base_name = rappid_original_data + '/ZTF_MSIP_MODEL'

    name = f'{base_name}{model_number}' if type(model_number) is str else f'{base_name}{model_number:02d}'
    if peak_mag:
        name += f'_peakmag{peak_mag}'
    name += '.pkl'

    return name


def getfn(typ, model, inds=range(40)):

    fns = [
        rappid_original_data + '/ZTF_MSIP_MODEL{:02d}/ZTF_MSIP_NONIaMODEL0-{:04d}_{:s}.FITS.gz'.format(
            model, i+1, typ
        )
        for i in inds
    ]

    return fns


def write_model_to_pickle(model_number, peak_mag=None):

    logger.info(f'writing model file for model {model_number:02d} to pickle. Peak Mag is {peak_mag}')

    hfiles = getfn('HEAD', model_number)
    pfiles = getfn('PHOT', model_number)

    lcs, heads = read_lc(hfiles, pfiles)

    sim = {'meta':
               {'t0': [],
                'z': [],
                'photo_z': [],
                'photo_z_err': [],
                'peak_mag_r': [],
                'peak_mag_g': [],
                'idx_orig': [],
                'lumdist': [],
                'hostebv': [],
                'model': [],
                'model_type': [],
                't_peak': []},
           'lcs': []}
    for head, lc in tqdm.tqdm(list(zip(heads, lcs)), desc='format lightcurves',
                              leave=False, file=tqdm_info, mininterval=5):

        peak_mag_r, peak_mag_g = head[3], head[2]

        if peak_mag:
            if min((peak_mag_g, peak_mag_r)) > peak_mag:
                continue

        sim['meta']['t0'].append(None)
        sim['meta']['t_peak'].append(head[6])

        sim['meta']['z'].append(head[4])
        sim['meta']['photo_z'].append(head[11])
        sim['meta']['photo_z_err'].append(head[12])

        sim['meta']['peak_mag_r'].append(head[3])
        sim['meta']['peak_mag_g'].append(head[2])

        sim['meta']['idx_orig'].append(int(head[1]))
        sim['meta']['lumdist'].append(head[5])
        sim['meta']['hostebv'].append(head[7])
        sim['meta']['model'].append(head[-1])
        sim['meta']['model_type'].append('mosfit' if int(head[0]) == 0 else
                                         'template' if int(head[0]) == 3 else
                                         None)
        pbnew = [f'ztf{thisb}' for thisb in lc['pb']]
        newlc = Table([lc['mjd'], lc['flux'], lc['dflux'], pbnew, lc['zpt'], ['ab'] * len(lc)],
                      names=['time', 'flux', 'fluxerr', 'band', 'zp', 'zpsys'])

        sim['lcs'].append(newlc)

    fname = rappid_pkl_name(model_number, peak_mag)
    logger.info(f'writing {len(sim["lcs"])} lightcurves to {fname}')
    with open(fname, 'wb+') as fout:
        pickle.dump(sim, fout)


if __name__ == '__main__':

    logger = get_custom_logger(main_logger_name)
    logger.setLevel(logging.DEBUG)
    logger.debug('logging level is DEBUG')

    write_model_to_pickle(3)
    write_model_to_pickle(13)
