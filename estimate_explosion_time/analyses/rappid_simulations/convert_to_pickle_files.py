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


def rappid_pkl_name(model_number):
    return rappid_original_data + '/ZTF_MSIP_MODEL{0}.pkl'.format(
        model_number if type(model_number) is str else '{:02d}'.format(model_number)
    )


def getfn(typ, model, inds=range(40)):

    fns = [
        rappid_original_data + '/ZTF_MSIP_MODEL{:02d}/ZTF_MSIP_NONIaMODEL0-{:04d}_{:s}.FITS.gz'.format(
            model, i+1, typ
        )
        for i in inds
    ]

    return fns


def write_model_to_pickle(model_number):

    logger.info('writing model file for model {:02d} to pickle'.format(model_number))

    hfiles = getfn('HEAD', model_number)
    pfiles = getfn('PHOT', model_number)

    lcs, heads = read_lc(hfiles, pfiles)

    sim = {'meta':
               {'t0': [],
                'z': [],
                'idx_orig': [],
                'dlum': [],
                'hostebv': [],
                'model': [],
                'model_type':[],
                't_peak': []},
           'lcs': []}
    for head, lc in tqdm.tqdm(list(zip(heads, lcs)), desc='format lightcurves',
                              leave=False, file=tqdm_info, mininterval=5):

        sim['meta']['t0'].append(None)
        sim['meta']['t_peak'].append(head[6])
        sim['meta']['z'].append(head[4])
        sim['meta']['idx_orig'].append(int(head[1]))
        sim['meta']['dlum'].append(head[5])
        sim['meta']['hostebv'].append(head[7])
        sim['meta']['model'].append(head[-1])
        sim['meta']['model_type'].append('mosfit' if int(head[0]) == 0 else
                                         'template' if int(head[0]) == 3 else
                                         None)
        pbnew = [f'ztf{thisb}' for thisb in lc['pb']]
        newlc = Table([lc['mjd'], lc['flux'], lc['dflux'], pbnew, lc['zpt'], ['ab'] * len(lc)],
                      names=['time', 'flux', 'fluxerr', 'band', 'zp', 'zpsys'])

        sim['lcs'].append(newlc)

    with open(rappid_pkl_name(model_number), 'wb+') as fout:
        pickle.dump(sim, fout)


if __name__ == '__main__':

    logger = get_custom_logger(main_logger_name)
    logger.setLevel(logging.DEBUG)
    logger.debug('logging level is DEBUG')

    write_model_to_pickle(3)
    write_model_to_pickle(13)
