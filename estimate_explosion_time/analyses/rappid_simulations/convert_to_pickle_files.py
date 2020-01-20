from estimate_explosion_time.analyses.rappid_simulations.read_light_curves_from_snana_fits \
    import read_light_curves_from_snana_fits_files as read_lc
import tqdm
from astropy.table import Table
import pickle


def getfn(typ, model, inds=range(40)):
    fns = ['../ZTF_20190512/ZTF_MSIP_MODEL{:02d}/ZTF_MSIP_NONIaMODEL0-{:04d}_{:s}.FITS.gz'.format(model, i+1, typ)
           for i in inds]
    return fns


def write_model_to_pickle(model_number):

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
                'texp': []},
           'lcs': []}
    for head, lc in tqdm.tqdm(list(zip(heads, lcs)), desc='format lightcurves', leave=False):
        sim['meta']['t0'].append(head[6])
        sim['meta']['texp'].append(None)
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

    pklname = '../ZTF_20190512/ZTF_MSIP_MODEL{:02d}.pkl'.format(model_number)
    with open(pklname, 'wb+') as fout:
        pickle.dump(sim, fout)
