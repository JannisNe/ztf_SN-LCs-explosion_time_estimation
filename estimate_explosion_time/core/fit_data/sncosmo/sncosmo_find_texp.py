import sncosmo
import numpy as np
import json
import sncosmo_register_ztf_bands
import sncosmo_utils
import get_texp_from_spectral_timeseries
from tqdm import tqdm

outfile_name = 'sncosmo_texp.json'
with open('sncosmo_model_names.json', 'rb') as f:
    model_names = json.load(f)

t0 = {}
for name in tqdm(model_names, desc='fitting templates'):
    source = sncosmo.get_source(name)
    if 'nugent' in name: t0[name] = 0
    else: t0[name] = get_texp_from_spectral_timeseries.get_texp(source)[1]
    # if "snana" in name:
    #     t0[name] = sncosmo_utils.find_t0(source)
    #     # band = 'ztfg'
    #     # time = np.linspace(source.minphase(), source.minphase()+30, 100000)
    #     # model = sncosmo.Model(source)
    #     # amp = 1e-9
    #     # model.set(z=0, t0=0, amplitude=amp)
    #     # flux = model.bandflux(band, time)
    #     # t0[name] = time[np.where(flux/amp > 1e10)[0][0]]
    #     if source.name in ['snana-2007nc', 'snana-2005hm']: t0[name] = -19
    # else:
    #     t0[name] = source.minphase()
with open(outfile_name, 'w') as f:
    json.dump(t0, f)