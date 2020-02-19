import estimate_explosion_time.shared
import sncosmo
import os
from numpy import loadtxt


ztf_bandpass_dir = os.environ['ZTF_BANDPASS_DIR']

for f in ['g', 'r', 'i']:
    dat = loadtxt(ztf_bandpass_dir + '/ztf' + f + '.txt')
    band = sncosmo.Bandpass(dat[:, 0], dat[:, 1], name='ztf' + f)
    sncosmo.registry.register(band)
